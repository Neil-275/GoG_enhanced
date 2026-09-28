import networkx as nx
import pickle as pkl
import time
import copy
import numpy as np
import torch
import os
import logging
import copy
import multiprocessing as mp
import hashlib
import json
import tempfile
from tqdm import tqdm
from scipy.sparse import csr_matrix, coo_matrix
from collections import defaultdict
from collections import deque

def checkPath(path):
    os.makedirs(path, exist_ok=True)
    return

def _local_push_compute(h, out_degree, indptr, indices, alpha=0.85, epsilon=1e-6):
    ''' pure function version of the Andersen et al. local push algorithm,
    factored out so it can run identically in-process or in a worker process.
    adjacency is a CSR-style pair of numpy arrays (indptr, indices) over the
    homogeneous graph, keyed by contiguous entity id 0..n_ent-1 '''
    p = {}
    r = {h: 1.0}
    queue = deque([h])
    in_queue = {h}
    pushed_nodes = set()
    while queue:
        u = queue.popleft()
        pushed_nodes.add(u)
        in_queue.discard(u)
        ru = r.get(u, 0.0)
        deg_u = int(out_degree[u])
        if deg_u == 0 or ru <= epsilon * deg_u:
            continue
        p[u] = p.get(u, 0.0) + alpha * ru
        push_mass = (1 - alpha) * ru
        share = push_mass / deg_u
        r[u] = 0.0
        for v in indices[indptr[u]:indptr[u + 1]]:
            v = int(v)
            r[v] = r.get(v, 0.0) + share
            if r[v] > epsilon * max(int(out_degree[v]), 1) and v not in in_queue:
                queue.append(v)
                in_queue.add(v)
    
    return p, len(pushed_nodes)

LOCAL_PPR_SCHEMA_VERSION = 1

def _atomic_pickle(path, value):
    fd, tmp = tempfile.mkstemp(prefix='.tmp-', dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, 'wb') as f:
            pkl.dump(value, f, protocol=pkl.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def _compact_scores(payload, n_ent):
    """Normalize old and new cache payloads without densifying sparse data."""
    if isinstance(payload, dict) and payload.get('schema_version') == LOCAL_PPR_SCHEMA_VERSION:
        ids, scores = payload['entity_ids'], payload['scores']
    else:
        if isinstance(payload, tuple) and len(payload) == 2 and not isinstance(payload[0], np.ndarray):
            payload = payload[0]
        if isinstance(payload, dict):
            ids = np.fromiter(payload.keys(), dtype=np.int64, count=len(payload))
            scores = np.fromiter(payload.values(), dtype=np.float32, count=len(payload))
        elif isinstance(payload, tuple) and len(payload) == 2:
            ids, scores = payload
        else:
            dense = np.asarray(payload).reshape(-1)
            ids = np.flatnonzero(dense)
            scores = dense[ids]
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if ids.size != scores.size or np.any(ids < 0) or np.any(ids >= n_ent):
        raise ValueError('Invalid local-PPR cache payload')
    if ids.size:
        order = np.argsort(ids, kind='stable')
        ids, scores = ids[order], scores[order]
        # Last value wins for malformed legacy duplicate entries.
        keep = np.r_[ids[1:] != ids[:-1], True]
        ids, scores = ids[keep], scores[keep]
    return ids, scores

def _compact_payload(scores):
    ids = np.fromiter(scores.keys(), dtype=np.int64, count=len(scores))
    values = np.fromiter(scores.values(), dtype=np.float32, count=len(scores))
    ids, values = _compact_scores((ids, values), max(int(ids.max()) + 1, 1) if ids.size else 1)
    return {'schema_version': LOCAL_PPR_SCHEMA_VERSION, 'entity_ids': ids, 'scores': values}

def _local_push_worker_init(out_degree, indptr, indices, save_path, alpha, epsilon):
    # runs once per worker process; stashes the (read-only, shared) adjacency
    # in globals so it isn't repickled for every task
    global _LP_OUT_DEGREE, _LP_INDPTR, _LP_INDICES, _LP_SAVE_PATH, _LP_ALPHA, _LP_EPSILON
    _LP_OUT_DEGREE = out_degree
    _LP_INDPTR = indptr
    _LP_INDICES = indices
    _LP_SAVE_PATH = save_path
    _LP_ALPHA = alpha
    _LP_EPSILON = epsilon

def _local_push_worker_task(h):
    scores, _ = _local_push_compute(h, _LP_OUT_DEGREE, _LP_INDPTR, _LP_INDICES, _LP_ALPHA, _LP_EPSILON)
    ent_ppr_savePath = os.path.join(_LP_SAVE_PATH, f'{int(h)}.pkl')
    _atomic_pickle(ent_ppr_savePath, _compact_payload(scores))

class pprSampler():
    def __init__(self, n_ent:int, n_rel:int, topk:int, topm:int, homoEdges:list, triples:list, data_path:str, split='train', args=None):
        '''
            args:
            topk: number of sampled nodes for one head entity
            triples: list of triples [(h,r,t)]
            data_path: path to save the ppr/subgraphs files
        '''
        print('==> initializing ppr sampler...')
        self.args = args
        self.n_ent = n_ent
        self.n_samp_ent = args.n_samp_ent
        self.n_rel = n_rel
        self. topk = topk
        self. topm = topm
        self.max_nodes_per_query = int(getattr(args, 'max_nodes_per_query', 1024))
        self.max_edges_per_query = int(getattr(args, 'max_edges_per_query', 10000))
        self.local_ppr_alpha = float(getattr(args, 'local_ppr_alpha', 0.85))
        self.local_ppr_epsilon = float(getattr(args, 'local_ppr_epsilon', 1e-6))
        if self.max_nodes_per_query <= 0:
            raise ValueError('max_nodes_per_query must be positive')
        if self.max_edges_per_query == 0 or self.max_edges_per_query < -1:
            raise ValueError('max_edges_per_query must be -1 or a positive integer')
        if not 0 < self.local_ppr_alpha <= 1 or self.local_ppr_epsilon <= 0:
            raise ValueError('local_ppr_alpha must be in (0, 1] and epsilon must be positive')
        self.cache_hits = self.cache_misses = self.ppr_support_total = 0
        self.sampled_node_counts = []
        self.sampled_edge_counts = []
        self.sampling_seconds = []
        # print(triples[:10])
        self.triples = triples
        self.data_folder = data_path
        self. homoEdges = homoEdges
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        
        # Use matrix method for GPU acceleration if specified
        self.use_gpu_ppr = getattr(args, 'use_gpu_ppr', True)
        # Use localized k-hop PPR when a positive k is provided.
        self.ppr_method = 'local_push' if self.args.local_ppr else ('matrix' if self.use_gpu_ppr else 'nx')

        
        if self.ppr_method == 'matrix':
            print('==> Using GPU-accelerated PPR computation')
            self.homoTrainGraph = self.triplesToNxGraph(self.homoEdges)
            self._build_ppr_matrix_sparse()
        elif self.ppr_method == 'nx':
            print('==> Using CPU-based NetworkX PPR computation')
            self.homoTrainGraph = self.triplesToNxGraph(self.homoEdges)
        elif self.ppr_method == 'local_push':
            print('==> Using localized k-hop PPR computation (Andersen et al. local push algorithm)')
            self.construct_neigbors()
            digest = hashlib.sha256()
            digest.update(np.asarray([self.n_ent], dtype=np.int64).tobytes())
            digest.update(np.asarray(self.neighbor_indptr, dtype=np.int64).tobytes())
            digest.update(np.asarray(self.neighbor_indices, dtype=np.int32).tobytes())
            self.graph_fingerprint = digest.hexdigest()

        if self.ppr_method == 'local_push':
            namespace = (f'local-push-v{LOCAL_PPR_SCHEMA_VERSION}-a{self.local_ppr_alpha:g}'
                         f'-e{self.local_ppr_epsilon:g}-{self.graph_fingerprint}')
            self.ppr_savePath = os.path.join(self.data_folder, 'ppr_scores', split, namespace)
        else:
            self.ppr_savePath = os.path.join(self.data_folder, f'ppr_scores/{split}/')
        checkPath(self.ppr_savePath)
        if self.ppr_method == 'local_push':
            metadata = {'schema_version': LOCAL_PPR_SCHEMA_VERSION, 'algorithm': 'local_push',
                        'graph_fingerprint': self.graph_fingerprint, 'entity_count': self.n_ent,
                        'alpha': self.local_ppr_alpha, 'epsilon': self.local_ppr_epsilon,
                        'split': split}
            metadata_path = os.path.join(self.ppr_savePath, 'metadata.json')
            if not os.path.exists(metadata_path):
                fd, tmp = tempfile.mkstemp(prefix='.metadata-', dir=self.ppr_savePath, text=True)
                with os.fdopen(fd, 'w') as f:
                    json.dump(metadata, f, sort_keys=True)
                os.replace(tmp, metadata_path)
        print(f'==> checking ppr scores for each entity using {self.ppr_method}...')

        topic_entities = getattr(self.args, 'topic_entities', None)

        n_workers = getattr(self.args, 'cpu', 1)
        if self.ppr_method == 'local_push' and topic_entities is None:
            print('==> local-push cache will be populated lazily')
        elif self.ppr_method == 'local_push' and n_workers > 1:
            pending = [h for h in topic_entities
                       if not os.path.exists(os.path.join(self.ppr_savePath, f'{h}.pkl'))]
            print(f'==> computing local-push PPR for {len(pending)}/{self.n_ent} entities '
                  f'using {n_workers} processes...')
            if pending:
                chunksize = max(1, len(pending) // (n_workers * 20))
                with mp.Pool(
                    n_workers,
                    initializer=_local_push_worker_init,
                    initargs=(self.out_degree, self.neighbor_indptr, self.neighbor_indices,
                              self.ppr_savePath, self.local_ppr_alpha, self.local_ppr_epsilon),
                ) as pool:
                    for _ in tqdm(pool.imap_unordered(_local_push_worker_task, pending, chunksize=chunksize),
                                  total=len(pending), ncols=50, leave=False):
                        pass
        else:
            avg_pushed_nodes = 0
            if topic_entities is None:
                topic_entities = range(self.n_ent)
            for h in tqdm(topic_entities, ncols=50, leave=False):
                ent_ppr_savePath = os.path.join(self.ppr_savePath, f'{int(h)}.pkl')
                if os.path. exists(ent_ppr_savePath):
                    pass
                else:
                    # with default setting to generate ppr scores
                    h_ppr_scores, n_pushed = self.generatePPRScoresForOneEntity(h, method=self.ppr_method)
                    avg_pushed_nodes += n_pushed
                    payload = _compact_payload(h_ppr_scores) if self.ppr_method == 'local_push' else h_ppr_scores
                    _atomic_pickle(ent_ppr_savePath, payload)
            if len(topic_entities):
                print(f'Average number of pushed nodes: {avg_pushed_nodes / len(topic_entities):.2f}')
        print('finished.')
        
        # build head to edge_idxs with sparse matrix
        heads, edge_idxs = [h for (h,r,t) in triples], list(range(len(triples)))
        print(len(heads), len(edge_idxs), max(heads) if heads else -1, self.n_ent)
        self.sparseTrainMatrix = csr_matrix((np.ones(len(edge_idxs), dtype=np.int8), (heads, edge_idxs)),
                                            shape=(self.n_ent, len(triples)))

        # change data type and move to GPU
        self.triples = torch.LongTensor(self.triples). to(self.device)

        # clean cache: drop `homoEdges` but keep `homoTrainGraph` in memory
        # (required for k-hop localized PPR). To force dropping the graph
        # set `args.drop_graph = True` when creating the sampler.
        del self.homoEdges
        if getattr(self.args, 'drop_graph', False):
            if hasattr(self, 'homoTrainGraph'):
                del self.homoTrainGraph
        
        print('==> finish sampler initilization.')

    def construct_neigbors(self):
        print('==> Precomputing CSR adjacency index for local-push PPR...')
        # Build the undirected CSR adjacency directly from homoEdges via scipy.sparse,
        # instead of walking the networkx graph node-by-node (which is very slow and
        # memory-heavy as a Python dict-of-lists for graphs with millions of nodes).
        if len(self.homoEdges) > 0:
            edges = np.asarray(self.homoEdges, dtype=np.int64)
            src = np.concatenate([edges[:, 0], edges[:, 1]])
            dst = np.concatenate([edges[:, 1], edges[:, 0]])
        else:
            src = np.zeros(0, dtype=np.int64)
            dst = np.zeros(0, dtype=np.int64)
        data = np.ones(len(src), dtype=np.int8)
        adj = csr_matrix((data, (src, dst)), shape=(self.n_ent, self.n_ent))
        adj.sum_duplicates()
        self.neighbor_indptr = adj.indptr
        self.neighbor_indices = adj.indices.astype(np.int32)
        self.out_degree = np.diff(self.neighbor_indptr)

    def _build_ppr_matrix_sparse(self):
        """
        Builds the sparse PPR transition matrix (D^-1 * A) for GPU computation.
        This replaces the unscalable dense matrix approach.
        """
        print('==> Building SPARSE PPR matrix for GPU computation...')
        
        # 1. Prepare the edge_index and values for the sparse adjacency matrix A
        triples_tensor = torch.LongTensor(self.triples).to(self.device)
        # The sparse tensor needs an edge_index in (2, E) format
        # edge_index is (head, tail) pairs derived from the (h, r, t) triples
        edge_index = triples_tensor[:, [0, 2]].t().contiguous()
        # Values are all 1s (unweighted graph)
        values = torch.ones(edge_index.size(1), device=self.device)

        # 2. Create the sparse Adjacency Matrix A (COO format)
        A_sparse = torch.sparse_coo_tensor(edge_index, values, (self.n_ent, self.n_ent)).coalesce()
    
        # 3. Calculate out-degrees (row sums)
        # The rows are the source nodes (heads)
        row_sum = torch.sparse.sum(A_sparse, dim=1).to_dense().flatten()
        row_sum = torch.clamp(row_sum, min=1e-10) # Avoid division by zero
        
        # 4. Calculate D_inv * A (The Transition Matrix P)
        # Since D_inv is diagonal, we can perform the division element-wise on the sparse values.
        # P[i, j] = A[i, j] / D[i, i]
        
        # Get the row indices of the non-zero elements
        A_indices = A_sparse.indices()
        row_indices = A_indices[0]
        
        # Get the inverse degree corresponding to each edge's source node
        row_sum_inv = 1.0 / row_sum
        P_sparse_values = row_sum_inv[row_indices]
        
        # Create the final sparse transition matrix P = D_inv * A
        P_sparse = torch.sparse_coo_tensor(A_indices, P_sparse_values, (self.n_ent, self.n_ent))
        
        # NOTE ON THE ORIGINAL CODE'S FORMULA:
        # The original code defined self.PPR_W = I + D_inv * A.
        # To maintain the structure of your original PageRank iteration formula,
        # we must ensure the `self.PPR_W` used in generatePPRScoresForOneEntity is sparse.
        
        # For a truly sparse implementation, we only set PPR_W = P_sparse.
        self.PPR_W = P_sparse.to(self.device) # This is the sparse transition matrix P
        
        # If the original non-standard formula was strictly required:
        # Adding an identity matrix to a sparse matrix results in a dense matrix again.
        # For true sparsity, you would need to adjust the `generatePPRScoresForOneEntity` 
        # to handle the (1 - alpha) * scores * P term and the I term separately.
        
        print('==> PPR transition matrix (P) built and moved to GPU as a sparse tensor')


    def ppr_local_push(self, h, alpha=0.85, epsilon=1e-6):
        scores, n_pushed = _local_push_compute(
            h, self.out_degree, self.neighbor_indptr, self.neighbor_indices, alpha, epsilon
        )
        return scores, n_pushed

    def updateEdges(self, triples):
        # co-operate with shuffle_train
        if self.ppr_method == 'local_push':
            raise RuntimeError('local-push uses a frozen fact graph; updateEdges is not allowed')
        heads, edge_idxs = [h for (h,r,t) in triples], list(range(len(triples)))
        self.sparseTrainMatrix = csr_matrix((np.ones(len(edge_idxs), dtype=np.int8), (heads, edge_idxs)),
                                            shape=(self.n_ent, len(triples)))
        self.triples = torch.LongTensor(triples).to(self.device)
    
    def getPPRscores(self, ent):
        ent_ppr_savePath = os.path.join(self.ppr_savePath, f'{int(ent)}.pkl')
        if self.ppr_method == 'local_push':
            try:
                with open(ent_ppr_savePath, 'rb') as f:
                    payload = pkl.load(f)
                ids, scores = _compact_scores(payload, self.n_ent)
                self.cache_hits += 1
            except (FileNotFoundError, EOFError, pkl.UnpicklingError, ValueError, KeyError):
                raw, _ = self.generatePPRScoresForOneEntity(int(ent), method='local_push')
                ids, scores = _compact_scores(raw, self.n_ent)
                _atomic_pickle(ent_ppr_savePath, {'schema_version': LOCAL_PPR_SCHEMA_VERSION,
                                                   'entity_ids': ids, 'scores': scores})
                self.cache_misses += 1
            self.ppr_support_total += len(ids)
            return ids, scores
        scores = pkl.load(open(ent_ppr_savePath, 'rb'))
        # Accept caches written by older local-push implementations.
        if isinstance(scores, tuple):
            scores = scores[0]
        if isinstance(scores, dict):
            dense_scores = np.zeros(self.n_ent, dtype=np.float32)
            dense_scores[list(scores)] = list(scores.values())
            scores = dense_scores
        # print(scores, type(scores))
        return scores
        
    def generatePPRScoresForOneEntity(self, h, method=None):
        if method is None:
            method = 'matrix' if self.use_gpu_ppr else 'nx'
            # print("ahahahaha")
        # support localized PPR via the Andersen et al. local push algorithm
        if method == 'local_push':
            scores_sub, n_pushed = self.ppr_local_push(h, self.local_ppr_alpha, self.local_ppr_epsilon)
            return scores_sub, n_pushed
        if method == 'nx':
            '''
            nx. pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None)
            '''
            scores = nx.pagerank(self. homoTrainGraph, personalization={h: 1})
            n_pushed = len(scores)
        elif method == 'matrix':
            alpha, iteration = 0.85, 100
            scores = torch.zeros(1, self.n_ent). to(self.device)
            s = torch.zeros(1, self.n_ent).to(self.device)
            s[0, h] = 1
            for i in range(iteration):
                scores = alpha * s + (1 - alpha) * torch.matmul(scores, self.PPR_W)
            scores = scores. cpu().reshape(-1). numpy()
            n_pushed = int(np.count_nonzero(scores))
        return scores, n_pushed
    
    def triplesToNxGraph(self, edges):
        ''' edges is the list of [(h,t)] '''
        graph = nx.Graph()
        nodes = list(range(self.n_ent))
        graph.add_nodes_from(nodes)        
        graph.add_edges_from(edges)
        return graph
    
    def sampleSubgraph(self, ent: int, cand=None):    
        if self.ppr_method == 'local_push':
            return self._sample_local_subgraph(ent, cand)
        # sample subgraph to get the edges
        # ppr_scores = np.array(list(self.getPPRscores(ent). values()))
        ppr_scores = self.getPPRscores(ent)
        # gurantee the candidates are sampled
        if cand != None and self.topk < self.n_ent:
            tmp_ppr_scores = copy.deepcopy(ppr_scores)
            tmp_ppr_scores[cand] = 1e8
            topk_nodes = sorted(list(set([ent] + np.argsort(tmp_ppr_scores)[::-1][:self.topk]. tolist())))
        else:
            # topk sampling
            if self.topk < self.n_ent:    
                topk_nodes = sorted(list(set([ent] + np.argsort(ppr_scores)[::-1][:self.topk].tolist())))
            else:
                # no sampling
                topk_nodes = list(range(self.n_ent))

        # get candididate edges
        selectd_edges = self.sparseTrainMatrix[topk_nodes, :]	
        _, tmp_edge_index = selectd_edges.nonzero()
        
        # (h,r,t)
        edges = self.triples[tmp_edge_index]
        topk_nodes_tensor = torch.LongTensor(topk_nodes).to(self.device)
        
        # edge sampling
        mask = torch.isin(edges[:,2], topk_nodes_tensor)
        
        # [n_edges, 3]
        sampled_edges = edges[mask, :]
        
        # edge sampling (topm edges for each subgraph)
        edge_num = int(sampled_edges.shape[0])
        # NOTE: if self.topm== 0, then skip edge sampling 
        if self.topm > 0 and edge_num > self.topm:
            # ppr weight
            heads, tails = sampled_edges[:,0]. cpu(), sampled_edges[:,2].cpu()
            edge_weights = ppr_scores[heads] + ppr_scores[tails]
            edge_weights = torch. Tensor(edge_weights).to(self.device)
            index = torch.topk(edge_weights, self. topm).indices
            sampled_edges = sampled_edges[index]
        
        # get node indexing map (keep on CPU for indexing operations)
        topk_nodes = topk_nodes_tensor.cpu()
        node_index = torch.zeros(self.n_ent). long()
        node_index[topk_nodes] = torch.arange(len(topk_nodes))
              
        # connect head to all tails 
        if self.args.add_manual_edges:
            add_edges_head2tails = torch.zeros((len(topk_nodes), 3)). long(). to(self.device)
            add_edges_head2tails[:, 0] = ent
            add_edges_head2tails[:, 1] = 2*self.n_rel + 1
            add_edges_head2tails[:, 2] = topk_nodes_tensor
            add_edges_tails2head = torch.zeros((len(topk_nodes), 3)).long().to(self.device)
            add_edges_tails2head[:, 0] = topk_nodes_tensor
            add_edges_tails2head[:, 1] = 2*self.n_rel + 2
            add_edges_tails2head[:, 2] = ent
            sampled_edges = torch.cat([sampled_edges, add_edges_head2tails, add_edges_tails2head], dim=0)
        
        return topk_nodes, node_index, sampled_edges

    def _sample_local_subgraph(self, ent, cand=None):
        started = time.perf_counter()
        ent = int(ent)
        if not 0 <= ent < self.n_ent:
            raise ValueError('query head is outside the entity vocabulary')
        if cand is not None and not 0 <= int(cand) < self.n_ent:
            raise ValueError('candidate is outside the entity vocabulary')
        ids, scores = self.getPPRscores(ent)
        positive = scores > 0
        ranked_ids, ranked_scores = ids[positive], scores[positive]
        order = np.lexsort((ranked_ids, -ranked_scores))
        mandatory = [ent]
        if cand is not None and int(cand) != ent:
            mandatory.append(int(cand))
        if len(mandatory) > self.max_nodes_per_query:
            raise ValueError('node budget is too small for mandatory head/candidate nodes')
        selected, seen = list(mandatory), set(mandatory)
        for node in ranked_ids[order]:
            node = int(node)
            if node not in seen:
                selected.append(node); seen.add(node)
                if len(selected) == self.max_nodes_per_query:
                    break
        nodes = np.asarray(sorted(selected), dtype=np.int64)
        score_map = dict(zip(ids.tolist(), scores.tolist()))
        rows = self.sparseTrainMatrix[nodes, :].nonzero()[1]
        ordinary = self.triples[torch.as_tensor(rows, device=self.device, dtype=torch.long)]
        if ordinary.numel():
            pos = torch.searchsorted(torch.as_tensor(nodes, device=self.device), ordinary[:, 2].contiguous())
            valid = (pos < len(nodes))
            valid &= torch.as_tensor(nodes, device=self.device)[pos.clamp(max=max(len(nodes)-1, 0))] == ordinary[:, 2]
            ordinary = ordinary[valid]
        manual = []
        if self.args.add_manual_edges:
            for node in nodes.tolist():
                manual.extend(((ent, 2*self.n_rel + 1, node), (node, 2*self.n_rel + 2, ent)))
        manual = torch.as_tensor(manual, dtype=torch.long, device=self.device).reshape(-1, 3)
        cap = self.max_edges_per_query
        if cap != -1 and len(manual) > cap:
            raise ValueError('mandatory manual edges exceed max_edges_per_query')
        ordinary_cap = len(ordinary) if cap == -1 else cap - len(manual)
        if len(ordinary) > ordinary_cap:
            arr = ordinary.detach().cpu().numpy()
            weights = np.asarray([score_map.get(int(h), 0.) + score_map.get(int(t), 0.) for h, _, t in arr])
            order = np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0], -weights))[:ordinary_cap]
            ordinary = ordinary[torch.as_tensor(order, device=self.device)]
        edges = torch.cat((ordinary, manual), dim=0)
        node_tensor = torch.as_tensor(nodes, dtype=torch.long, device=self.device)
        if edges.numel():
            local_h = torch.searchsorted(node_tensor, edges[:, 0])
            local_t = torch.searchsorted(node_tensor, edges[:, 2])
            if not torch.all(node_tensor[local_h] == edges[:, 0]) or not torch.all(node_tensor[local_t] == edges[:, 2]):
                raise ValueError('retained edge endpoint is not sampled')
            edges = torch.stack((local_h, edges[:, 1], local_t), dim=1)
        query_local = int(np.searchsorted(nodes, ent))
        self.sampled_node_counts.append(len(nodes))
        self.sampled_edge_counts.append(len(edges))
        self.sampling_seconds.append(time.perf_counter() - started)
        return torch.as_tensor(nodes), edges, query_local

    def benchmark_stats(self):
        """Return JSON-serializable cache/support and sampling statistics."""
        calls = self.cache_hits + self.cache_misses
        percentiles = lambda values: ({str(p): float(np.percentile(values, p)) for p in (50, 90, 99)}
                                      if values else {})
        return {
            'graph_fingerprint': getattr(self, 'graph_fingerprint', None),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / calls if calls else None,
            'mean_ppr_support': self.ppr_support_total / calls if calls else None,
            'sampled_nodes_percentiles': percentiles(self.sampled_node_counts),
            'sampled_edges_percentiles': percentiles(self.sampled_edge_counts),
            'sampling_latency_ms_percentiles': percentiles([x * 1000 for x in self.sampling_seconds]),
            'max_nodes_per_query': self.max_nodes_per_query,
            'max_edges_per_query': self.max_edges_per_query,
            'local_ppr_alpha': self.local_ppr_alpha,
            'local_ppr_epsilon': self.local_ppr_epsilon,
        }

    def getOneSubgraph(self, head: int, cand=None):
        if self.ppr_method == 'local_push':
            return self.sampleSubgraph(head, cand)
        topk_nodes, node_index, sampled_edges = self.sampleSubgraph(head, cand) 
        return [head, topk_nodes, node_index, sampled_edges]
        
    def getBatchSubgraph(self, subgraph_list: list):  
        if self.ppr_method == 'local_push':
            return self._batch_local_subgraphs(subgraph_list)
        batchsize = len(subgraph_list)
        ent_delta_values = [0]
        batch_sampled_edges = []
        batch_idxs, abs_idxs = [], []
        query_sub_idxs = []
        edge_batch_idxs = []

        for batch_idx in range(batchsize):       
            sub, topk_nodes, node_index, sampled_edges = subgraph_list[batch_idx]
            num_nodes = len(topk_nodes)
            ent_delta = sum(ent_delta_values)

            sampled_edges[:,0] = node_index[sampled_edges[:,0]. cpu()].to(self.device) + ent_delta
            sampled_edges[:,2] = node_index[sampled_edges[:,2].cpu()]. to(self.device) + ent_delta
            batch_sampled_edges.append(sampled_edges)
            edge_batch_idxs += [batch_idx] * int(sampled_edges.shape[0])

            ent_delta_values.append(num_nodes)
            batch_idxs += [batch_idx] * num_nodes
            abs_idxs += topk_nodes.tolist()
            query_sub_idxs. append(int(node_index[sub]) + ent_delta)
        
        # [n_batch_ent]
        batch_idxs = torch.LongTensor(batch_idxs)
        # [n_batch_ent]
        abs_idxs = torch.LongTensor(abs_idxs)
        # [n_batch_edges, 3]
        batch_sampled_edges = torch.cat(batch_sampled_edges, dim=0)
        # [n_batch_edges]
        edge_batch_idxs = torch.LongTensor(edge_batch_idxs)
        # [n_batch]
        query_sub_idxs = torch.LongTensor(query_sub_idxs)
        
        return batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges

    def _batch_local_subgraphs(self, subgraph_list):
        batch_idxs, abs_idxs, query_idxs, edge_batch, edges = [], [], [], [], []
        node_ptr = [0]
        for i, (nodes, local_edges, query_local) in enumerate(subgraph_list):
            offset = node_ptr[-1]
            batch_idxs.extend([i] * len(nodes)); abs_idxs.extend(nodes.tolist())
            query_idxs.append(offset + int(query_local))
            if len(local_edges):
                shifted = local_edges.clone()
                shifted[:, (0, 2)] += offset
                edges.append(shifted); edge_batch.extend([i] * len(shifted))
            node_ptr.append(offset + len(nodes))
        device = self.device
        batch_edges = torch.cat(edges) if edges else torch.empty((0, 3), dtype=torch.long, device=device)
        return (torch.tensor(batch_idxs), torch.tensor(abs_idxs), torch.tensor(query_idxs),
                torch.tensor(edge_batch), batch_edges, torch.tensor(node_ptr))
