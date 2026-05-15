import networkx as nx
import pickle as pkl
import time
import copy
import numpy as np
import torch
import os
import logging
import copy
from tqdm import tqdm
from scipy.sparse import csr_matrix, coo_matrix
from collections import defaultdict

def checkPath(path):
    os.makedirs(path, exist_ok=True)
    return

class pprSampler():
    def __init__(self, n_ent:int, n_rel:int, topk:int, topm:int, homoEdges:list, edge_index:list, data_path:str, split='train', args=None):
        ''' 
            args:
            topk: number of sampled nodes for one head entity 
            edge_index: list of triples [(h,r,t)]
            data_path: path to save the ppr/subgraphs files
        '''
        print('==> initializing ppr sampler...')
        self.args = args
        self.n_ent = n_ent
        self.n_samp_ent = args.n_samp_ent
        self.n_rel = n_rel
        self. topk = topk
        self. topm = topm
        self.edge_index = edge_index
        self.data_folder = data_path
        self. homoEdges = homoEdges
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        
        # Use matrix method for GPU acceleration if specified
        self.use_gpu_ppr = getattr(args, 'use_gpu_ppr', True)
        
        if self.use_gpu_ppr:
            print('==> Using GPU-accelerated PPR computation')
            # Build PPR_W matrix for GPU computation
            self.homoTrainGraph = self.triplesToNxGraph(self.homoEdges)
            self._build_ppr_matrix_sparse()
        else:
            print('==> Using CPU-based NetworkX PPR computation')
            self.homoTrainGraph = self.triplesToNxGraph(self. homoEdges)
        
        self.ppr_savePath = os.path.join(self.data_folder, f'ppr_scores/{split}/')
        checkPath(self.ppr_savePath)
        print('==> checking ppr scores for each entity...')
        
        for h in tqdm(range(self.n_ent), ncols=50, leave=False):
            ent_ppr_savePath = os.path.join(self.ppr_savePath, f'{int(h)}.pkl')
            if os.path. exists(ent_ppr_savePath):
                pass
            else:
                # with default setting to generate ppr scores
                h_ppr_scores = self.generatePPRScoresForOneEntity(h)
                pkl.dump(h_ppr_scores, open(ent_ppr_savePath, 'wb'))
        print('finished.')
        
        # build head to edges with sparse matrix
        heads, edges = [h for (h,r,t) in edge_index], list(range(len(edge_index)))
        print(len(heads), len(edges), max(heads), self.n_ent)
        self.sparseTrainMatrix = csr_matrix((edges, (heads, edges)), shape=(self.n_ent, len(edge_index)))

        # change data type and move to GPU
        self.edge_index = torch.LongTensor(self.edge_index). to(self.device)

        # clean cache
        del self.homoEdges
        if hasattr(self, 'homoTrainGraph'):
            del self.homoTrainGraph
        
        print('==> finish sampler initilization.')
    
    def _build_ppr_matrix_sparse(self):
        """
        Builds the sparse PPR transition matrix (D^-1 * A) for GPU computation.
        This replaces the unscalable dense matrix approach.
        """
        print('==> Building SPARSE PPR matrix for GPU computation...')
        
        # 1. Prepare indices and values for the sparse adjacency matrix A
        edge_index_tensor = torch.LongTensor(self.edge_index).to(self.device)
        # The sparse tensor needs indices in (2, E) format
        # Indices are (head, tail) from the homogeneous graph
        indices = edge_index_tensor[:, [0, 2]].t().contiguous()
        # Values are all 1s (unweighted graph)
        values = torch.ones(indices.size(1), device=self.device)
        
        # 2. Create the sparse Adjacency Matrix A (COO format)
        A_sparse = torch.sparse_coo_tensor(indices, values, (self.n_ent, self.n_ent)).coalesce()
    
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

    def updateEdges(self, edge_index):
        # co-operate with shuffle_train
        heads, edges = [h for (h,r,t) in edge_index], list(range(len(edge_index)))
        self. sparseTrainMatrix = csr_matrix((edges, (heads, edges)), shape=(self. n_ent, len(edge_index)))
        self.edge_index = torch.LongTensor(edge_index).to(self.device)
    
    def getPPRscores(self, ent):
        ent_ppr_savePath = os.path.join(self.ppr_savePath, f'{int(ent)}.pkl')
        scores = pkl.load(open(ent_ppr_savePath, 'rb'))
        # print(scores, type(scores))
        return scores
        
    def generatePPRScoresForOneEntity(self, h, method=None):
        if method is None:
            method = 'matrix' if self.use_gpu_ppr else 'nx'
            # print("ahahahaha")
        if method == 'nx':
            '''
            nx. pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None)
            '''
            scores = nx.pagerank(self. homoTrainGraph, personalization={h: 1})
        elif method == 'matrix':
            alpha, iteration = 0.85, 100
            scores = torch.zeros(1, self.n_ent). to(self.device)
            s = torch.zeros(1, self.n_ent).to(self.device)
            s[0, h] = 1
            for i in range(iteration):
                scores = alpha * s + (1 - alpha) * torch.matmul(scores, self.PPR_W)            
            scores = scores. cpu().reshape(-1). numpy()
        return scores
    
    def triplesToNxGraph(self, edges):
        ''' edges is the list of [(h,t)] '''
        graph = nx.Graph()
        nodes = list(range(self.n_ent))
        graph.add_nodes_from(nodes)        
        graph.add_edges_from(edges)
        return graph
    
    def sampleSubgraph(self, ent: int, cand=None):    
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
        edges = self.edge_index[tmp_edge_index]
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

    def getOneSubgraph(self, head: int, cand=None):
        topk_nodes, node_index, sampled_edges = self.sampleSubgraph(head, cand) 
        return [head, topk_nodes, node_index, sampled_edges]
        
    def getBatchSubgraph(self, subgraph_list: list):  
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