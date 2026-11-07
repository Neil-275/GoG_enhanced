import torch
import torch.nn as nn
try:
    from torch_scatter import scatter
except ImportError:
    def scatter(src, index, dim=0, dim_size=None, reduce='sum'):
        """Small sum-scatter fallback for environments without torch-scatter."""
        if dim != 0 or reduce != 'sum':
            raise NotImplementedError('fallback scatter supports dim=0, reduce=sum only')
        size = int(dim_size) if dim_size is not None else int(index.max()) + 1
        out = src.new_zeros((size,) + src.shape[1:])
        return out.index_add_(0, index, src)

class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x:x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.rela_embed = nn.Embedding(2*n_rel+1, in_dim)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha  = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
    
    def forward(self, q_rel, r_idx, hidden, edges, n_node, shortcut=False):
        # edges: [h, r, t]
        sub = edges[:,0]
        rel = edges[:,1]
        obj = edges[:,2]
        hs = hidden[sub]
        # Manual sampler edges use two structural relation IDs beyond the learned
        # relation table. Map them to the existing structural/self-loop embedding
        # so checkpoint parameter shapes remain unchanged.
        hr = self.rela_embed(rel.clamp_max(2 * self.n_rel)) # relation embedding of each edge
        h_qr = self.rela_embed(q_rel)[r_idx] # use batch_idx to get the query relation
        
        # message aggregation
        message = hs * hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message        
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum') #ori
        
        # get new hidden representations
        hidden_new = self.act(self.W_h(message_agg))
        
        if shortcut: hidden_new = hidden_new + hidden
        
        return hidden_new

class GNN_auto(torch.nn.Module):
    def __init__(self, params):
        super(GNN_auto, self).__init__()
        self.params = params
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.n_ent = params.n_ent
        # self.loader = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)
        
        if self.params.initializer == 'relation': self.query_rela_embed = nn.Embedding(2*self.n_rel+1, self.hidden_dim)
        if self.params.readout == 'linear':
            if self.params.concatHidden:
                self.W_final = nn.Linear(self.hidden_dim * (self.n_layer+1), 1, bias=False)
            else:
                self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        
    def forward(self, q_sub, q_rel, subgraph_data, mode='train'):
        ''' forward with extra propagation '''
        n = len(q_sub)
        batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges = subgraph_data[:5]
        node_ptr = subgraph_data[5] if len(subgraph_data) > 5 else None
        n_node = len(batch_idxs)
        param = next(self.parameters())
        h0 = torch.zeros((1, n_node, self.hidden_dim), device=param.device, dtype=param.dtype)
        hidden = torch.zeros((n_node, self.hidden_dim), device=param.device, dtype=param.dtype)
        
        # initialize the hidden
        if self.params.initializer == 'binary':
            hidden[query_sub_idxs, :] = 1
        elif self.params.initializer == 'relation':
            hidden[query_sub_idxs, :] = self.query_rela_embed(q_rel)
        
        # store hidden at each layer or not
        if self.params.concatHidden: hidden_list = [hidden]
        
        # propagation
        for i in range(self.n_layer):
            # forward
            hidden = self.gnn_layers[i](q_rel, edge_batch_idxs, hidden, batch_sampled_edges, n_node,
                                        shortcut=self.params.shortcut)
            
            # act_signal is a binary (0/1) tensor 
            # that 1 for non-activated entities and 0 for activated entities
            act_signal = (hidden.sum(-1) == 0).detach().int()
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)
            hidden = hidden * (1-act_signal).unsqueeze(-1)
            h0 = h0 * (1-act_signal).unsqueeze(-1).unsqueeze(0)
            
            if self.params.concatHidden: hidden_list.append(hidden)

        # readout
        if self.params.readout == 'linear':
            if self.params.concatHidden: hidden = torch.cat(hidden_list, dim=-1)
            scores = self.W_final(hidden).squeeze(-1)        
        elif self.params.readout == 'multiply':
            if self.params.concatHidden: hidden = torch.cat(hidden_list, dim=-1)
            scores = torch.sum(hidden * hidden[query_sub_idxs][batch_idxs], dim=-1)
        
        if getattr(self.params, 'local_ppr', False):
            if node_ptr is None:
                counts = torch.bincount(batch_idxs, minlength=n)
                node_ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
            return {'node_scores': scores, 'abs_idxs': abs_idxs, 'node_ptr': node_ptr}

        # legacy dense re-indexing
        scores_all = scores.new_zeros((n, self.params.n_ent))
        scores_all[batch_idxs, abs_idxs] = scores

        return scores_all

    @torch.no_grad()
    def inference(self, q_sub, q_rel, subgraph_data, topk=None):
        """Inference wrapper around forward().

        Args:
            q_sub: LongTensor of shape [n] (or compatible) with query subjects.
            q_rel: LongTensor of shape [n] with query relations.
            subgraph_data: tuple(batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges)
            topk: if provided, returns (topk_scores, topk_indices) instead of full score matrix.

        Returns:
            If topk is None: scores_all of shape [n, n_ent].
            Else: (topk_scores, topk_indices) of shape [n, topk].
        """
        self.eval()
        device = next(self.parameters()).device

        if isinstance(q_sub, torch.Tensor):
            q_sub = q_sub.to(device)
        if isinstance(q_rel, torch.Tensor):
            q_rel = q_rel.to(device)

        batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges = subgraph_data[:5]
        node_ptr = subgraph_data[5] if len(subgraph_data) > 5 else None
        batch_idxs = batch_idxs.to(device)
        abs_idxs = abs_idxs.to(device)
        query_sub_idxs = query_sub_idxs.to(device)
        edge_batch_idxs = edge_batch_idxs.to(device)
        batch_sampled_edges = batch_sampled_edges.to(device)

        output = self.forward(
            q_sub,
            q_rel,
            (batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges, node_ptr),
            mode='test',
        )
        if topk is None:
            return output
        if isinstance(output, dict):
            all_values, all_ids = [], []
            for i in range(len(output['node_ptr']) - 1):
                start, end = int(output['node_ptr'][i]), int(output['node_ptr'][i + 1])
                scores, ids = output['node_scores'][start:end], output['abs_idxs'][start:end]
                order = sorted(range(len(scores)), key=lambda j: (-float(scores[j]), int(ids[j])))
                take = torch.as_tensor(order[:min(int(topk), len(order))], device=scores.device)
                all_values.append(scores[take]); all_ids.append(ids[take])
            if len(all_values) == 1:
                return all_values[0].unsqueeze(0), all_ids[0].unsqueeze(0)
            return all_values, all_ids
        return torch.topk(output, k=min(int(topk), output.shape[1]), dim=1)
