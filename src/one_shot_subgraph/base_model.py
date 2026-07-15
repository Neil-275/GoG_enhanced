import os
import torch
import numpy as np
import time
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from model import *
from utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict
import torch.nn.functional as F
from torch_scatter import scatter
import copy
from torch.utils.data import Subset

class BaseModel(object):
    def __init__(self, args, loaders, samplers):
        self.args = args
        loader, val_loader, test_loader = loaders
        # scoring_mode controls global vs local/candidate-only scoring
        self.scoring_mode = getattr(args, 'scoring_mode', 'global')
        self.dataset = getattr(args, 'dataset', '')
        self.args = args
        self.args.scoring_mode = self.scoring_mode

        self.loader = loader
        self.model = GNN_auto(args)
        self.model.cuda()
        self.n_ent = loader.n_ent
        self.n_samp_ent = args.n_samp_ent
        self.n_rel = loader.n_rel
        self.train_sampler, self.test_sampler = samplers
        self.trainLoader = DataLoader(loader, batch_size=args.n_batch, num_workers=0, collate_fn=loader.collate_fn, shuffle=False, pin_memory=False)
        self.valLoader = DataLoader(val_loader, batch_size=108, num_workers=0, collate_fn=val_loader.collate_fn, shuffle=False, pin_memory=False)
        self.testLoader = DataLoader(test_loader, batch_size=108, num_workers=0, collate_fn=test_loader.collate_fn, shuffle=False, pin_memory=False)
        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2, min_lr=args.lr/20)
        self.smooth = 1e-5
        self.t_time = 0
        self.mean_rank_dict = {}
            
    def saveModelToFiles(self, args, best_metric, deleteLastFile=True):
        output_path = getattr(self.args, 'output_path', self.args.data_path)
        if args.val_num == -1:
            savePath = os.path.join(
                output_path,
                'saveModel',
                f'topk_{self.args.topk}_layer_{self.args.layer}_{best_metric}.pt'
            )
        else:
            savePath = os.path.join(
                output_path,
                'saveModel',
                f'topk_{self.args.topk}_layer_{self.args.layer}_valNum_{self.args.val_num}_{best_metric}.pt'
            )
            
        print(f'Save checkpoint to : {savePath}')
        torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_mrr':best_metric,
                }, savePath)
        
    def _merge_optimizer_state_dict(self, checkpoint_opt_state, current_opt_state):
        """
        Merge checkpoint optimizer state with current optimizer state,
        using current values for missing parameters
        """
        merged_state = {}
        
        # Handle param_groups
        if 'param_groups' in checkpoint_opt_state:
            merged_param_groups = []
            for i, checkpoint_group in enumerate(checkpoint_opt_state['param_groups']):
                # Start with current group settings
                current_group = current_opt_state['param_groups'][i].copy()
                
                # Override with checkpoint values where available
                for key, value in checkpoint_group.items():
                    current_group[key] = value
                    
                merged_param_groups.append(current_group)
            merged_state['param_groups'] = merged_param_groups
        else:
            merged_state['param_groups'] = current_opt_state['param_groups']
        
        # Handle state (momentum buffers, etc.)
        if 'state' in checkpoint_opt_state:
            merged_state['state'] = checkpoint_opt_state['state']
        else:
            merged_state['state'] = current_opt_state.get('state', {})
            
        return merged_state

    def loadModel(self, filePath):
        print(f'Load weight from {filePath}')
        assert os.path.exists(filePath)
        checkpoint = torch.load(filePath, map_location=torch.device(f'cuda:{self.args.gpu}'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            print('✓ Restoring optimizer state from checkpoint')
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print('⚠ No optimizer state found, creating new optimizer')
            # re-build optimizer
            self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.lamb)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2, min_lr=self.args.lr/20)
        
    def prepareData(self, batch_data):
        subs, rels, objs, batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges = batch_data
        subgraph_data = [batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs.cuda(), batch_sampled_edges.cuda()]
        subs = subs.cuda().flatten()
        rels = rels.cuda().flatten()
        objs = objs.cuda()
        return subs, rels, objs, subgraph_data

    @torch.no_grad()
    def predict_topk(
        self,
        entity_id: int,
        relation_id: int,
        topk: int = 10,
        sampler=None,
        use_test_sampler: bool = True,
    ):
        """Run inference for a single query (entity_id, relation_id).

        This extracts the query subgraph using the existing sampler, feeds it into the GNN,
        and returns the top-k predicted tail entity ids.

        Args:
            entity_id: head entity id (global id in [0, n_ent)).
            relation_id: relation id (global id; can include inverse if your dataset uses it).
            topk: number of predictions to return.
            use_test_sampler: if True uses self.test_sampler (typically built from all triples);
                otherwise uses self.train_sampler (typically built from fact edges).

        Returns:
            (scores, entity_ids): both are 1D CPU tensors of shape [topk].
        """
        if sampler is None:
            sampler = self.test_sampler if use_test_sampler else self.train_sampler

        # Construct a single-item "batch" subgraph
        subgraph = sampler.getOneSubgraph(int(entity_id))
        subgraph_data = sampler.getBatchSubgraph([subgraph])

        q_sub = torch.tensor([int(entity_id)], dtype=torch.long)
        q_rel = torch.tensor([int(relation_id)], dtype=torch.long)

        values, indices = self.model.inference(q_sub, q_rel, subgraph_data, topk=int(topk))

        # values/indices are shape [1, topk]
        return values[0].detach().cpu(), indices[0].detach().cpu()
        
    def train_batch(self,):        
        # ov_str = ""
        epoch_loss = 0
        reach_tails_list = []
        t_time = time.time()
        self.model.train()
        for batch_data in tqdm(self.trainLoader, ncols=50, leave=False):                      
            subs, rels, objs, subgraph_data = self.prepareData(batch_data)
            
            # forward
            self.model.zero_grad()

            if self.scoring_mode != 'local':
                scores = self.model(subs, rels, subgraph_data)
                # global softmax loss (numerically stable)
                pos_scores = scores[[torch.arange(len(scores)).cuda(), objs.flatten()]]
                max_n = torch.max(scores, 1, keepdim=True)[0]
                loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1)))
                # cover tail entity or not
                reach_tails = (pos_scores == 0).detach().int().reshape(-1).cpu().tolist()
                reach_tails_list += reach_tails
            else:
                # Local scoring: candidate-only logits + OTHER logit per query
                node_scores, other_logit = self.model(subs, rels, subgraph_data)
                batch_idxs, abs_idxs, _, _, _, node_ptr = subgraph_data
                if node_ptr is None:
                    counts = torch.bincount(batch_idxs, minlength=len(subs))
                    node_ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)], dim=0)

                tails = objs.flatten().long()
                per_q_losses = []
                for i in range(int(len(subs))):
                    start = int(node_ptr[i].item())
                    end = int(node_ptr[i + 1].item())
                    cand_logits = node_scores[start:end]
                    cand_abs = abs_idxs[start:end]
                    K = int(cand_logits.numel())
                    if K == 0:
                        # No candidates sampled: force OTHER
                        logits_i = other_logit[i].view(1, 1)
                        target_i = torch.zeros(1, dtype=torch.long, device=logits_i.device)
                        per_q_losses.append(F.cross_entropy(logits_i, target_i))
                        continue

                    match = (cand_abs == tails[i]).nonzero(as_tuple=False)
                    if match.numel() > 0:
                        target_idx = int(match[0, 0].item())
                        reach_tails_list.append(0)
                    else:
                        target_idx = K  # OTHER
                        reach_tails_list.append(1)

                    logits_i = torch.cat([cand_logits, other_logit[i:i+1]], dim=0).view(1, K + 1)
                    target_i = torch.tensor([target_idx], dtype=torch.long, device=logits_i.device)
                    per_q_losses.append(F.cross_entropy(logits_i, target_i))

                loss = torch.stack(per_q_losses).sum()

            # backward
            loss.backward()
            self.optimizer.step()

            # avoid NaN
            # for p in self.model.parameters():
            #     X = p.data.clone()
            #     flag = X != X
            #     X[flag] = np.random.random()
            #     p.data.copy_(X)

            # cover tail entity or not
            reach_tails = (pos_scores == 0).detach().int().reshape(-1).cpu().tolist()
            reach_tails_list += reach_tails
            epoch_loss += loss.item()
        
        self.t_time += time.time() - t_time
        
        # evaluate on val/test set
        valid_mrr, out_str = self.evaluate(eval_train=False)    
        self.scheduler.step(valid_mrr)
        
        # shuffle train set
        if self.args.not_shuffle_train:
            pass
        else:
            self.loader.shuffle_train()
            fact_data = np.concatenate([np.array(self.loader.fact_data), self.loader.idd_data], 0)
            self.train_sampler.updateEdges(fact_data)
        
        return valid_mrr, out_str
    
    @torch.no_grad()
    def evaluate(self, eval_train=False, eval_val=True, eval_test=True, verbose=False, rank_CR=False, mean_rank=False):
        self.model.eval()
        i_time = time.time()
        
        # eval on train set
        if eval_train:
            print("evaluating on train set...")
            ranking = []
            stop = 0
            train_reach_tails_list = []
            if mean_rank: mean_rank_list = []
            for batch_data in tqdm(self.trainLoader, ncols=50, leave=False):      
                # prepare data            
                subs, rels, objs, subgraph_data = self.prepareData(batch_data)
                
                # forward
                scores = self.model(subs, rels, subgraph_data, mode='train')  # keep on GPU
                
                # calculate rank - train mode has different obj format, all on GPU
                objs = objs.flatten()  # flatten to get single target indices
                batch_size = scores.size(0)
                
                # Create ranking for each query in the batch on GPU
                for i in range(batch_size):
                    # Get filter for this specific query
                    filt = self.loader.filters[(subs[i].item(), rels[i].item())]
                    filt_1hot = torch.zeros(self.n_ent, device=scores.device)
                    filt_1hot[list(filt)] = 1
                    
                    # Calculate rank for single target entity on GPU
                    target_score = scores[i, objs[i]]
                    # Count how many entities score higher (excluding filtered entities)
                    higher_scores = scores[i] > target_score
                    higher_scores = higher_scores & (1 - filt_1hot).bool()
                    rank = torch.sum(higher_scores).item() + 1
                    ranking.append(rank)
                    
                    if mean_rank:
                        mean_rank_list.append(rank)

                # cover tails or not - adapted for train mode, on GPU
                for i in range(batch_size):
                    target_score = scores[i, objs[i]]
                    reach_tail = 1 if target_score.item() == 0 else 0
                    train_reach_tails_list.append(reach_tail)
                    
                stop += 1
                if stop == 15:
                    break

            ranking = np.array(ranking)
            tr_mrr, tr_h1, tr_h10 = cal_performance(ranking)
            
            if rank_CR:
                target_rank = torch.Tensor(ranking).reshape(-1)
                rank_thre = [int(i/100 * self.loader.n_ent) for i in range(1,101)]
                rank_CR = []
                for thre in rank_thre:
                    ratio = torch.sum((target_rank <= thre).int()) / len(target_rank)
                    rank_CR.append(float(ratio))
                print('Train set:\n', rank_CR)
        
            # save mean rank
            if mean_rank: self.mean_rank_dict['train'] = copy.deepcopy(mean_rank_list)
                
        else:
            tr_mrr, tr_h1, tr_h10 = -1, -1, -1
        
        # eval on val set
        if eval_val:
            print("evaluating on val set...")
            ranking = []
            val_reach_tails_list = []
            if mean_rank: mean_rank_list = []
            for batch_data in tqdm(self.valLoader, ncols=50, leave=False):      
                # prepare data            
                subs, rels, objs, subgraph_data = self.prepareData(batch_data)
                
                # forward
                scores = self.model(subs, rels, subgraph_data, mode='valid')  # keep on GPU
                # global scores for all entities
                # calculate rank on GPU
                batch_size = scores.size(0)
                filters = []
                for i in range(batch_size): # create ground truth filters
                    filt = self.loader.filters[(subs[i].item(), rels[i].item())]
                    filt_1hot = torch.zeros(self.n_ent, device=scores.device)
                    filt_1hot[list(filt)] = 1 # answers are marked as 1, others 0
                    filters.append(filt_1hot)
                filters = torch.stack(filters)  # [batch_size, n_ent]
                
                # Calculate ranks on GPU using cal_ranks equivalent
                ranks = []
                for i in range(batch_size):
                    # Get target entities for this query (multi-hot format)
                    target_entities = torch.nonzero(objs[i]).squeeze(-1)
                    query_ranks = []
                    
                    for target_ent in target_entities:
                        target_score = scores[i, target_ent]
                        # Count entities with higher scores (excluding filtered)
                        higher_scores = scores[i] > target_score
                        higher_scores = higher_scores & (1 - filters[i]).bool()
                        rank = torch.sum(higher_scores).item() + 1
                        query_ranks.append(rank)
                    
                    # Use minimum rank (best) for this query
                    ranks.extend(query_ranks)
                    if mean_rank:
                        mean_rank_list.extend(query_ranks)

                ranking += ranks

                # cover tails or not - on GPU
                for i in range(batch_size):
                    target_entities = torch.nonzero(objs[i]).squeeze(-1)
                    for target_ent in target_entities:
                        target_score = scores[i, target_ent]
                        reach_tail = 1 if target_score.item() == 0 else 0
                        val_reach_tails_list.append(reach_tail)

            ranking = np.array(ranking)
            v_mrr, v_h1, v_h10 = cal_performance(ranking)
            # print(f'[val]  covering tail ratio: {len(val_reach_tails_list)}, {1 - sum(val_reach_tails_list) / len(val_reach_tails_list)}')
            
            if rank_CR:
                target_rank = torch.Tensor(ranking).reshape(-1)
                rank_thre = [int(i/100 * self.loader.n_ent) for i in range(1,101)]
                rank_CR = []
                for thre in rank_thre:
                    ratio = torch.sum((target_rank <= thre).int()) / len(target_rank)
                    rank_CR.append(float(ratio))
                print('Val set:\n', rank_CR)
                
            # save mean rank
            if mean_rank: self.mean_rank_dict['val'] = copy.deepcopy(mean_rank_list)
                
        else:
            v_mrr, v_h1, v_h10 = -1, -1, -1
        
        # eval on test set
        if eval_test:
            print("evaluating on test set...")
            ranking = []
            test_reach_tails_list = []
            if mean_rank: mean_rank_list = []
            for batch_data in tqdm(self.testLoader, ncols=50, leave=False):        
                # prepare data            
                subs, rels, objs, subgraph_data = self.prepareData(batch_data)
                
                # forward
                scores = self.model(subs, rels, subgraph_data, mode='test')  # keep on GPU

                # calculate rank on GPU
                batch_size = scores.size(0)
                filters = []
                for i in range(batch_size):
                    filt = self.loader.filters[(subs[i].item(), rels[i].item())]
                    filt_1hot = torch.zeros(self.n_ent, device=scores.device)
                    filt_1hot[list(filt)] = 1
                    filters.append(filt_1hot)
                filters = torch.stack(filters)  # [batch_size, n_ent]
                
                # Calculate ranks on GPU
                ranks = []
                for i in range(batch_size):
                    # Get target entities for this query (multi-hot format)
                    target_entities = torch.nonzero(objs[i]).squeeze(-1)
                    query_ranks = []
                    
                    for target_ent in target_entities:
                        target_score = scores[i, target_ent]
                        # Count entities with higher scores (excluding filtered)
                        higher_scores = scores[i] > target_score
                        higher_scores = higher_scores & (1 - filters[i]).bool()
                        rank = torch.sum(higher_scores).item() + 1
                        query_ranks.append(rank)
                    
                    # Use all ranks for this query
                    ranks.extend(query_ranks)
                    if mean_rank:
                        mean_rank_list.extend(query_ranks)

                ranking += ranks

                # cover tails or not - on GPU
                for i in range(batch_size):
                    target_entities = torch.nonzero(objs[i]).squeeze(-1)
                    for target_ent in target_entities:
                        target_score = scores[i, target_ent]
                        reach_tail = 1 if target_score.item() == 0 else 0
                        test_reach_tails_list.append(reach_tail)

            ranking = np.array(ranking)
            t_mrr, t_h1, t_h10 = cal_performance(ranking)
            # print(f'[test] covering tail ratio: {len(test_reach_tails_list)}, {1 - sum(test_reach_tails_list) / len(test_reach_tails_list)}')
            
            if rank_CR:
                target_rank = torch.Tensor(ranking).reshape(-1)
                rank_thre = [int(i/100 * self.loader.n_ent) for i in range(1,101)]
                rank_CR = []
                for thre in rank_thre:
                    ratio = torch.sum((target_rank <= thre).int()) / len(target_rank)
                    rank_CR.append(float(ratio))
                print('Test set:\n', rank_CR)
                
            # save mean rank
            if mean_rank: self.mean_rank_dict['test'] = copy.deepcopy(mean_rank_list)
            
        else:
            t_mrr, t_h1, t_h10 = -1, -1, -1
            
        i_time = time.time() - i_time
        out_str = '[TRAIN] MRR:%.4f H@1:%.4f H@10:%.4f\t [VALID] MRR:%.4f H@1:%.4f H@10:%.4f\t [TEST] MRR:%.4f H@1:%.4f H@10:%.4f \t[TIME] train:%.4f inference:%.4f\n'%(tr_mrr, tr_h1, tr_h10, v_mrr, v_h1, v_h10, t_mrr, t_h1, t_h10, self.t_time, i_time)
        return v_mrr, out_str
