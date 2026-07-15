import math
import pickle as pkl
import random
import pandas as pd
from dotenv import load_dotenv
import torch
from typing import Literal, Optional, List, Dict, Tuple
import numpy as np
from queue import Queue
from ast import literal_eval
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as model_util
from loguru import logger
from pathlib import Path
from sampler_utils import extract_notations, extract_numbers
# import prompt_list

load_dotenv()

# Setup logging
LOGGER = logger
TIMEIT = logger  # For timing operations

SENTENCE_TRANSFORMER_MODEL = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


class SubgraphSampler:
    adj = None
    rel_embs = None

    def __init__(
        self,
        kg_interface,
        args,
    ):
        """
        Initialize SubgraphSampler.
        
        Args:
            graph: Pandas DataFrame with columns [head, relation, tail] (kept for compatibility)
            kg_interface: KGInterface object for knowledge graph operations
            args: Configuration arguments with keys: k_rel, k_cands, threshold, device, cands_lim
        """
        self.args = args
        self.k_rel_org = args.k_rel
        self.k_cands_org = args.k_cands
        self.k_rel = args.k_rel
        self.threshold = args.threshold
        self.k_cands = args.k_cands
        self.cands_lim = args.cands_lim if args else 100

        self.query = None
        subgraph_key = None
        self.answers_id = None
        self._name_cache = {}
        self.visited = set()
        self.expanded_from_nodes = set()

        # Initialize model
        self.model = SENTENCE_TRANSFORMER_MODEL.to(self.args.device if args else 'cpu')

        # Initialize KG interface
        self.kg = kg_interface
        if self.kg is None:
            LOGGER.warning("No KG interface provided. KG operations will not be available.")


        self.drop_edges = set()  # Set of edges to exclude from sampling
        self.GoG_args = {
            'drop_ratio': 0.1  # Default drop ratio for edges
        }

        # Utility dictionaries for ID mapping
        self.id2rel: dict[int, str] = {}
        self.id2name: dict[int, str] = {}
        self.id2mid: dict[int, str] = {}
        self.start_entities: list[str] = []

    def assign_query(
        self,
        query: pd.Series,
        question_key: str = 'question',
        start_entities_key: str = 'q_entity',
        dataset = "family"
    ):
        self.query = query
        self.query_emb = self.model.encode(
            self.query[question_key],
            convert_to_tensor=True
        ).to(self.args.device)

        self.mid2name: dict[str, str] = {}
        self.start_entities = []  # List of entity IDs (MIDs) that are the starting points for subgraph expansion
        self.start_entities = literal_eval(query[start_entities_key])


    # def id2name(self, idx):
    #     """Cached version to avoid repeated dictionary lookups."""
    #     idx = idx.item()
    #     if idx not in self._name_cache:
    #         self._name_cache[idx] = ExpandSubgraph.ent2name[ExpandSubgraph.id2ent[idx]]
    #     return self._name_cache[idx]

    def return_top_k(self, scores):
        k = min(self.k_rel, scores.shape[1])
        scores, sorted_indices = torch.topk(scores, k, largest=True, dim=-1)
        # print("scores:", scores, "sorted_indices:", sorted_indices)
        return sorted_indices[0].tolist()

    def convert_relation(self, rel: str) -> str:
        """Convert relation name to readable format"""
        if self.kg.dataset_name == "family":
            return rel
        if self.kg.dataset_name == "fb15k_237":
            return ("_").join(rel.split("/")[1:])
        # return (" ").join(rel.split("/")[1:]) if "/" in rel else rel.replace("_", " ")

    def prune_cands(self, np_triplets: np.array, rel_scores):
        if len(np_triplets) == 0:
            return np_triplets

        # 1. Convert scores safely
        scores = rel_scores.detach().cpu().numpy() if hasattr(rel_scores, 'numpy') else np.array(rel_scores)
        
        # 2. Identify the true 'target' for each triplet without mutating the original array
        # We determine if we look at the tail (col 2) or the head (col 0) based on expansion
        target_ids = []
        for head, rel, tail in np_triplets:
            if tail in self.expanded_from_nodes:
                target_ids.append(head)
            else:
                target_ids.append(tail)
        
        target_ids = np.array(target_ids)
        
        # 3. Aggregate scores per target entity
        entity_scores = {}
        # If you want to ensure a relation is only counted once per entity:
        entity_seen_rels = {} 

        for i in range(len(np_triplets)):
            target = target_ids[i]
            rel = np_triplets[i, 1]
            score = scores[i]
            
            if target not in entity_scores:
                entity_scores[target] = 0.0
                entity_seen_rels[target] = set()
                
            if rel not in entity_seen_rels[target]:
                entity_scores[target] += score
                entity_seen_rels[target].add(rel)

        # 4. Get the top K entities
        # Sorting a dict by value
        sorted_entities = sorted(entity_scores, key=entity_scores.get, reverse=True)
        top_entities = set(sorted_entities[:min(self.k_cands, len(sorted_entities))])

        # 5. Create mask based on our calculated target_ids
        mask = np.isin(target_ids, list(top_entities))
        
        return np_triplets[mask]

    def compare_rel_query_and_return_topk(self, triplets_rel: list):
        # Extract edges for output and rel_seq for comparison
        triplets = np.array([triplet['edge'] for triplet in triplets_rel])
        rel_seqs = np.array([triplet['rel_seq'] for triplet in triplets_rel])
        unique_rel_seqs, rel_cnt =  np.unique(rel_seqs, return_counts=True)

        # Embed unique rel_seqs
        rels_emb = self.model.encode(
            unique_rel_seqs.tolist(),
            convert_to_tensor=True,
            # normalize_embeddings=True
        ).to(self.args.device)
        # print("rels_emb shape:", rels_emb.shape, "query_emb shape:", self.query_emb.shape)
        # scores = self.util.dot_score(self.query_emb, rels_emb).to('cpu')
        scores = model_util.cos_sim(self.query_emb, rels_emb).to('cpu')
        rel_penalty = torch.from_numpy(rel_cnt).float().unsqueeze(0)  # Shape: [1, num_unique_rels]
        rel_scores = scores 
        # Get top-k indices BEFORE thresholding to preserve alignment
        top_k_indices = self.return_top_k(scores)
        
        # Apply threshold AFTER selecting top-k to maintain index consistency
        threshold_mask = scores[0, top_k_indices] > self.threshold
        valid_indices = np.array(top_k_indices)[threshold_mask]
        
        if len(valid_indices) == 0:
            return np.array([]).reshape(0, 3), torch.tensor([])
        
        # Get the valid relation sequences and create mapping
        valid_rel_seqs_list = unique_rel_seqs[valid_indices]
        valid_rel_seqs_set = set(valid_rel_seqs_list)
        
        rel_seq_to_score = {}
        for idx in valid_indices:
            rel_seq = unique_rel_seqs[idx]
            rel_seq_to_score[rel_seq] = rel_scores[0, idx]

        # Filter to keep ALL triplets whose rel_seq is in the valid set
        filtered_indices = [i for i, rel_seq in enumerate(rel_seqs) if rel_seq in valid_rel_seqs_set]
        
        if len(filtered_indices) == 0:
            return np.array([]).reshape(0, 3), torch.tensor([])
        
        filtered_triplets = triplets[filtered_indices]
        filtered_rel_seqs = rel_seqs[filtered_indices]
        
        # Map each triplet to its rel_seq's score
        triplet_scores = torch.tensor([rel_seq_to_score[rel_seq] for rel_seq in filtered_rel_seqs], dtype=torch.float32)

        return filtered_triplets, triplet_scores

   

    def reset(self):
        subgraph_key = None
        self.visited = set()
        self.expanded_from_nodes = set()
        self.k_rels = self.k_rel_org
        self.k_cands = self.k_cands_org
        self._name_cache = {}

    def sampleSubgraph(self, mode="train"):
        assert (self.query is not None), "Please assign a query first using assign_query()"
        assert (self.kg is not None), "KG interface not initialized. Cannot sample subgraph."
        subgraph_key = None
        # print("Sampling subgraph...")
        last_num_cands = 0
        start_entities = sorted(list(set(self.start_entities.copy())))
        start_entities = [(None, mid) for mid in start_entities]  # Initialize with (None, mid)
        self.reset()

        while len(start_entities) > 0:
            # print(f"Expand from {len(start_entities)} start entities")
            triplets = []
            
            # Create a copy to avoid modifying list during iteration
            start_entities_copy = []
            for r, head_mid in start_entities:
                if head_mid in self.expanded_from_nodes:
                    continue  # Skip if already expanded

                if isinstance(head_mid, np.str_):
                    head_mid = head_mid.item()
                start_entities_copy.append((r, head_mid))

            head_mids = [head_mid for r, head_mid in start_entities_copy]

            # Get 1-hop triples from KG interface
            try:
                if isinstance(head_mids[0], (int, np.integer)):
                    # Handle numeric IDs
                    edges = []
                    relations = []
                    for head_mid in head_mids:
                        edges.extend(self.kg.get_1hop_triples(str(head_mid)))
                else:
                    # Handle string IDs
                    edges_list = [self.kg.get_1hop_triples(str(head_mid)).values.tolist() for head_mid in head_mids]
                    edges = [item for sublist in edges_list for item in sublist]
                    relations = [edge[1] for edge in edges]
            except Exception as e:
                LOGGER.warning(f"Error retrieving 1-hop triples: {e}")
                edges = []
                relations = []

            # Filter out visited edges
            filtered_edges = [
                {
                    "edge": [edge[0], edge[1], edge[2]], 
                    "rel_seq": r + " " + self.convert_relation(edge[1]) if r else self.convert_relation(edge[1])
                }
                for edge in edges
                if tuple(edge) not in self.visited
                and tuple(edge) not in self.drop_edges
                and (edge[2] not in self.expanded_from_nodes 
                     and edge[0] not in self.expanded_from_nodes)
            ]
            triplets.extend(filtered_edges)

            # Mark nodes as expanded
            for r, head_mid in start_entities_copy:
                self.expanded_from_nodes.add(head_mid)

            # print(f"Collected {len(triplets)} triplets from KG for current start entities.")
            if len(triplets) == 0:
                LOGGER.warning("No valid triplets found for the current start entities.")
                break
            # print(triplets[:5], "...")
            np_triplets, rel_scores = self.compare_rel_query_and_return_topk(triplets)
            # print(len(np_triplets), "triplets after relation comparison and filtering.")
            # Prune candidates
            if len(np_triplets) > self.k_cands:
                np_triplets = self.prune_cands(np_triplets, rel_scores)

            # Mark triplets as visited
            for triplet in np_triplets:
                self.visited.add(tuple(triplet))

            # Helper function to update dicts

            # Check candidate limit
            if subgraph_key is not None:
                temp_subgraph = np.concatenate([subgraph_key, np_triplets], axis=0)
                temp_num_cands = len(np.unique(temp_subgraph[:, [0, 2]].flatten()))
                if temp_num_cands > self.cands_lim:
                    for triplet in np_triplets:
                        temp_subgraph_single = np.concatenate([subgraph_key, [triplet]], axis=0)
                        temp_num_cands_single = len(np.unique(temp_subgraph_single[:, [0, 2]].flatten()))
                        if temp_num_cands_single <= self.cands_lim:
                            subgraph_key = temp_subgraph_single
                        else:
                            continue
                    break
                else:
                    subgraph_key = temp_subgraph
            else:
                subgraph_key = np_triplets

            # Prepare next start entities
            start_entities = set()
            for triplet in np_triplets:
                if triplet[2] not in self.expanded_from_nodes:
                    start_entities.add((None, triplet[2]))
                elif triplet[0] not in self.expanded_from_nodes:
                    start_entities.add((None, triplet[0]))
            start_entities = sorted(list(set(start_entities)))

            num_cands = len(np.unique(subgraph_key[:, [0, 2]].flatten()))
            # print(f"Collected {num_cands} candidate entities so far.\t- Found {num_cands - last_num_cands} new candidates in this iteration.")
            last_num_cands = num_cands
            if num_cands >= self.cands_lim:
                break

        if subgraph_key is None:
            empty_nodes = torch.tensor([], dtype=torch.long)
            empty_index = torch.zeros(0, dtype=torch.long)
            empty_edges = torch.tensor([], dtype=torch.long).reshape(0, 3)
            return empty_nodes, empty_index, empty_edges

        topk_mids = np.unique(subgraph_key[:, [0, 2]].flatten())
        return topk_mids, subgraph_key


    def evaluate_subgraph(self, type_eval="train"):
        if not subgraph_key:
            self.sampleSubgraph()

        subgraph_key = subgraph_key
        # print("subgraph has", len(subgraph_key), "triplets.")

        entities_id = set()
        rels = set()
        for h, r, t in subgraph_key:
            entities_id.add(h)
            entities_id.add(t)
            rels.add(r)
        # print("subgraph has", len(entities_id), "unique entities and", len(rels), "relation types.")
        entity_score = self.evaluate_ans_coverage(entities_id, type_eval=type_eval)
        # relation_score = self.evaluate_rel_coverage(rels, type_eval=type_eval)
        return entity_score

    def evaluate_ans_coverage(self, entities_id, type_eval="train"):
        answers_id = set(self.query['answers'])
        entity_score = len(answers_id.intersection(entities_id)) / len(answers_id)

        return entity_score

    def evaluate_rel_coverage(self, rels, type_eval="train"):
        # print(self.raw_query)
        notations = extract_notations(self.query['query_type'])
        rels_ans = extract_numbers(self.raw_query)
        rels_ans = set([rel for i, rel in enumerate(rels_ans) if notations[i] == 'r'])

        relation_score = len(rels_ans.intersection(rels)) / len(rels_ans)
        return  relation_score

    def getOneSubgraph(self):
        assert (self.query is not None), "Please assign a query first using assign_query()"
        topk_nodes, node_index, sampled_edges = self.sampleSubgraph()
        return [self.start_entities, topk_nodes, node_index, sampled_edges]

    def getBatchSubgraph(self, subgraph_list: list):
        """
        Process a batch of subgraph queries and return batch-level indexed data.
        
        Args:
            subgraph_list: List of subgraphs from getOneSubgraph()
            
        Returns:
            Tuple of (batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges)
        """
        LOGGER.debug("Getting batch subgraph...")
        batchsize = len(subgraph_list)
        ent_delta_values = [0]
        batch_sampled_edges = []
        batch_idxs, abs_idxs = [], []
        query_sub_idxs = []
        edge_batch_idxs = []

        for batch_idx in range(batchsize):
            sub, topk_nodes, node_index, sampled_edges = subgraph_list[batch_idx]
            num_nodes = len(topk_nodes)
            ent_delta = sum(ent_delta_values)  # Calculate offset

            # Adding ent_delta to make node indices unique in the batch
            sampled_edges[:, 0] = node_index[sampled_edges[:, 0]] + ent_delta
            sampled_edges[:, 2] = node_index[sampled_edges[:, 2]] + ent_delta
            batch_sampled_edges.append(sampled_edges)
            edge_batch_idxs += [batch_idx] * int(sampled_edges.shape[0])

            ent_delta_values.append(num_nodes)
            batch_idxs += [batch_idx] * num_nodes
            abs_idxs += topk_nodes.tolist()
            query_sub_idxs.append(int(node_index[sub]) + ent_delta)
        
        # Convert to tensors
        batch_idxs = torch.LongTensor(batch_idxs)
        abs_idxs = torch.LongTensor(abs_idxs)
        batch_sampled_edges = torch.cat(batch_sampled_edges, dim=0)
        edge_batch_idxs = torch.LongTensor(edge_batch_idxs)
        query_sub_idxs = torch.LongTensor(query_sub_idxs)
        
        LOGGER.debug(f"Batch subgraph processed: {len(batch_idxs)} entities, {len(batch_sampled_edges)} edges")
        
        return batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges
    
    # def updateEdges(self, new_edges):
    #     self.edge_index = np.array(new_edges)
        # ExpandSubgraph.adj = None
        # if ExpandSubgraph.adj is None:
        #     ExpandSubgraph.adj = self.build_adjacency_list()

   