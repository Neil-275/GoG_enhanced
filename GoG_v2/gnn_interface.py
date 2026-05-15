from GoG_v2.kg_interface import KGInterface
# from NBFNet_PyG.nbfnet.inference import *
# from NBFNet_PyG.nbfnet import util as gnn_util
from one_shot_subgraph.model import GNN_auto
from one_shot_subgraph.PPR_sampler import pprSampler
import os
from gnn_config import nbfnet_config, one_shot_subgraph_config
import types
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

Entity = Union[int, str]
Relation = Union[int, str]
Triple = Tuple[Entity, Relation, Entity]


def _is_int(x: Any) -> bool:
    return isinstance(x, (int, np.integer))


def _build_id_map(items: Sequence[Union[int, str]]) -> Dict[Union[int, str], int]:
    """Build a mapping from provided ids/names to contiguous [0..n-1] ids.

    - If items are ints and already look like 0..n-1, mapping is identity.
    - Otherwise, mapping assigns ids by position.
    """
    if len(items) == 0:
        return {}

    if all(_is_int(x) for x in items):
        ints = [int(x) for x in items]
        if sorted(ints) == list(range(len(ints))):
            return {int(x): int(x) for x in ints}
        return {int(x): i for i, x in enumerate(ints)}

    return {x: i for i, x in enumerate(items)}


def _to_id(x: Union[int, str], mapping: Dict[Union[int, str], int], kind: str) -> int:
    if _is_int(x):
        return int(x)
    if x not in mapping:
        raise KeyError(f"Unknown {kind}: {x}")
    return int(mapping[x])


def _normalize_edges(
    edges: Sequence[Triple],
    entity2id: Dict[Union[int, str], int],
    relation2id: Dict[Union[int, str], int],
) -> np.ndarray:
    triples: List[Tuple[int, int, int]] = []
    for h, r, t in edges:
        hid = _to_id(h, entity2id, 'entity')
        rid = _to_id(r, relation2id, 'relation')
        tid = _to_id(t, entity2id, 'entity')
        triples.append((hid, rid, tid))
    return np.asarray(triples, dtype=np.int64)


def _add_inverse_edges(edge_index: np.ndarray, n_rel: int) -> np.ndarray:
    """Append inverse edges (t, r+n_rel, h) for edges with r in [0, n_rel).

    If your input already includes inverse relations, set add_inverse_edges=False.
    """
    if edge_index.size == 0:
        return edge_index

    r = edge_index[:, 1]
    mask = (r >= 0) & (r < n_rel)
    base = edge_index[mask]
    inv = np.stack([base[:, 2], base[:, 1] + n_rel, base[:, 0]], axis=1)
    return np.concatenate([edge_index, inv], axis=0)

class OneShotInterface:
    def __init__(self, dataset_name: str, n_ent: int, n_rel: int):
        assert dataset_name in one_shot_subgraph_config, f"Dataset {dataset_name} not found in config"
        args = one_shot_subgraph_config[dataset_name]["args"]
        self.args = types.SimpleNamespace(**args)
        self.args.n_rel = n_rel
        self.args.n_ent = n_ent
        self.model = GNN_auto(
            self.args,
        ).to(self.args.device)

        weight_path = one_shot_subgraph_config[dataset_name].get("checkpoint")
        if weight_path is not None:
            checkpoint = torch.load(weight_path, map_location=self.args.device)
            state = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            self.model.load_state_dict(state)
        self.model.eval()

        # print(self.args.device)
        if not hasattr(self.args, 'device'):
            raise AttributeError("args must have a 'device' attribute")
        if not hasattr(self.args, 'cache_dir'):
            raise AttributeError("args must have a 'cache_dir' attribute")
        self.cache_dir = self.args.cache_dir

        if not hasattr(self.args, 'add_manual_edges'):
            self.args.add_manual_edges = False
        if not hasattr(self.args, 'use_gpu_ppr'):
            self.args.use_gpu_ppr = True
        

        self.topk_ratio = float(getattr(self.args, 'topk', 0.1))
        self.topm_ratio = float(getattr(self.args, 'topm', -1))



    def assign_graph(self, kg: KGInterface):
        self.kg: KGInterface = kg
        entities, relations, edges = kg.entities, kg.relations, kg.list_edges
        self.entity2id = kg.entity2id
        self.relation2id = kg.rel2id
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        
        self.n_ent = len(self.entity2id)
        self.n_rel = len(self.relation2id)

        edge_index = _normalize_edges(edges, self.entity2id, self.relation2id)
        self.edge_index = np.array(edge_index)
        if self.args.add_inverse_edges:
            edge_index = _add_inverse_edges(edge_index, self.n_rel)
        
        if self.args.add_idd_edges:
            idd = np.stack([np.arange(self.n_ent), 2 * self.n_rel * np.ones(self.n_ent), np.arange(self.n_ent)], axis=1)
            edge_index = np.concatenate([edge_index, idd.astype(np.int64)], axis=0)

        n_samp_ent = int(max(1, round(self.topk_ratio * self.n_ent)))
        n_samp_edge = int(self.topm_ratio * edge_index.shape[0]) if self.topm_ratio > 0 else -1
        self.args.n_samp_ent = n_samp_ent
        self.args.n_samp_edge = n_samp_edge
        homo_edges = list(set([(int(h), int(t)) for (h, _, t) in edge_index]))
        print("okay")
        self.sampler = pprSampler(
            self.n_ent,
            self.n_rel,
            n_samp_ent,
            n_samp_edge,
            homo_edges,
            edge_index,
            self.cache_dir,
            split='infer',
            args=self.args,
        )

    def predict_topk(
        self,
        head: Entity,
        relation: Relation,
        k: int = 10,
        known=True,
    ):
        """Predict top-k tails for a single (head, relation) query.

        head/relation can be int ids or names (matching the provided entity/relation lists).

        Returns:
            (scores, tail_ids): both are 1D CPU tensors of shape [topk].
        """
        assert hasattr(self, "kg")
        
        # print("entity:", head)
        # print("relation:", relation)
        hid = _to_id(head, self.entity2id, 'entity')
        rid = _to_id(relation, self.relation2id, 'relation')
        # print("rid:", rid)
        q_sub = torch.tensor([hid], dtype=torch.long)
        q_rel = torch.tensor([rid], dtype=torch.long)

        subgraph = self.sampler.getOneSubgraph(int(hid))
        subgraph_data = self.sampler.getBatchSubgraph([subgraph])

        values = self.model.inference(q_sub, q_rel, subgraph_data)
        # print("shape of values:", values.shape)
        maxx = values.max().item()
        # known_tails = set(neighbors[:, 1])
        #extract edges that have head as head or tail and relation as rid use edge_index
        neighbors = self.edge_index[(self.edge_index[:, 0] == hid) & (self.edge_index[:, 1] == rid)]
        # print("neighbors:", neighbors)
        neighbors = np.concatenate([neighbors, self.edge_index[(self.edge_index[:, 2] == hid) & (self.edge_index[:, 1] == rid)]], axis=0)
        ent_neighbors = set(neighbors[:, 2]).union(set(neighbors[:, 0])).difference({hid})
        # print("known neighbors:", ent_neighbors)
        if known and ent_neighbors:
            values[0, list(ent_neighbors)] = maxx + 1.0  # Promote known tails to the top

        # sort 
        values, indices = torch.topk(values[0], k=k)
        # print(len(indices), indices)
        ## Map back to original entity ids
        tail_ids = [self.id2entity[int(idx)] for idx in indices.detach().cpu().numpy()]

        return  tail_ids
