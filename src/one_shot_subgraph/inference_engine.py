import os
import types
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from one_shot_subgraph.model import GNN_auto
from one_shot_subgraph.PPR_sampler import pprSampler


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


class InferenceEngine:
    """Inference-only interface that does not read KG files.

    You provide:
      - entities: list of entity ids or names
      - relations: list of relation ids or names
      - edges: list of (h, r, t) triples using either ids or names

    It builds:
      - a pprSampler on the provided graph (with its own cache dir)
      - a GNN_auto model sized for the provided entity/relation sets
      - loads weights from a checkpoint

    Then you can call predict_topk(head, relation, topk).
    """

    def __init__(
        self,
        *,
        args,
        entities: Sequence[Entity],
        relations: Sequence[Relation],
        edges: Sequence[Triple],
        weight_path: str,
        cache_dir: str = './infer_cache/',
        add_inverse_edges: bool = True,
        add_idd_edges: bool = False,
    ):
        """Create an inference engine using in-memory KG data.

        Args:
            args: an argparse-like object (same as training) providing fields like:
                - gpu, topk, topm, add_manual_edges, use_gpu_ppr
                - hidden_dim, attn_dim, n_layer/layer, dropout, act, initializer,
                  concatHidden, shortcut, readout
            entities/relations/edges: in-memory KG definition.
            weight_path: checkpoint path.
            cache_dir: where to store PPR caches for this inference graph.
            add_inverse_edges: if True, add inverse edges (t, r+n_rel, h) for base relations.
            add_idd_edges: if True, append identity/self-loop edges.
        """

        self.entity2id = _build_id_map(list(entities))
        self.relation2id = _build_id_map(list(relations))
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.id2relation = {v: k for k, v in self.relation2id.items()}

        self.n_ent = len(self.entity2id)
        self.n_rel = len(self.relation2id)

        edge_index = _normalize_edges(edges, self.entity2id, self.relation2id)
        if add_inverse_edges:
            edge_index = _add_inverse_edges(edge_index, self.n_rel)

        if add_idd_edges:
            idd = np.stack([np.arange(self.n_ent), 2 * self.n_rel * np.ones(self.n_ent), np.arange(self.n_ent)], axis=1)
            edge_index = np.concatenate([edge_index, idd.astype(np.int64)], axis=0)

        # pprSampler expects certain fields on args
        if not hasattr(args, 'gpu'):
            raise AttributeError("args must have a 'gpu' attribute")
        if not hasattr(args, 'add_manual_edges'):
            args.add_manual_edges = False
        if not hasattr(args, 'use_gpu_ppr'):
            args.use_gpu_ppr = True

        # number of sampled nodes/edges (match training conventions)
        if hasattr(args, 'n_samp_ent') and args.n_samp_ent is not None:
            n_samp_ent = int(args.n_samp_ent)
        else:
            topk_ratio = float(getattr(args, 'topk', 0.1))
            n_samp_ent = int(max(1, round(topk_ratio * self.n_ent)))
            args.n_samp_ent = n_samp_ent

        topm_ratio = float(getattr(args, 'topm', -1))
        n_samp_edge = int(topm_ratio * edge_index.shape[0]) if topm_ratio > 0 else -1

        homo_edges = list(set([(int(h), int(t)) for (h, _, t) in edge_index]))

        os.makedirs(cache_dir, exist_ok=True)
        self.sampler = pprSampler(
            self.n_ent,
            self.n_rel,
            n_samp_ent,
            n_samp_edge,
            homo_edges,
            edge_index,
            cache_dir,
            split='infer',
            args=args,
        )

        # Build model params from args (must match checkpoint)
        params = types.SimpleNamespace()
        params.n_ent = self.n_ent
        params.n_rel = self.n_rel
        params.n_layer = int(getattr(args, 'n_layer', getattr(args, 'layer', 6)))
        params.hidden_dim = int(getattr(args, 'hidden_dim', 64))
        params.attn_dim = int(getattr(args, 'attn_dim', 32))
        params.dropout = float(getattr(args, 'dropout', 0.0))
        params.act = getattr(args, 'act', 'relu')
        params.initializer = getattr(args, 'initializer', 'binary')
        params.concatHidden = bool(getattr(args, 'concatHidden', False))
        params.shortcut = bool(getattr(args, 'shortcut', False))
        params.readout = getattr(args, 'readout', 'linear')

        loader_stub = types.SimpleNamespace(n_ent=self.n_ent)
        self.model = GNN_auto(params, loader_stub)

        device = torch.device(f'cuda:{int(args.gpu)}' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        if weight_path is not None:
            checkpoint = torch.load(weight_path, map_location=device)
            state = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            self.model.load_state_dict(state)
        self.model.eval()

    def predict_topk(
        self,
        *,
        head: Entity,
        relation: Relation,
        topk: int = 10,
    ):
        """Predict top-k tails for a single (head, relation) query.

        head/relation can be int ids or names (matching the provided entity/relation lists).

        Returns:
            (scores, tail_ids): both are 1D CPU tensors of shape [topk].
        """
        hid = _to_id(head, self.entity2id, 'entity')
        rid = _to_id(relation, self.relation2id, 'relation')

        q_sub = torch.tensor([hid], dtype=torch.long)
        q_rel = torch.tensor([rid], dtype=torch.long)

        subgraph = self.sampler.getOneSubgraph(int(hid))
        subgraph_data = self.sampler.getBatchSubgraph([subgraph])

        values, indices = self.model.inference(q_sub, q_rel, subgraph_data, topk=int(topk))
        return values[0].detach().cpu(), indices[0].detach().cpu()
