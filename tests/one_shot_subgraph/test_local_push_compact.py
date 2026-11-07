import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from one_shot_subgraph.PPR_sampler import pprSampler


def args(**overrides):
    values = dict(n_samp_ent=3, gpu=0, use_gpu_ppr=False, local_ppr=True,
                  cpu=1, add_manual_edges=False, drop_graph=False,
                  max_nodes_per_query=3, max_edges_per_query=10,
                  local_ppr_alpha=.85, local_ppr_epsilon=1e-6)
    values.update(overrides)
    return SimpleNamespace(**values)


def sampler(tmp_path, **overrides):
    triples = np.asarray([[0, 0, 1], [1, 0, 2], [2, 0, 3], [3, 0, 0]])
    return pprSampler(4, 1, 3, 10, [(0, 1), (1, 2), (2, 3), (3, 0)],
                      triples, str(tmp_path), split='train', args=args(**overrides))


def test_lazy_versioned_cache_and_compact_subgraph(tmp_path):
    sample = sampler(tmp_path)
    assert sample.cache_misses == 0
    nodes, edges, query_local = sample.getOneSubgraph(0)
    assert sample.cache_misses == 1
    assert len(nodes) <= 3 and nodes.tolist() == sorted(set(nodes.tolist()))
    assert nodes[query_local] == 0
    assert edges[:, (0, 2)].max().item() < len(nodes)
    assert 'local-push-v1' in sample.ppr_savePath

    batch = sample.getBatchSubgraph([(nodes, edges, query_local), (nodes, edges, query_local)])
    assert len(batch) == 6
    assert batch[-1].tolist() == [0, len(nodes), 2 * len(nodes)]
    assert batch[1].tolist() == nodes.tolist() * 2


def test_legacy_sparse_cache_and_truncated_recovery(tmp_path):
    sample = sampler(tmp_path)
    path = f'{sample.ppr_savePath}/0.pkl'
    with open(path, 'wb') as handle:
        pickle.dump({2: .2, 0: .8}, handle)
    ids, scores = sample.getPPRscores(0)
    assert ids.tolist() == [0, 2]
    assert scores.dtype == np.float32
    with open(path, 'wb') as handle:
        handle.write(b'broken')
    ids, scores = sample.getPPRscores(0)
    assert len(ids) == len(scores)
    assert sample.cache_misses == 1


def test_head_candidate_budgets_and_frozen_graph(tmp_path):
    sample = sampler(tmp_path, max_nodes_per_query=2, max_edges_per_query=1)
    nodes, edges, _ = sample.getOneSubgraph(0, cand=3)
    assert nodes.tolist() == [0, 3]
    assert len(edges) <= 1
    with pytest.raises(ValueError):
        sample.getOneSubgraph(0, cand=4)
    with pytest.raises(RuntimeError):
        sample.updateEdges([])


def test_empty_support_still_keeps_head(tmp_path, monkeypatch):
    sample = sampler(tmp_path)
    monkeypatch.setattr(sample, 'getPPRscores', lambda _: (np.array([], dtype=np.int64),
                                                          np.array([], dtype=np.float32)))
    nodes, _, query_local = sample.getOneSubgraph(2)
    assert nodes.tolist() == [2]
    assert query_local == 0

