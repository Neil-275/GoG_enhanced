import os
import sys

import torch

sys.path.insert(0, os.path.abspath('one_shot_subgraph'))
from base_model import BaseModel


def bare_model(n_ent=5):
    model = BaseModel.__new__(BaseModel)
    model.n_ent = n_ent
    return model


def test_compact_loss_matches_dense_reference_and_gradient():
    model = bare_model()
    scores = torch.tensor([1.2, -0.4, 0.0], requires_grad=True)
    output = {'node_scores': scores, 'abs_idxs': torch.tensor([0, 2, 4]),
              'node_ptr': torch.tensor([0, 3])}
    loss, covered = model.compact_global_loss(output, torch.tensor([2]))
    dense = torch.zeros(5)
    dense[torch.tensor([0, 2, 4])] = scores
    reference = torch.logsumexp(dense, 0) - dense[2]
    assert torch.allclose(loss, reference)
    assert covered == [True]
    assert torch.autograd.grad(loss, scores)[0].isfinite().all()


def test_compact_loss_missing_target_and_filtered_rank():
    model = bare_model(6)
    scores = torch.tensor([-1.0, 0.0, 2.0])
    output = {'node_scores': scores, 'abs_idxs': torch.tensor([0, 2, 5]),
              'node_ptr': torch.tensor([0, 3])}
    loss, covered = model.compact_global_loss(output, torch.tensor([3]))
    dense = torch.zeros(6); dense[[0, 2, 5]] = scores
    assert torch.allclose(loss, torch.logsumexp(dense, 0))
    assert covered == [False]
    rank, present = model.compact_filtered_rank(scores, output['abs_idxs'], 0, [0, 5, 5, 99])
    # Entity 5 is filtered; sampled entity 2 and three unfiltered outside zeros beat -1.
    assert (rank, present) == (5, True)
