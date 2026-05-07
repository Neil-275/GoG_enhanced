import torch

from . import tasks


@torch.no_grad()
def predict_topk(
    model,
    graph,
    entity: int,
    relation: int,
    k: int = 10,
    *,
    predict: str = "tail",
    filter_known: bool = True,
    filter_graph=None,
    promote_known: bool = False,
):
    """Rank candidate entities for a (entity, relation, ?) query.

    This is an inference helper for knowledge graph completion.

    Parameters
    ----------
    model:
        A trained NBFNet-PyG model. Must be callable as `model(graph, batch)`.
    graph:
        A `torch_geometric.data.Data`-like object with `edge_index`, `edge_type`, `num_nodes`.
        This graph provides message-passing context.
    entity:
        The given entity ID. Interpreted as head if `predict="tail"`, or as tail if
        `predict="head"`.
    relation:
        The relation ID.
    k:
        Number of candidates to return.
    predict:
        Either `"tail"` (default) for (head, relation, ?) or `"head"` for (?, relation, tail).
    filter_known:
        If True, mask out candidates that would create a triple already present in `filter_graph`
        (or `graph` if `filter_graph` is None).

    filter_graph:
        Optional. A graph used only for filtering known triples (e.g., the full graph in a
        transductive setting). Defaults to `graph`.

    promote_known:
        If True and `filter_known` is False, add a large positive bias to candidates that are
        already known true triples in `filter_graph`, so they will be ranked above unknown
        candidates.

    Returns
    -------
    (topk_entities, topk_scores)
        `topk_entities`: LongTensor of shape (k,)
        `topk_scores`: FloatTensor of shape (k,)

    Notes
    -----
    This function scores *all* entities as candidates, so runtime is O(num_nodes) per query.
    """

    if k <= 0:
        raise ValueError("k must be a positive integer")

    # Prefer graph's device; fall back to model's device if graph tensors are missing.
    device = getattr(getattr(graph, "edge_index", None), "device", None)
    if device is None:
        device = next(model.parameters()).device

    num_nodes = int(getattr(graph, "num_nodes", getattr(graph, "num_nodes", 0)))
    if not num_nodes:
        raise ValueError("graph.num_nodes must be set")

    entity_t = torch.as_tensor(int(entity), device=device, dtype=torch.long)
    relation_t = torch.as_tensor(int(relation), device=device, dtype=torch.long)

    all_entities = torch.arange(num_nodes, device=device)

    if filter_graph is None:
        filter_graph = graph

    predict = str(predict).lower()
    if predict == "tail":
        # (head, relation, ?) => score all tails
        h_index = entity_t.view(1, 1).expand(1, num_nodes)
        t_index = all_entities.view(1, num_nodes)
        r_index = relation_t.view(1, 1).expand(1, num_nodes)
        known_mask = _known_candidate_mask_tail(filter_graph, entity_t, relation_t, num_nodes, device)
    elif predict == "head":
        # (?, relation, tail) => score all heads
        h_index = all_entities.view(1, num_nodes)
        t_index = entity_t.view(1, 1).expand(1, num_nodes)
        r_index = relation_t.view(1, 1).expand(1, num_nodes)
        known_mask = _known_candidate_mask_head(filter_graph, entity_t, relation_t, num_nodes, device)
    else:
        raise ValueError("predict must be either 'tail' or 'head'")

    batch = torch.stack([h_index, t_index, r_index], dim=-1)  # (1, num_nodes, 3)

    model_was_training = model.training
    model.eval()
    scores = model(graph, batch).view(-1)  # (num_nodes,)
    if model_was_training:
        model.train()

    if filter_known:
        scores = scores.masked_fill(~known_mask, float("-inf"))
    elif promote_known:
        # Promote known-true candidates to the top when doing *unfiltered* ranking.
        # `known_mask` is True for allowed (unknown) candidates, so known-true are ~known_mask.
        known_true = ~known_mask

        finite = torch.isfinite(scores)
        if finite.any():
            s = scores[finite]
            boost = (s.max() - s.min()).abs() + 1
        else:
            boost = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
        scores = scores + known_true.to(scores.dtype) * boost

    k = min(int(k), num_nodes)
    topk_scores, topk_entities = torch.topk(scores, k=k)
    return topk_entities, topk_scores


def _known_candidate_mask_tail(graph, head_index, rel_index, num_nodes: int, device):
    """Mask for candidate tails for query (h, r, ?).

    True => allowed candidate tail.
    False => candidate tail would form a known true triple in graph.
    """

    mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    if not (hasattr(graph, "edge_index") and hasattr(graph, "edge_type")):
        return mask

    # Find all edges matching (head, relation) and exclude their true tails.
    edge_index = torch.stack([graph.edge_index[0], graph.edge_type])  # (2, E)
    query_index = torch.stack([head_index.view(1), rel_index.view(1)])  # (2, 1)
    edge_id, _ = tasks.edge_match(edge_index, query_index)
    if edge_id.numel():
        true_tails = graph.edge_index[1, edge_id]
        mask[true_tails] = False
    return mask


def _known_candidate_mask_head(graph, tail_index, rel_index, num_nodes: int, device):
    """Mask for candidate heads for query (?, r, t).

    True => allowed candidate head.
    False => candidate head would form a known true triple in graph.
    """

    mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    if not (hasattr(graph, "edge_index") and hasattr(graph, "edge_type")):
        return mask

    # Find all edges matching (tail, relation) in (t, r) space and exclude their true heads.
    edge_index = torch.stack([graph.edge_index[1], graph.edge_type])  # (2, E)
    query_index = torch.stack([tail_index.view(1), rel_index.view(1)])  # (2, 1)
    edge_id, _ = tasks.edge_match(edge_index, query_index)
    if edge_id.numel():
        true_heads = graph.edge_index[0, edge_id]
        mask[true_heads] = False
    return mask
