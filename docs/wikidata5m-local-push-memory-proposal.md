# Memory-efficient Wikidata5m training with local-push PPR

## Purpose and recommendation

Preserve global Wikidata entity IDs and the existing scoring semantics while making per-query data structures scale with local PPR support and sampled subgraph size. Training on a subset of queries should not require allocating a full entity-sized vector for every query.

This document is a proposal based on the current `one_shot_subgraph` implementation. It does not implement the changes or claim measured performance improvements.

The recommended sequence is:

1. Keep local-push PPR results sparse through candidate selection.
2. Bound sampled nodes and edges with absolute budgets.
3. Replace full-vocabulary node mappings and evaluation labels with compact IDs.
4. Return sampled-node scores and account for unsampled zero scores analytically.
5. Limit PPR preprocessing to required query heads and make cache validity explicit.

## Findings in the current implementation

Let `N` be the total entity count, `B` the query batch size, `S` the number of nonzero entries in a query's local-push result, and `K` the number of distinct sampled entities for that query.

| Code location | Current behavior | Consequence |
| --- | --- | --- |
| `one_shot_subgraph/PPR_sampler.py:_local_push_compute` | Returns PPR scores as a dictionary | Provides a compact starting representation |
| `PPR_sampler.py:getPPRscores` | Converts that dictionary into a float32 vector of length `N` | Reintroduces a global allocation for every sampled query |
| `PPR_sampler.py:sampleSubgraph` | Calls `np.argsort` on the full score vector | Sorts all entities, including zero-score entries |
| `one_shot_subgraph/train_auto.py` | Sets the node budget to `int(args.topk * loader.n_ent)` | The default fraction `0.1` requests about 500,000 nodes when `N = 5,000,000` |
| `PPR_sampler.py:sampleSubgraph` | Creates `torch.zeros(self.n_ent).long()` for `node_index` | Retains a dense global-to-local map for every query until collation |
| `one_shot_subgraph/load_data.py:__getitem__` | Creates a length-`N` multi-hot answer vector during validation/test | Labels scale with the full vocabulary; training already uses target IDs |
| `one_shot_subgraph/model.py:GNN_auto.forward` | Expands sampled scores into `[B, N]` | Allocates global scores although propagation uses sampled nodes |
| `one_shot_subgraph/base_model.py:evaluate` | Creates full-vocabulary filter vectors | Adds another source of dense evaluation memory |

For an illustrative `N = 5,000,000`:

- One float32 PPR vector costs 20 MB, excluding sorting workspace.
- One int64 node map or answer vector costs 40 MB.
- At the hardcoded evaluation batch size of 108, node maps alone occupy approximately 4.32 GB; stacked answer labels occupy another 4.32 GB.
- One float32 `[108, N]` score matrix occupies approximately 2.16 GB.

These are decimal byte estimates, not a measured peak. Temporary conversions, stacking, device copies, graph storage, activations, and gradients add further costs. These allocations do not necessarily all coexist on the same device.

## Proposed design

### 1. Preserve sparse PPR support

For the local-push path, return either `{global_entity_id: score}` or aligned arrays of entity IDs and scores from `getPPRscores`. Do not expand to length `N`.

Select candidates from positive-score entries only. Use a bounded heap or partial selection when support is large, and define deterministic tie-breaking by entity ID. Retain support-based score lookup for edge weighting: `score(head) + score(tail)`, with missing values treated as zero.

Do not fill unused candidate slots with arbitrary zero-score entities when `S` is smaller than the node budget. Always retain the query head, including when the PPR result is empty. Reserve its slot so the total remains within the configured budget.

This changes the sampled graph relative to the current zero-padding behavior. Record the change in experiment metadata and compare answer coverage. Preserving the loss formula does not make different sampled graphs equivalent.

Keep dense cache compatibility explicit: either convert legacy arrays to compact support on load or regenerate them. Newly generated local-push caches should remain compact. Other PPR methods may retain their existing representation behind a separate adapter.

### 2. Use absolute node and edge budgets

Introduce `max_nodes_per_query` and `max_edges_per_query` for the local-push path. Do not derive these limits from total graph size. Preserve legacy fractional options for existing experiments, with explicit precedence when absolute budgets are supplied.

An initial node-budget sweep could use 256, 512, 1,024, and 2,048 nodes. These are experimental starting points, not validated defaults. Select budgets using peak memory, sampling latency, answer coverage, and validation metrics.

Expose local-push tolerance and restart probability as configuration, and record them in caches and experiment metadata. The current local-push implementation uses `alpha` as restart mass; do not assume this matches the damping convention of every alternative implementation.

Handle empty support and isolated heads explicitly. A node cap bounds the final sample but does not itself bound local-push work or the temporary outgoing-edge set used to build the induced subgraph. High-degree heads can still require substantial work. Any additional push-work or edge-expansion cap must be documented as an approximation and evaluated for coverage loss.

### 3. Use compact global-to-local indexing

Replace the length-`N` `node_index` with sorted sampled global IDs and `searchsorted`, or a dictionary containing only sampled IDs.

The compact subgraph contract should contain:

- Sampled global entity IDs, used for output interpretation and target matching.
- Edges whose endpoints use local indices in `[0, K)`.
- The query head's local index.

Collation adds a cumulative node offset to each query's local edge endpoints and head index. Keep `batch_idxs` and `abs_idxs` for compatibility with the existing GNN propagation. An optional `node_ptr` of length `B + 1` makes per-query score segments explicit.

Validate that both endpoints belong to the sampled set before accepting mapped indices. Avoid mutating reusable cached subgraphs during collation.

Per-query mapping storage becomes proportional to `K`, rather than `N`.

### 4. Store evaluation answers and filters as IDs

In `load_data.py:__getitem__`, replace evaluation multi-hot construction with:

```python
obj = torch.as_tensor(answer[idx], dtype=torch.long)
```

This is not a standalone replacement: update the consumers together.

- Keep training targets as the current integer IDs.
- For validation/test, collate answers as a list of variable-length tensors, or concatenate IDs with query offsets.
- Update `prepareData` to handle this representation instead of assuming one tensor with `.cuda()`.
- Iterate answer IDs directly during evaluation instead of using `torch.nonzero` on dense labels.
- Use existing filter ID collections for membership checks instead of allocating dense filter masks.

Changing the multi-hot dtype to boolean would reduce memory but would retain the underlying `B × N` scaling.

### 5. Keep sampled scores and preserve global scoring semantics

`GNN_auto.forward` already produces one score per sampled node before creating `scores_all`. Expose those scores with the batch/global-ID mapping. Use an explicit output contract so training, evaluation, and inference agree on the representation.

For a query with sampled set `C`, the current model assigns score zero to every entity outside `C`. Therefore its global softmax partition function is exactly:

```text
Z = sum(exp(s_v) for v in C) + (N - K)
loss = log(Z) - s_target
```

Use the sampled target score when present and zero otherwise. Compute the expression stably:

```text
sampled_log_Z = logsumexp(sampled_scores)
outside_log_Z = log(N - K) if N > K else -infinity
log_Z = logaddexp(sampled_log_Z, outside_log_Z)
```

Require unique global entity IDs within each query's sample so `K` is the correct count. Use a sum over per-query losses to preserve the intended batch reduction. The current global loss expression mixes a `[B, 1]` maximum with `[B]` terms, producing unintended broadcasting; correct that shape issue explicitly rather than preserving its accidental scaling.

This preserves the intended global softmax objective for a fixed sampled graph without allocating `[B, N]`. A missing target still has a fixed zero score, so this change alone does not teach the model to retrieve an absent answer.

### 6. Compute filtered global ranks without dense matrices

Preserve the current evaluator's strict-greater comparison, which assigns optimistic ranks to ties. For each answer `t`, let `F` be the unique filtered answer IDs for the query and let `s_t` be its sampled score or zero when absent.

```text
rank(t) = 1
        + count(v in C, v not in F, s_v > s_t)
        + outside_unfiltered_count * indicator(0 > s_t)

outside_unfiltered_count = N - K - |F outside C|
```

Apply filters within the evaluation entity universe. Count all answers in each multi-answer query as the existing evaluator does. If a different tie policy is desired, treat it as a separate metric change.

Do not equate score zero with a missing candidate. Sampled nodes can legitimately score zero; compute answer coverage from ID membership.

Candidate-only ranks are a different metric. They should not be presented as full-vocabulary filtered MRR. In particular, optimistic zero-score ties can make missing answers look deceptively strong; report answer coverage alongside ranking metrics.

For inference, use candidate-only top-k over sampled nodes. Return their original global entity IDs through `abs_idxs`; local indices must never escape the model/sampler boundary. This deliberately avoids arbitrary unsampled zero-score entities when sampled scores are negative. It changes inference ranking semantics from the dense global tensor, so document it in the API and report candidate coverage separately. Exact implicit-global top-k and dense output are outside the first implementation.

### 7. Restrict preprocessing and manage cache validity

Compute local-push PPR only for heads used by the selected training queries and validation/test queries. Include heads introduced by inverse triples. Support lazy computation and a bounded in-memory cache rather than requiring a file for every entity in the vocabulary.

The existing `topic_ent_file` restricts PPR preprocessing but does not itself subset the loader's training examples. Align query selection and preprocessing, or provide a cache-miss computation path.

Cache metadata should identify the graph version or fingerprint, entity mapping, split, PPR parameters, and algorithm version. Candidate caches also depend on candidate-selection rules and budgets.

Currently `updateEdges` replaces the sampling edge index and triples, but does not rebuild local-push adjacency or invalidate PPR scores. Choose one explicit policy:

- Freeze the fact graph and reuse matching caches for the experiment.
- When reshuffling changes the fact graph, rebuild adjacency and invalidate/recompute affected caches; full invalidation is the straightforward correct baseline.

Using a separate fixed graph for PPR is another experimental design, but its edge visibility must be stated and checked against the intended train/evaluation protocol.

## Training on a subset versus reducing the graph

These are separate choices:

| Choice | Benefit | Tradeoff |
| --- | --- | --- |
| Subset training queries; retain the full allowed fact graph | Less training and PPR preprocessing while preserving graph context | Still requires full graph storage |
| Build a smaller graph and remap its retained entities contiguously | Reduces graph storage and all global structures | Changes PPR neighborhoods and the evaluation entity universe |

Prefer the first approach when full graph storage fits and per-query allocations are the bottleneck. If graph storage itself does not fit, construct a documented subset with mappings back to Wikidata IDs. Rebuild caches after remapping, preserve split rules, and report metrics as subset experiments.

The proposed changes do not remove all global costs. The current program constructs three dataset loaders, retains several triple representations, creates global self-loop triples, and moves sampler triples to the selected device. Profile those separately. Possible follow-up work includes shared immutable dataset storage, CPU-resident compact edge arrays, transferring only sampled edges to the GPU, and generating self-loops after sampling.

## Alternative: candidate-only learning

After establishing the compact global-scoring baseline, candidate-only softmax can be evaluated as a separate objective. Missing training targets require an explicit policy, such as training-only target inclusion or an `OTHER` class. Never insert known validation/test answers into candidates.

The current `scoring_mode='local'` branch is not ready to enable: it expects candidate scores, an `OTHER` logit, and a six-element subgraph contract, while the model and preparation path supply different outputs. It also leaves later training code dependent on `pos_scores` from the global branch. Complete and validate that path before using it.

## Implementation plan

### Locked decisions and compatibility boundary

The first implementation applies only when `--local_ppr` is enabled. Matrix and NetworkX PPR retain their current fractional `topk`/`topm`, dense score, and inference behavior so existing small-dataset experiments remain usable. The local-push path uses these defaults:

```text
max_nodes_per_query = 1024
max_edges_per_query = 10000
local_ppr_alpha = 0.85
local_ppr_epsilon = 1e-6
```

Add corresponding CLI arguments to training, HPO, and standalone inference entry points. Positive absolute limits take precedence in local-push mode; `topk` and `topm` remain accepted but are ignored there with one startup message. Reject non-positive node limits and edge limits other than `-1` or a positive integer. Use `-1` to disable the edge cap explicitly.

Local-push training uses a frozen fact graph. If `--local_ppr` is combined with training-time fact reshuffling, fail at startup with a clear message requiring `not_shuffle_train=True`. Do not silently reuse PPR computed for a different graph. Checkpoints remain compatible because model parameter shapes and names do not change.

### Phase 1: compact local-push sampling and cache lifecycle

Implement a local-push-specific compact path in `PPR_sampler.py`:

1. Represent PPR output as aligned `int64` global entity IDs and `float32` scores. `getPPRscores` converts legacy dictionary/tuple/dense cache payloads to that compact in-memory form without allocating a length-`N` array. New local-push cache files store versioned compact arrays.
2. Select up to `max_nodes_per_query` positive-score nodes using the order `(score descending, global ID ascending)`. Reserve a slot for the query head, insert it if absent, deduplicate IDs, then sort selected global IDs ascending for deterministic local indexing. If `cand` is supplied, pin the requested candidate after the head and fill remaining slots by PPR rank; reject a candidate outside `[0, N)`.
3. Build the induced edge set from those IDs. When it exceeds `max_edges_per_query`, retain edges by `(PPR(head) + PPR(tail)) descending`, then `(head, relation, tail)` ascending. Add manual edges only after ordinary-edge truncation and include them in the final 10,000-edge limit by reserving their required capacity; reject configurations where mandatory manual edges alone exceed the limit.
4. Replace the dense global-to-local vector with `searchsorted` over the sorted selected IDs. Validate every retained endpoint before remapping. Store edges with local endpoints immediately so batching only adds an offset and never mutates global/cached edge tensors.
5. Change one-subgraph output to `(global_node_ids, local_edges, query_local_idx)`. Change batched output to `(batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_edges, node_ptr)`, where `node_ptr` has `B + 1` entries and `abs_idxs` always contains original global IDs.

For local-push mode, compute cache misses lazily in `getPPRscores`. If `topic_ent_file` is supplied, retain eager parallel preprocessing for that finite list; otherwise do not enumerate all entities at sampler construction. Use a versioned directory derived from split, PPR parameters, and a SHA-256 fingerprint of canonical CSR adjacency arrays plus `N`; old cache directories remain untouched. Write one metadata JSON file with schema version, fingerprint, entity count, alpha, epsilon, and algorithm name. Atomic per-head writes use a temporary file followed by rename so interrupted jobs cannot leave valid-looking partial entries.

The train and test samplers have independent graph fingerprints and cache namespaces. `updateEdges` raises in local-push mode because the selected frozen-graph policy makes an in-place graph replacement invalid. Add cache hit/miss and PPR support-size counters for benchmark output; a process-local LRU is optional only after profiling and is not required for correctness.

### Phase 2: compact labels and batch/device contract

Update `load_data.py` so training samples continue to return one global target ID, while validation/test samples return a one-dimensional tensor of global answer IDs. In `collate_fn`, stack training targets but retain evaluation answers as a list of variable-length tensors. Append `node_ptr` to the sampler fields returned by the collator.

Update `BaseModel.prepareData` to branch on target representation: move the stacked training target tensor to the model device, and preserve evaluation answers as per-query ID tensors. Move only tensors needed by GNN propagation to the model device. Keep `abs_idxs` and `node_ptr` available to all loss, evaluation, and inference consumers. Use device-neutral `.to(device)` calls rather than hardcoded `.cuda()` so CPU tests and inference remain valid.

The invariant at this boundary is: entity IDs are global in targets and `abs_idxs`; edge endpoints and `query_sub_idxs` are batch-local; `node_ptr[i]:node_ptr[i+1]` selects exactly query `i`'s nodes and scores.

### Phase 3: compact model output and global training objective

In local-push mode, `GNN_auto.forward` stops creating `scores_all`. It returns a mapping with:

```python
{
    "node_scores": Tensor[sum_K],
    "abs_idxs": LongTensor[sum_K],
    "node_ptr": LongTensor[B + 1],
}
```

Keep the existing dense tensor return for non-local PPR modes. Add small shared helpers in `base_model.py` to find a global ID within one query segment, calculate compact global-softmax loss, and calculate filtered rank. Avoid duplicating this logic across train/validation/test loops.

For each training query, compute the exact implicit-global loss for the sampled graph:

```text
sampled_log_Z = logsumexp(sampled_scores)
outside_log_Z = log(N - K), or -infinity when K == N
log_Z = logaddexp(sampled_log_Z, outside_log_Z)
target_score = sampled score if target is present, otherwise 0
query_loss = log_Z - target_score
batch_loss = sum(query_loss)
```

Assert unique `abs_idxs` per segment and `0 < K <= N`. Determine answer coverage by target-ID membership, never by `target_score == 0`. Remove the unfinished `scoring_mode='local'`/`OTHER` branch from this execution path; candidate-only inference does not change the global training objective.

### Phase 4: compact filtered evaluation

Unify train/validation/test rank calculation around one helper. Given query segment `C`, answer `t`, and unique filter set `F`, use strict-greater ranking to match the current evaluator:

```text
target_score = sampled score for t, or 0 when t is outside C
sampled_higher = count(v in C and v not in F where score(v) > target_score)
outside_unfiltered = N - |C| - |F outside C|
rank = 1 + sampled_higher + outside_unfiltered * indicator(0 > target_score)
```

Validate and deduplicate filter IDs before counting them, and ignore IDs outside `[0, N)`. Evaluate every answer ID in a multi-answer query. Preserve existing MRR/Hits calculations and optional mean-rank collection. Add candidate answer coverage to the evaluation output for train, validation, and test. Remove dense label, filter, and score allocations from the local-push branch.

### Phase 5: candidate-only inference with stable global IDs

Make local-push `GNN_auto.inference` return the compact mapping. Update `BaseModel.predict_topk`, `one_shot_subgraph/inference_engine.py`, and `GoG/gnn_interface.py` to select top-k only within each sampled segment and map positions through `abs_idxs`.

Preserve each wrapper's external return type: score/ID wrappers return scores plus global integer entity IDs, while the GoG wrapper resolves those global IDs through `id2entity`. Cap requested `k` at candidate count and return an empty result only for an empty segment, which should be impossible because the query head is mandatory. Resolve equal scores by global entity ID ascending. Known-neighbor promotion/filtering in the GoG wrapper continues to operate on global IDs after candidate extraction.

Update CLI help and inference docstrings to state that local-push top-k is candidate-only. Add `max_nodes_per_query`, `max_edges_per_query`, alpha, and epsilon to inference configuration so it uses the same sampling contract as training. Local indices remain internal and are never serialized or returned.

### Phase 6: tests and benchmarking

Add focused pytest coverage under `tests/one_shot_subgraph/` using small synthetic graphs:

- Sparse cache conversion, lazy cache miss, version/fingerprint isolation, truncated-cache recovery, and refusal to update a frozen local-push graph.
- Deterministic candidate and edge selection; query-head and pinned-candidate inclusion; empty support; isolated heads; invalid candidates; manual-edge capacity; unique nodes; and correct local endpoint remapping.
- Collation of variable-length multi-answer labels and exact `node_ptr`, `abs_idxs`, batch offsets, and device behavior on CPU and CUDA when available.
- Compact loss values and gradients versus a correctly shaped dense reference for present/missing targets, negative/zero scores, `K=1`, and `K=N`.
- Compact filtered ranks versus a dense strict-greater reference, including filtered IDs inside/outside the sample, zero-score ties, missing answers, duplicate filters, and multi-answer queries.
- Candidate-only inference returns original global IDs, deterministic ties, fewer than requested candidates, and correct GoG name resolution/filtering.
- Non-local PPR smoke tests confirm the legacy five-field subgraph contract and dense model output still work.

Add a Wikidata5m benchmark command or script that records peak host/GPU memory, PPR cache hit rate, support-size percentiles, sampled node/edge percentiles, sampler and batch latency, answer coverage, MRR, and Hits@1/10. Run the fixed 1,024/10,000 baseline on a fixed query subset and compare it with at least 256/2,500, 512/5,000, and 2,048/20,000. Store the command, graph fingerprint, query IDs or query-file fingerprint, seed, and all PPR parameters with the results.

Acceptance requires:

- No per-query length-`N` PPR vectors, node maps, answer vectors, or filter vectors in local-push mode.
- No `[B, N]` score tensor in local-push training, evaluation, or inference.
- Compact loss and filtered ranks match dense references within numerical tolerance for identical sampled graphs.
- Inference returns unchanged global entity IDs and never exposes local indices.
- Peak memory stays within the selected 1,024-node/10,000-edge bound apart from shared graph storage and model activations.
- Existing checkpoints load without parameter migration, and non-local PPR behavior passes regression smoke tests.

## References

- [Andersen, Chung, and Lang: Local Graph Partitioning using PageRank Vectors](https://www.cs.cmu.edu/~15859n/RelatedWork/local_partitioning_full.pdf) — background on local approximate PageRank; this does not establish equivalence between the repository's implementation and every variant in the paper.
- [PyTorch CrossEntropyLoss documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — integer class targets and cross-entropy semantics.
