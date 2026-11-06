# Relation inference model

A small model that predicts which relations an entity is **missing** from the incomplete knowledge graph, using only
counts of the relations the entity already has. It does not see the question. It answers "which (relation, direction)
slot at this entity is most likely missing?", not "which relation does the question ask about?".

Code lives in `relation_inference/`:

| File | Contents |
|---|---|
| `create_data.ipynb` | Builds the training data (currently for `family`), trains and saves the model |
| `relation_inference_model.py` | `make_args`, `RelationInferenceModel`, `RelationInferenceData`, `RelationInferenceTrainer` |
| `relation_inference_interface.py` | `RelationInferenceInterface`, which queries a trained model for an entity of a `KGInterface` |
| `model/` | Saved weights (`state_dict` only) |

## Input and output

**Input: relation-count profile.** For an entity `e`, a vector of length `2 * n_rel` built from
`kg.get_1hop_triples(e)` on the **incomplete** graph:

- index `r` (for `0 <= r < n_rel`): number of edges `(e, r, ?)`, where `e` is the head (**outgoing**)
- index `r + n_rel`: number of edges `(?, r, e)`, where `e` is the tail (**incoming**)

`r` is `kg.rel2id[relation]`. For `family`, `n_rel = 12`, so the vector has length 24. Raw counts are used, with no
normalisation.

**Output:** one score per slot in the same `2 * n_rel` layout. A softmax over them gives a distribution over which
slot is missing.

## Training data (`create_data.ipynb`)

**Masked samples (training signal).** For every distinct profile in the incomplete graph, and for every slot `i` with
a count above 0, create one sample:

- input: the profile with `count[i] - 1`
- target: slot `i`

This simulates "one existing edge was removed; which one?". Profiles and samples are built as Python `set`s, so
duplicates are removed: 3,007 `family` entities collapse to 2,017 distinct profiles.

**Leak samples (excluded from training).** For every BRINK question in **train, val and test**, take `q_entity`'s
profile as it is, and as target each edge in `kg.drop_edges` between `q_entity` and `hard_answer` (in either
direction). These are exactly the "real" missing-edge queries the benchmark poses. `populate_inputfeat` removes any
masked sample that is identical to a leak sample (`data = data.difference(remove_data)`). For `family`, that leaves
**14,964** training samples.

Notes on the construction:

- Profiles come from the **incomplete** graph. `drop_edges` are exactly the edges in the complete graph that are
  missing from the incomplete one, so a leak sample's input already lacks its dropped edge and is used without
  masking. An earlier version subtracted 1 again, which drove about 18% of counts to -1. That was removed.
- The `crucial_edges` lookup matches both `(q_entity, r, hard_answer)` and `(hard_answer, r, q_entity)`.
- Every train/val/test row has exactly one `q_entity` and one `hard_answer`, so indexing `[0]` loses nothing.

## Dataset: grouped, multi-hot targets

One profile can have several valid targets, because several different edges could have been masked from it.
`RelationInferenceData` therefore stores **one row per distinct profile** `X`, with a **multi-hot** target `y` marking
every valid slot.

This has two consequences:

- `random_split` on the dataset splits **by profile**, so the same `X` never appears in both train and val/test.
- Evaluation sees the complete set of valid answers for each `X`.

## Training (`RelationInferenceTrainer`)

- **Split:** 80/10/10 train/val/test with `random_split`, seeded with 42. These are splits of the masked samples, not
  the BRINK questions.
- **Loss:** `CrossEntropyLoss` with a soft target, `y / y.sum(dim=1, keepdim=True)`: uniform probability over the valid
  slots of each profile. Each profile has weight 1. This is equivalent to training on one sample per valid target,
  except for that weighting.
- **Optimiser:** AdamW.
- **Validation:** every `val_cycle` epochs (first check at epoch `val_cycle`). Reports val loss, filtered MRR and
  Hits@3.
- **Early stopping on val MRR.** Stops after `patience` validation checks without MRR improving by more than
  `min_delta`. At the end, the best checkpoint is restored and `train()` returns the best MRR.

**Filtered rank** (`_filtered_ranks`): for each valid slot of a profile, its rank counts only the **invalid** slots
that score higher. Other valid slots are masked to `-inf`, so one correct answer is never penalised for ranking below
another. MRR is the mean of `1 / rank` over all valid targets. Hits@3 is the fraction of valid targets with filtered
rank ≤ 3. Ranks from all batches are concatenated before the metrics are computed.

**Default arguments (`make_args`):**

| Arg | Value |
|---|---|
| `in_feature` | `2 * n_rel` (24 for `family`) |
| `n_layer` | 3 |
| `lr` | 5e-4 |
| `batchsize` | 32 |
| `n_epoch` | 200 |
| `val_cycle` | 5 |
| `patience` | 3 (validation checks, not epochs) |
| `min_delta` | 1e-4 |
| `device` | `cuda` if available, else `cpu` |

## Model (`RelationInferenceModel`)

`n_layer` residual blocks, each `Linear(in_feature, in_feature) → ReLU → Dropout(0.1)`, applied as `x = x + block(x)`.
The output keeps the input width, and its values are used directly as logits.

## Interface (`RelationInferenceInterface`)

```python
from relation_inference.relation_inference_interface import RelationInferenceInterface

ri = RelationInferenceInterface(family_kg, model_path="relation_inference/model/family_exclude_BRINK.pt")
ri.query("811", k=3)
# [{'relation': 'wife_of',   'direction': 'incoming', 'score': 0.269},
#  {'relation': 'mother_of', 'direction': 'incoming', 'score': 0.150},
#  {'relation': 'father_of', 'direction': 'outgoing', 'score': 0.130}]
```

- **Constructor:** `RelationInferenceInterface(kg, model_path=None, args=None, kg_type="incomplete")`.
  - `args` defaults to `make_args(kg.dataset_name, 2 * kg.n_rel)`. It must match the architecture used in training
    (`in_feature`, `n_layer`).
  - Weights are loaded onto the CPU first, then moved to `args.device`. This lets GPU-trained weights load on a
    machine without a usable GPU.
  - The model is put in `eval()` mode.
- **`create_input_feat(entity)`:** builds the count profile with `kg.get_1hop_triples`, exactly as the notebook does.
- **`query(entity, k)`:** returns the top-k slots, best first. `score` is the softmax probability.
  - `outgoing` means the missing edge is `(entity, relation, ?)`.
  - `incoming` means the missing edge is `(?, relation, entity)`.

Behaviour to be aware of:

- The default `model_path` is `relation_inference/model/{dataset_name}.pt`. The saved files use other names (see
  below), so pass `model_path` explicitly.
- An entity that isn't in the graph gets an all-zero profile and still returns predictions; no error is raised.
- Relations the entity already has are not filtered out, because an entity can have several edges of the same
  relation.
- `create_input_feat` duplicates the notebook's `create_inputfeat_per_entity`. The two must stay identical, or the
  model will see inputs at inference that differ from training.

## Saved models

| File | Trained on |
|---|---|
| `model/family_exclude_BRINK.pt` | Masked samples, with the leak samples of all BRINK train/val/test questions removed |

## Result

`family_exclude_BRINK.pt`, scored on the 218 `family` **val** questions. For each question, a hit means that one of
the edges actually dropped between `q_entity` and `hard_answer` is among the top-k predicted slots for `q_entity`:

| Hits@1 | Hits@3 | Hits@5 |
|---|---|---|
| 0.225 | 0.514 | 0.693 |

These questions are held out: their leak samples were removed from training, and checkpoint selection used the
separate masked-sample val split. For reference, a uniform random guess over 24 slots gives Hits@3 of about 0.125 (3/24) when a question has one dropped edge.
