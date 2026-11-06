# CLAUDE.md

Research code for LLM-agent question answering over an **incomplete** knowledge graph. There is no build system,
package manifest, or test suite: work happens through Python entry points and notebooks. Facts below describe branch
`feat/propose-collect-finish`; dated items are snapshots, so confirm them against the code before relying on them.

## Core work of this branch

**Problem.** Each benchmark (BRINK) question has a **hard answer** reachable only through an edge dropped from the
graph the agent walks (`G_inc`). Graph traversal alone cannot reach it, so the agent needs a link predictor to recover
it. The headline question is whether the agent finds that answer, measured by `Hits@Hard`.

**This branch's change.** It replaces the GoG loop `Search`/`Generate`/`Collect`/`Finish` with
**`Propose`/`Collect`/`Finish`**, making the GNN link predictor first-class evidence on every entity visit. Before,
Predict fired only on a detected "structural break" (about 2-13% of the time). Now `Search` and the trigger are gone
and Predict runs on every selected relation.

```
question -> LLM Thought/Action loop (up to --max_steps)
  Propose[e_current | q_sub]  one entity per call
     pick (relation, direction) slots -> observed edges in G_inc + GNN Predict -> labelled Observation
  Collect[a | b]              add answers to R_t, then fan out along their relation path
  Finish                      stop; R_t is the prediction
```

- **Propose** absorbs both jobs of `Search`. The agent chooses the next `e_current` inside its Thought, from any entity
  id seen earlier in the trajectory; no navigation action or ranking module exists. Observations label observed and
  predicted candidates separately.
- **Collect fan-out** recovers the siblings Propose's top-K cap hid: it traces `e`'s relation path to the topic entity
  through first-production-wins `provenance`, replays that path on `G_inc`, and adds the whole frontier to `R_t`. The
  Observation reports only a count, because `e_current` selection reads the Observation text.
- **Two Propose modes**, chosen by `--propose_mode`:
  - `prob` (default, `GoG/prob_propose.py`, in `KGEnv.prob_propose_action`): LLM-free. Calibrated `P(r,d)` from the
    Propose-Predict baseline score times calibrated `P(e|r,d)` from the GNN, marginalised over slots. Only predicted
    entities are ranked and thresholded; observed ones are added unranked. The LLM still drives the Thought/Action loop.
  - `llm` (`KGEnv.propose`, `propose_action`): relation-selection and direction-specification prompts.
- **Relation prior** (`relation_inference/`): a small model predicting which `(relation, direction)` slot an entity is
  missing, from its relation-count profile alone. `prob` mode needs it for `P(r,d)`.

### Sources of truth

| Before you touch | Read |
|---|---|
| The action space, Propose, Predict, or Observation format | [docs/New_agent_framework_instruction.md](docs/New_agent_framework_instruction.md) |
| `collect_action`, `_collect_one`, `_trace_path`, `_record_provenance`, the Collect branch in `GoG.py` | [docs/Collect_fanout_instruction.md](docs/Collect_fanout_instruction.md) (decisions D1-D5 and invariants are settled) |
| `relation_inference/` model, data, or interface | [docs/relation_inference.md](docs/relation_inference.md) (written around `family`; code has moved past it) |

### Scope boundaries

Spec section 7 fixes the design. Treat each of these as a **scope change to raise with the user** before implementing:
a `Focus`/`Move` action, confidence gating on Predict, a hard observed-only rule for `e_current`, a corroboration
requirement before `Collect`, or batched multi-entity Propose.

### Deliberately broken or inert

- `--hard_only` is a no-op: `find_crucial_rel` still runs, but its only consumer was the deleted `search()`. The dropped
  gold edge can appear in results.
- `--ablate` selects the prompt and filename but ablates nothing. Re-implement either flag before reporting its numbers.
- `--sc_num > 1` can fire the GNN in both directions for one relation.
- Unreachable code in `GoG/` is ablation scaffolding, kept to switch experiments on and off. Leave it in place.
- `generate_call_count` keeps its old name for result-file compatibility; it counts Propose calls.

### Reading results

- `family` gold answer sets are incomplete. Before calling a miss an agent error, check the collected entities against
  `knowledge_graph_complete.tsv`. Fan-out precision on `family` is understated for the same reason.
- `prompts_v4` Evidence Policy can make the agent discard correct top GNN predictions (open as of 2026-09-14), so a
  "GNN adds little" result may come from prompt wording. Ask the user about its status before concluding.
- Propose evaluation gold is the single dropped `q_entity -> hard_answer` edge, always outgoing, per the benchmark.
  `run_baseline.py`'s oracle differs: it unions dropped edges in both directions.
- `evaluation.py` is the canonical evaluator (Hits@Any, Hits@Hard, Precision, Recall, F1, `HHR = Hits@Hard / Hits@Any`).

### Status (2026-11-06)

- **Toggles.** The `return` lines commented `Uncomment this line for testing ...` in `KGEnv.prob_propose_action` and
  `KGEnv.propose_action` turn Propose into a test harness that returns a tuple. Both are live now, so `obs.replace` in
  `find_answer` raises, and each question is written with `prediction: None` plus an `error`, without stopping the run.
  Comment them out for agent runs and check the `error` field of results.
- **Calibration guard.** The stale-calibration check in `ProbabilisticPropose.__init__` is commented out. After changing
  `BASELINE_CFG`, `PRIOR_PATHS`, or `PREDICT_POOL`, refit `relation_inference/baseline/calibrate.ipynb` yourself.
- **Prior models.** `relation_inference` input data was re-split by profile on 2026-10-25; confirm the `family` and
  `fb15k_237` weights were retrained after that before comparing results.
- **Extending to wikidata5m (uncommitted).** `KGInterface` and the sparse prior data are in place, and
  `relation_inference/model/wikidata5m_*.pt` exist. `KGEnv` cannot start yet because it always builds
  `ProbabilisticPropose`, even with `--propose_mode llm` (`KeyError: 'wikidata5m'`). Still missing: a `PRIOR_PATHS`
  entry, a calibration pkl, `one_shot_subgraph_config["wikidata5m"]` and its checkpoint, and a `pipeline.DATASET_DIRS`
  entry. `server_wd5m/` holds the copies used on the server.

## Running

The project interpreter is `C:/Users/MTBH/anaconda3/envs/hybkgqa/python.exe`; `python` is not on PATH. Keep the GPU to
one job at a time (4 GB). API keys live in `.env` at the repo root. Run everything from the repo root.
`GoG/hybkgqa.yaml` records the environment.

```bash
python -m GoG.GoG --dataset brink_dataset/family/test.tsv --LLM_type gpt-4o-mini        # agent
python view_traces.py results/.../xxx_failed_cases.json --limit 10                      # read trajectories
python postprocess_predictions.py <predictions.jsonl>                                   # convert, failed cases, metrics
python evaluation.py --prediction_file <predictions.json> --output_file <metrics.json>
python -m relation_inference.baseline.run_baseline --dataset family --stages oracle,text,presence,full   # 1-hop reference
python -m relation_inference.baseline.eval_prob_propose --dataset family                # prob Propose alone
```

- `--dataset` needs the form `brink_dataset/<name>/...` with forward slashes: `<name>` is `split("/")[1]` and selects
  the KG and GNN config. The TSV needs `id`, `question`, `q_entity`, `answer`, `hard_answer`.
- Non-test runs write `results/<LLM>/<dataset>/..._predictions_<timestamp>.jsonl`, each with its own log, and postprocess
  automatically. Every run is a fresh file, so `--force` and resume have no effect outside `--test`. Each record stores
  the command line that produced it.
- `--test` runs the first 3 rows into a fixed filename. `--run_fail_case <path>` reruns ids from a failed-cases JSON.
- Experiment flags live in the `GoG/GoG.py` argparse block, including `--include_score`, `--representative_only`,
  `--collect_predict`, `--type_check`, `--direction_blind_observed`, `--collect_fanout_max_hops`, `--ablate_collect`.

## Code map

- `GoG/GoG.py`: driver. `find_answer` runs the loop; Collect runs after the record is appended so `KGEnv` can attach its
  fan-out log.
- `GoG/GoG_env_tools.py`: `KGEnv`, the actions. It is reused across questions, so per-question state resets in
  `assign_query`.
- `GoG/GoG_env_utils.py`: rendering, argument parsing, path trace and replay helpers.
- `GoG/prob_propose.py`: LLM-free Propose. `tau` and `theta` are hard-coded; `propose_question` resets `theta` to 0.1
  while the constructor default is 1, so `--collect_predict` replay uses `theta = 1` until the first Propose call.
- `GoG/kg_interface.py`: `KGInterface` loads triples, relations, and NetworkX graphs. `KGEnv` needs the graphs;
  `args={"build_nx_graphs": False}` is for data preparation only.
- `GoG/gnn_interface.py`: `OneShotInterface` wraps the GNN and PPR sampler. Call `assign_graph(kg)` before `predict_topk`;
  `known=False` filters existing edges, giving the candidates Predict shows.
- `GoG/prompts_v4/`: live prompts; `GoG.py` picks the `examples*` file from the flags. `prompts_v3` and `prompts_v2` are
  archives that no longer match the dispatcher.
- `relation_inference/baseline/`: `pipeline.py` is the deterministic 1-hop Propose-Predict baseline, whose score
  `prob_propose.py` reuses. `T` is redundant there and the HHR criterion picks degenerate large-K configs, so flag both
  when comparing.
- `one_shot_subgraph/`: vendored ICLR 2024 GNN, trained separately. See its `README.md`; `--brink` reads BRINK-format data.
- `gnn_config.py`: per-dataset GNN hyperparameters and checkpoints under `data_for_LP/weights/<dataset>/`.

## Data

`brink_dataset/`, `data_for_LP/`, `results/`, and `exp_run/` are gitignored and local.

- **Two id spaces** must stay consistent. BRINK ids come from `brink_dataset/<name>/`, and GNN ids from
  `data_for_LP/id_processed/<name>/`. `OneShotInterface` builds its mapping from the assigned `KGInterface`, and
  `data_for_LP/README.md` explains why both exist. `family` relations differ by an `_of` suffix between the two; look for
  silent label mismatches like this when adding a dataset.
- **Adding a KG** touches `DATASET_CONFIG_MAP`, `one_shot_subgraph_config`, `data_for_LP/id_processed/<name>/`, plus the
  prob-Propose pieces listed under Status.
- **wikidata5m** has 20.5M edges, 4.59M entities, and 822 relations. A full `KGInterface` costs about 55 GB and 70
  minutes, so use the ~64 GB server. `KGInterface` replaces relation ids with labels from `relation_labels.txt`
  (originals in `kg.relation_ids`).
- **wikidata5m questions.** `{split}_cleaned_final.tsv` has no `hard_answer`, so `GoG.py` fails on it; only the
  `smoke_cleaned_final*.tsv` files from `create_smoke_subset.py` carry the column. Smoke contains 250 test questions, so
  exclude them from any final test numbers. Where `q` is the gold edge's tail, gold answers the inverse of the question
  text (445 rows, 31 in smoke); this is a benchmark property the user chose to leave as is.

## Conventions

- Prove a bug claim by running the code, then report the output.
- Edit live code under `GoG/` and `one_shot_subgraph/`. `exp_run/` and `*.zip` are frozen experiment snapshots.
- `offline_evaluation.py` is legacy. Root-level scripts and notebooks are independent one-offs, so read each docstring.
- Vietnamese READMEs (`GoG/README.md`, `data_for_LP/README.md`, `brink_dataset/*/README.md`) hold setup details for
  their folders.
