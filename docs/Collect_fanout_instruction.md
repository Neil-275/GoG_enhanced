# Collect Relation-Path Fan-Out — Resolved Decisions

Companion to `New_agent_framework_instruction.md`. That file is the spec; this
file records the points where the spec was underspecified, the decision that
was made, and why — so they are not re-litigated. **Implemented and in effect.**

Status: implemented in `GoG/GoG_env_tools.py` (`provenance`, `_record_provenance`,
`_trace_path`, `_observed_neighbors`, `_replay_path`, `_collect_one`,
`collect_action`) and `GoG/GoG.py` (driver routes `Collect` through the env).

---

## 0. What "fan-out" means

The frontier is **the set of entities that satisfy the relation-type path**, and
its size is the fan-out count. Each hop replaces the frontier with everything
reachable from anything currently in it via that `(relation, direction)`, so the
final frontier is every entity satisfying the whole sequence from the root. This
is what recovers the siblings that Propose's top-K cap hid — the reason the
mechanism exists.

---

## D1. `e_current_pool` lives in the Observation text, not in a variable

There is **no data structure** that feeds e_current selection. `explored_entities`
in `KGEnv` is written and never read. The agent picks its next `e_current` from
the trajectory text, per §4 of the main spec ("any entity ID that has appeared
anywhere earlier in this trajectory").

**Therefore spec §5 is binding on the Observation string.** After a Collect, do
**not** list the fan-out members. The Collect observation echoes only the ids the
agent itself named, plus a bare count:

```
Collected the answers: [3 | 4]
Also recorded 1 further answer(s) reachable from the topic entity by the same relation path.
```

Bulk-collected entities also get no provenance record and are not added to
`explored_entities`. They exist only in `R_t`.

When touching the Collect observation format, preserve this. Listing the members
would satisfy §5 in code while violating it in effect, and would inflate context
on wide fan-outs.

## D2. Replay succeeds but `e` is not in the frontier → keep `e`

Spec §4 asserts `e in frontier`. **This assertion can legitimately fail** and must
not be an assert. Union `e` into the frontier and log a warning.

Concrete case, real data in `brink_dataset/family` (1508 instances of this shape
exist there; this is one):

- `(8, aunt_of, 26)` is a **dropped** edge — present in
  `knowledge_graph_complete.tsv`, absent from `knowledge_graph_incomplete.tsv`.
  Only Predict can surface `26`.
- `(8, aunt_of, *)` still has **21 observed tails** in G_inc
  (`13, 14, 17, 21, 24, 27, 38, 39, 73, 74, 75, 76, 77, 83, 86, 95, 97, 121, 129, 310, 311`).

Trajectory: `Propose[8 | who are 8's nieces and nephews]` shows some of the 21
under `(observed, direct edge)` and `26` under `(predicted, unverified)`. The
agent collects `26`. Provenance of `26` is `(parent=8, aunt_of, outgoing)`, so
the traced path is `[(aunt_of, outgoing)]` with root `8`.

Replaying that path against G_inc returns the 21 observed tails — **non-empty, so
no fallback fires** — but `26` is not among them, because its edge was dropped.
The three options:

| | result | consequence |
|---|---|---|
| (a) keep `e`, union | 21 + `26` | correct |
| (b) trust the frontier, drop `e` | 21 only | **discards the hard answer** — the exact entity the GNN exists to recover. Destroys Hits@Hard. |
| (c) treat as failed replay | `26` only | loses the 21 real siblings; defeats fan-out |

**(a) is the decision**, confirmed explicitly after the case above was put to
the author — not a default picked by an implementer. Do not "fix" it back to an
assert. It means a predicted entity can enter `R_t` alongside a fan-out, a case
spec §4 does not contemplate; that is intended.

## D3. `Collect[a | b | c]` — per-entity union, deduped by traced path

The spec is written for `Collect(e)`; the prompt
(`GoG/prompts_v4/examples`) advertises a pipe-separated list. Run the algorithm
per entity and union the results.

Entities that trace to the same `(root, path)` **replay once** — `collect_action`
holds a `replay_cache` keyed on that pair. Three collected siblings on one
relation path therefore cost one graph traversal, not three.

## D4. Path-length gating is a separate axis from cardinality gating

Spec §4 forbids **cardinality** gating (no singular/plural classification) and
defers precision impact to measurement. It says nothing about **path length**,
which is a different axis and is gated by a flag:

```
--collect_fanout_max_hops N     # 0 = unlimited (default, = spec behavior)
```

Default is unlimited so the spec'd behavior is what runs and the measurement
still happens. `--collect_fanout_max_hops 1` is the ablation.

**Why the flag exists** — replaying the *correct* path for all 198
`family/test` questions (so this isolates the mechanism's own cost from bad
agent reasoning):

```
                    1-hop paths (188 q)    2-hop paths (10 q)
fan-out precision          0.869                 0.427
fan-out recall             0.504                 0.767
gold set size (mean)       5.57                  2.10
fan-out size (mean/max)    3.36 / 14             4.00 / 8
```

1-hop is strongly F1-positive (single-entity collect would be precision 1.0 /
recall ~0.2). 2-hop inverts: precision more than halves while gold sets are
*smaller*; estimated F1 ~0.55 vs ~0.65 for single-entity. Small n — confirm at
scale before concluding.

## D5. Frontier-size cap is opt-in

```
--collect_fanout_max_entities N  # 0 = no cap (default, = spec behavior)
```

When a hop's frontier exceeds `N`, fall back to single-entity collect for that
entity. Default off, so spec behavior is unchanged.

`family` maxes out at a frontier of 14, so this is not needed there. It is a
safety valve for `fb15k_237` / `wikidata5m`, where a multi-hop relation-type
fan-out from a hub entity can return a huge frontier straight into `R_t`.
`records[*]["collect"][*]["hop_sizes"]` logs every hop width, so inspect that
after the first run on a large KG rather than guessing a cap up front.

---

## Interpreting the metrics (read before concluding anything)

`evaluation.py` computes `precision = |overlap| / |prediction|`, so
over-collection costs precision linearly.

**`family` gold answer sets are incomplete** — correct agent reasoning already
scores as a miss there. Fan-out interacts with this badly in a specific way: it
adds *genuinely correct* same-relation entities that the gold set may simply not
list, so the measured precision drop **overstates** the real one. Verify a
handful of fan-out members against `knowledge_graph_complete.tsv` before reading
an F1 delta as a verdict on the mechanism.

## Invariants not to break

- Provenance is **first-production-wins**. A later re-appearance via a different
  path must not overwrite it; that is what makes tracing well-defined.
- Provenance is recorded for **predicted** candidates too. The observed/predicted
  distinction is applied at collect time — by whether the path replays against
  G_inc — not at record time.
- Topic entities are roots and carry no provenance record; tracing terminates at
  any of them (or at any entity lacking a record, which collapses to a
  single-entity collect).
- `_trace_path` guards cycles with a visited set and falls back rather than
  hanging.
- `_observed_neighbors` reads `kg.incomplete_graph_nx`, built from the same
  `incomplete_kg` DataFrame that `observed_index` comes from, so a hop surfaced
  as observed always replays.
- All `KGEnv` per-question state resets in `assign_query`, not `__init__` — the
  env is reused across questions. `provenance` and `topic_entity_set` are reset
  there.
