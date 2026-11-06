# Implementation Spec: Propose–Collect–Finish Agent Loop

## Context for the coding agent

This replaces the existing GoG-derived agent loop's action space. The prior
architecture used **Search** (retrieve local relations of an entity) as a
separate action from **Generate** (Propose + Predict, triggered only on a
detected "structural break" where the target relation was absent from
Search's output). This spec **removes Search as a standalone action** and
**removes the "structural break" trigger condition**. Every entity visit now
goes through Propose directly, and Predict (the GNN link predictor) runs
unconditionally on every relation Propose selects — there is no longer an
activation policy that decides whether Predict runs.

**Before touching code:** locate the current implementation of the
Thinking–Searching–Generating loop (likely a class/module driving the
Thought → Action → Observation cycle), the existing `Search` action handler,
the existing `Generate` module (Propose + Predict submodules), and the
`Collect`/`Finish` action handlers referenced in the thesis's Chapter 3
(Collect/Finish were already split from a single Finish — that part is
unchanged and should be reused as-is). Do not assume specific file names;
grep for `Search`, `Propose`, `Predict`, `Generate`, `Collect`, `Finish`
action identifiers to find them.

---

## 1. Action space

Final action space: **`Propose`, `Collect`, `Finish`**. `Search` is removed
entirely as an action the LLM can call. Its two prior roles (fetch local
relations of an entity; advance which entity is being explored) are now both
absorbed into `Propose`, per sections 2 and 4 below.

Do not add a fourth "navigation" action (e.g. `Focus`/`Move`) — entity
selection for the next iteration happens implicitly inside the LLM's Thought,
not as a separate tool call. See section 4.

## 2. Propose: single-entity, per-call contract

`Propose(q_sub, e_current)` — takes exactly **one** entity per call. Do not
batch multiple candidate entities into a single Propose call (rejected in
design: multiplies GNN calls by `entities × relations`, and Predict already
runs on every selected relation per call — see section 3 — so batching would
compound an already GPU-constrained cost).

Internal steps, in order (this reuses the existing Propose implementation
almost unchanged — the only new requirement is section 2a):

1. Bi-Encoder top-5 relation retrieval against the full relation vocabulary
   (candidates for `r_target`) — **unchanged from current implementation**.
2. Bi-Encoder top-5 local-context retrieval (outgoing + incoming, ≤10 total)
   against relations actually connected to `e_current` in `G_inc` —
   **unchanged from current implementation**.
3. LLM selects a **subset** of relations from the top-5 global candidates
   (not a single `r_target` — this is a change from the original
   single-relation-per-call contract, since every selected relation now
   triggers Predict independently; see section 3). LLM may select zero
   (Propose returns empty, same as before).
4. For each selected relation, LLM determines direction
   `d ∈ {outgoing, incoming}` — unchanged.

### 2a. New requirement: expose local context in the Observation

The local-context relations fetched in step 2 (previously consumed only
internally by the relation-selection prompt) must now also be **returned to
the agent as part of the Observation**, not just used internally. This is
required because, with Search removed, this is the *only* schema signal the
LLM has for choosing which entity to visit next — see section 4. Find where
Propose currently constructs its return value / Observation payload and add
the local-context relation list (both directions) to it, labeled clearly
(e.g. `local_context: {outgoing: [...], incoming: [...]}`), even for
relations that were not selected as `r_target`.

## 3. Predict: always-on, no activation policy

For **every** relation selected in step 3 above (not just one), run:

1. Existing-edge lookup in `G_inc` for that `(e_current, relation, d)` →
   "observed" candidates if found.
2. Predict/GNN inference (existing One-Shot-Subgraph pipeline, unchanged) →
   "predicted" candidates, with **existence filtering already applied**
   (unchanged — filter before top-K, as currently implemented) so predicted
   and observed sets are always disjoint.

This removes whatever activation-policy code currently decides *whether* to
run Predict (the ~13%/~2% activation-rate logic mentioned in prior
performance notes). Locate that gating logic and remove it — Predict must run
unconditionally per selected relation, every call.

## 4. Entity selection for the next iteration (no new module)

There is **no dedicated entity-selection function or action**. This is
implicit in the LLM's next Thought, exactly matching how the original GoG
Search-target selection worked (free-form choice from context, no scoring
algorithm) — do not build a ranking/selection module for this.

**Candidate pool the LLM can choose `e_current` from:** any entity ID that
has appeared anywhere in the trajectory's Observation history so far —
`e_topic`, local-context entities from any prior Propose call, observed-edge
entities, predicted entities, and entities already in `R_t` (Collect'd).
**Full trajectory history is in scope, not just the most recent Observation**
— do not add recency-scoping/windowing that restricts selection to only the
latest Observation.

Implementation implication: whatever context-window/memory object currently
accumulates Observations for the LLM prompt must retain entity IDs from
*all* prior iterations in a form the LLM can reference by ID when choosing
the next `Propose` call's `e_current` argument — verify nothing in the
current context-trimming logic drops earlier entities before this.

## 5. Observation formatting: observed vs. predicted labeling

When Propose+Predict return their combined result for a relation, format the
Observation so **observed (direct-edge) candidates are visually distinct
from predicted candidates** — e.g. list observed first, and label each group
explicitly (`(observed, direct edge)` / `(predicted, unverified)`). This is
not just cosmetic: it doubles as a **soft prior for entity-selection**, per
the design decision that observed entities should be treated as
higher-confidence continuation points than Predict-only ones, without a hard
rule forbidding navigation through Predict-only entities.

If there's a system/Thinking-step prompt template, add a short instruction
there (not hard-coded logic) noting that observed entities are stronger
evidence for continuing exploration than predicted ones, but predicted
entities remain usable if they better match the sub-question's semantics.

## 6. Collect / Finish

**Unchanged** from the existing split-Collect/Finish design already
implemented (per Chapter 3): `Collect(e)` accumulates into `R_t` without
halting; `Finish` is a pure stop signal. No code changes needed here beyond
ensuring these two actions remain available alongside the new `Propose`
action (replacing `Search`/`Generate` as separate actions).

## 7. Explicitly NOT in this version (do not implement)

- No `Focus`/`Move` navigation action.
- No confidence-threshold gating on Predict candidates.
- No hard rule forcing `e_current` to only ever be an observed (never
  predicted-only) entity.
- No corroboration requirement before `Collect` (a Predict-only chain can be
  collected as-is).
- No multi-entity batched Propose calls.

These are documented as known limitations (silent error compounding through
unverified Predict-only hops; no bound on chained low-confidence
navigation), not guarded against in code. If asked to add any of the above
later, flag it as a scope change against this spec rather than assuming it's
in scope.

## 8. Suggested validation after implementation

- Confirm `Search` and the old structural-break detection are fully removed
  from the action-dispatch logic (not just unreachable).
- Confirm Predict is invoked once per selected relation per Propose call,
  with no remaining gate/condition.
- Confirm the Observation payload includes local-context relations (2a) and
  observed/predicted labeling (5) in every Propose response.
- Run the two-iteration trace from the design discussion (employer →
  headquarters style query) end-to-end and check the Observation text at
  each step matches the labeling described in sections 2a and 5.