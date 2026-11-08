"""LLM-free Propose: calibrated P(r, d | e, text) times calibrated P(e | r, d), marginalised over slots.

  C         = candidate slots of the Propose-Predict baseline (bi-encoder U prior U present at e)
  P(r, d)   = softmax_C(score_baseline(r, d) / T_propose)          score from pipeline.rank_candidates
  S         = slots in descending P(r, d) until the cumulative mass reaches `mass`, at most `max_slots`
  P(e|r, d) = softmax over the GNN pool (top `predict_pool`, existence-filtered) of score / T_predict,
              truncated to the top `k_predict` afterwards, so the kept mass is < 1
  P(e)      = sum over S of P(r, d) * P(e | r, d)

Only predicted entities are ranked by P(e). Observed far ends of each slot in S are returned unranked, in a
shuffle seeded by (e, slot), for the caller to add unconditionally.

T_propose / T_predict were fitted (relation_inference/baseline/calibration.py) for one baseline config, one
prior, one pool size and one GNN sampler; all four are stored next to them and checked on load. The fit only
used questions whose gold slot is in C and whose gold entity is in the pool, so both probabilities are
conditional on that.

Prior variants: 'finetune' and 'pretrain' load relation_inference/model/{dataset}_{variant}.pt. 'noprior' sets
lam = 0 and k_p = 0, so the prior neither scores nor adds slots; the finetune model is still loaded only
because make_question computes its features, and the calibration stores prior_path = None.
Calibration files: results/calibration_results/calibration_results_{dataset}_{variant}_{sampler}.pkl, with
sampler 'localpush' or 'matrix' from the GNN's local_ppr.
"""
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import softmax

from GoG.GoG_env_utils import entity_sort_key
from relation_inference.baseline import pipeline as P
from relation_inference.relation_inference_interface import RelationInferenceInterface

CALIBRATION_DIR = Path("results/calibration_results")
DATASETS = ("family", "fb15k_237", "wikidata5m")
PRIOR_VARIANTS = ("finetune", "pretrain", "noprior")
DEFAULT_PRIOR_VARIANT = "finetune"
BASELINE_CFG = {"T": 0.02, "lam": 1, "gamma": 2, "k_b": 3, "k_p": 3}
NO_PRIOR_CFG = {**BASELINE_CFG, "lam": 0, "k_p": 0}  # ablate the prior


def prior_model_path(dataset: str, variant: str) -> str:
    """Relation-prior weights loaded for `variant`; 'noprior' loads finetune, whose output is unused."""
    assert variant in PRIOR_VARIANTS, variant
    return f"relation_inference/model/{dataset}_{'finetune' if variant == 'noprior' else variant}.pt"


def baseline_cfg(variant: str) -> dict:
    return NO_PRIOR_CFG if variant == "noprior" else BASELINE_CFG


# Default variant per dataset, for callers that predate the variant argument.
PRIOR_PATHS = {dataset: prior_model_path(dataset, DEFAULT_PRIOR_VARIANT) for dataset in DATASETS}
PREDICT_POOL = 200
MASS = 0.95    # tau: cumulative P(r, d) at which slot expansion stops
THETA = 1   # P(e) threshold for keeping a predicted entity; the top one is always kept


def sampler_signature(gnn_args) -> dict:
    """GNN sampler settings that change the subgraph, hence the GNN scores T_predict was fitted on.

    Local push: local_ppr, epsilon, node and edge caps, alpha. Matrix: local_ppr, topk, topm, the only
    settings the matrix sampler reads.
    """
    a = gnn_args if isinstance(gnn_args, dict) else vars(gnn_args)
    if a.get("local_ppr"):
        return {"local_ppr": True,
                "local_ppr_epsilon": float(a["local_ppr_epsilon"]),
                "max_nodes_per_query": int(a["max_nodes_per_query"]),
                "max_edges_per_query": int(a["max_edges_per_query"]),
                "local_ppr_alpha": float(a["local_ppr_alpha"])}
    return {"local_ppr": False, "topk": float(a["topk"]), "topm": float(a["topm"])}


def sampler_name(signature: dict) -> str:
    return "localpush" if signature["local_ppr"] else "matrix"


def calibration_path(dataset: str, variant: str, gnn_args) -> Path:
    name = sampler_name(sampler_signature(gnn_args))
    return CALIBRATION_DIR / f"calibration_results_{dataset}_{variant}_{name}.pkl"


def expected_calibration(dataset: str, variant: str, gnn_args) -> dict:
    """What a calibration file must store to be valid for this setup; compared key by key on load."""
    return {"cfg": baseline_cfg(variant),
            "prior_path": None if variant == "noprior" else prior_model_path(dataset, variant),
            "predict_pool": PREDICT_POOL,
            "sampler": sampler_signature(gnn_args)}


def load_calibration(dataset: str, variant: str, gnn_args) -> dict:
    path = calibration_path(dataset, variant, gnn_args)
    if not path.exists():
        raise FileNotFoundError(f"no calibration {path}; fit it with relation_inference/baseline/calibration.py")
    calib = pickle.loads(path.read_bytes())
    stale = {k: (calib.get(k), v) for k, v in expected_calibration(dataset, variant, gnn_args).items()
             if calib.get(k) != v}
    if stale:
        raise ValueError(f"{path} was fitted under a different setup (stored, expected): {stale}; "
                         f"refit with relation_inference/baseline/calibration.py")
    return calib


@dataclass
class SlotProposal:
    slot: int
    relation: str
    direction: str
    p: float                                        # P(r, d)
    predicted: list = field(default_factory=list)   # [(entity, P(e | r, d))], best first
    observed: list = field(default_factory=list)    # every far end in G_inc, seeded shuffle


@dataclass
class ProposeResult:
    e: str
    candidates: list        # [(relation, direction, P(r, d))] over all of C, best first
    slots: list             # [SlotProposal] for S
    ranked: list            # [(entity, P(e), home slot)] for every predicted entity, best first
    kept: list              # the prefix of `ranked` that passes theta

    def observed(self, n: int) -> dict:
        """slot -> first `n` observed far ends."""
        return {s.slot: s.observed[:n] for s in self.slots}


class ProbabilisticPropose:
    def __init__(self, kg, gnn, dataset: str, max_slots: int, k_predict: int,
                 mass: float = MASS, theta: float = THETA, prior_variant: str = DEFAULT_PRIOR_VARIANT):
        self.kg, self.gnn = kg, gnn
        self.max_slots, self.k_predict, self.mass, self.theta = max_slots, k_predict, mass, theta
        self.prior_variant = prior_variant
        self.cfg = baseline_cfg(prior_variant)
        calib = load_calibration(dataset, prior_variant, gnn.args)
        self.T_propose, self.T_predict = float(calib["T_propose"]), float(calib["T_predict"])

        self.space = P.SlotSpace(kg)
        model_path = prior_model_path(dataset, prior_variant)
        self.ri = RelationInferenceInterface(kg, model_path=model_path)
        self.predict_cache = {}   # (e, slot) -> (entities, raw GNN scores) over the whole pool
        print(f"Initiate ProbabilisticPropose: prior {prior_variant} ({model_path}), "
              f"calibration {calibration_path(dataset, prior_variant, gnn.args)}")

    def propose(self, text: str, e: str) -> ProposeResult:
        return self.propose_question(P.make_question(self.kg, self.ri, self.space, text, str(e)))

    def propose_question(self, q: P.Question) -> ProposeResult:
        self.theta = 0.1
        self.mass = 0.95
        slots, p_slot = self.slot_probs(q)
        chosen = []
        for s, p in zip(slots, p_slot):
            chosen.append(self._slot_proposal(q, int(s), float(p)))
            if len(chosen) >= self.max_slots or sum(c.p for c in chosen) >= self.mass:
                break

        p_entity, home = {}, {}
        for c in chosen:
            for entity, p in c.predicted:
                share = c.p * p
                p_entity[entity] = p_entity.get(entity, 0.0) + share
                if share > home.get(entity, (None, -1.0))[1]:
                    home[entity] = (c.slot, share)
        ranked = sorted(((entity, p, home[entity][0]) for entity, p in p_entity.items()),
                        key=lambda x: (-x[1], entity_sort_key(x[0])))
        kept = [x for i, x in enumerate(ranked) if i == 0 or x[1] >= self.theta]
        
        # print("len(ranked) =", len(ranked), "len(chosen) =", len(chosen), "len(kept) =", len(kept))

        candidates = [(*self.space.name(int(s)), float(p)) for s, p in zip(slots, p_slot)]
        return ProposeResult(e=q.e, candidates=candidates, slots=chosen, ranked=ranked, kept=kept)

    def slot_probs(self, q: P.Question) -> tuple[np.ndarray, np.ndarray]:
        """C ordered best first, with P(r, d) = softmax of the baseline score over C."""
        cfg = self.cfg
        sources = P.candidate_sources(q, self.space, cfg["k_b"], cfg["k_p"])
        ranked, scores = P.rank_candidates(q, self.space, np.array(sorted(sources), dtype=np.int64),
                                           cfg["T"], cfg["lam"], cfg["gamma"])
        return ranked, softmax(scores / self.T_propose)

    def _slot_proposal(self, q: P.Question, s: int, p: float) -> SlotProposal:
        relation, direction = self.space.name(s)
        entities, probs = self.predict(q.e, s)
        observed = list(q.observed.get(s, []))
        random.Random(f"{q.e}|{s}").shuffle(observed)
        return SlotProposal(slot=s, relation=relation, direction=direction, p=p,
                            predicted=list(zip(entities[:self.k_predict], probs[:self.k_predict].tolist())),
                            observed=observed)

    def predict(self, e: str, s: int) -> tuple[list, np.ndarray]:
        """GNN pool for (e, slot), existence-filtered, with calibrated P(e | r, d) over the whole pool."""
        key = (e, s)
        if key not in self.predict_cache:
            relation, direction = self.space.name(s)
            out = self.gnn.predict_topk(e, relation, direction, k=PREDICT_POOL, known=False, return_scores=True)
            scores, entities = out if out else ([], [])
            self.predict_cache[key] = ([str(x) for x in entities], np.asarray(scores, dtype=float))
        entities, scores = self.predict_cache[key]
        return entities, softmax(scores / self.T_predict) if len(scores) else scores
