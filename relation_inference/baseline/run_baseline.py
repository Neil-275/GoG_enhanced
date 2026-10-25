"""Grid-search the Propose-Predict baseline on val, then report the chosen configurations on test.

    python -m relation_inference.baseline.run_baseline --dataset family --stages oracle,text
    python -m relation_inference.baseline.run_baseline --dataset fb15k_237 --stages oracle,text,presence,full

Stages are cumulative ablations of the fusion score: `text` (lam = gamma = 0), `presence` (lam = 0), `full`.
`oracle` replaces relation selection with the gold (r, d) slots, recovered as G_complete - G_inc.
Each fusion stage exports two configurations chosen on val only: best F1, and best HHR subject to
Hits@Any >= HHR_FLOOR * Hits@Any of the best-F1 configuration.
"""
import argparse
import itertools
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from GoG.kg_interface import KGInterface
from relation_inference.baseline.pipeline import (
    PredictCache, SlotSpace, build_questions, candidate_sources, rank_candidates, relation_order,
)
from relation_inference.relation_inference_interface import RelationInferenceInterface

DEFAULT_PRIOR = {
    "family": "relation_inference/model/family_finetune.pt",
    "fb15k_237": "relation_inference/model/fb15k_237_finetune.pt",
}
GRID = {
    "T": [0.02, 0.05, 0.1, 0.2],
    "lam": [0, 0.25, 0.5, 1, 2],
    "gamma": [0, 0.5, 1, 2, 4],
    "k_b": [3, 5, 10],
    "k_p": [3, 5, 10],
    "m": [1, 2, 3],
    "K": [0, 1, 3, 5, 10],
}
STAGE_FIXED = {"text": {"lam": [0], "gamma": [0]}, "presence": {"lam": [0]}, "full": {}}
HHR_FLOOR = 0.9
M_MAX = max(GRID["m"])
K_MAX = max(GRID["K"])
GROUPS = ("all", "present", "absent")


# ----------------------------------------------------------------------------------------------------------
# QA metrics, identical to evaluation.PredictionEntry / evaluate()
# ----------------------------------------------------------------------------------------------------------

def qa_row(pred: set, q) -> np.ndarray:
    """[hits_any, hits_hard, precision, recall, f1]; an empty prediction scores 0 on everything."""
    if not pred:
        return np.zeros(5)
    gold = q.answers | q.hard
    overlap = pred & gold
    p = len(overlap) / len(pred)
    r = len(overlap) / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return np.array([bool(overlap), bool(overlap & q.hard), p, r, f1], dtype=float)


def summarize(rows: np.ndarray) -> dict:
    if len(rows) == 0:
        return {"n": 0}
    hits_any, hits_hard, p, r, f1 = rows.mean(axis=0)
    return {"n": len(rows), "Hits@Any": hits_any, "Hits@Hard": hits_hard, "Precision": p, "Recall": r,
            "F1": f1, "HHR": hits_hard / hits_any if hits_any else 0.0}


def predict_answer(q, slots, K, cache: PredictCache) -> tuple[set, dict]:
    """Observed far ends in G_inc plus GNN top-K, united over `slots`. Also returns per-slot provenance."""
    pred, detail = set(), {}
    for s in slots:
        observed = q.observed.get(s, [])
        predicted = cache.get(q.e, s, K)
        pred.update(observed)
        pred.update(predicted)
        detail[int(s)] = {"observed": observed, "predicted": predicted}
    return pred, detail


# ----------------------------------------------------------------------------------------------------------
# Ranking every (T, lam, gamma, k_b, k_p) once per question
# ----------------------------------------------------------------------------------------------------------

def ranking_grid(stage: str) -> list[dict]:
    axes = {name: STAGE_FIXED[stage].get(name, GRID[name]) for name in ("T", "lam", "gamma", "k_b", "k_p")}
    return [dict(zip(axes, values)) for values in itertools.product(*axes.values())]


def rank_all(questions, space, configs) -> list[list[dict]]:
    """For each ranking config, per question: top slots, winner sources and relation-selection hits."""
    source_cache = {}
    results = []
    for cfg in configs:
        per_q = []
        for qi, q in enumerate(questions):
            key = (qi, cfg["k_b"], cfg["k_p"])
            if key not in source_cache:
                sources = candidate_sources(q, space, cfg["k_b"], cfg["k_p"])
                source_cache[key] = (sources, np.array(sorted(sources), dtype=np.int64))
            sources, slots = source_cache[key]
            ranked, _ = rank_candidates(q, space, slots, cfg["T"], cfg["lam"], cfg["gamma"])
            oracle, oracle_rels = set(q.oracle), {s % space.n_rel for s in q.oracle}
            rels = relation_order(ranked, space.n_rel)
            per_q.append({
                "top": tuple(int(s) for s in ranked[:M_MAX]),
                "winner_sources": sources[int(ranked[0])] if len(ranked) else (),
                "rd@1": bool(oracle & set(ranked[:1].tolist())),
                "rd@3": bool(oracle & set(ranked[:3].tolist())),
                "r@1": bool(oracle_rels & set(rels[:1])),
                "r@3": bool(oracle_rels & set(rels[:3])),
                "oracle_in_C": bool(oracle & set(slots.tolist())),
            })
        results.append(per_q)
    return results


def evaluate_configs(questions, configs, rankings, cache: PredictCache) -> list[dict]:
    """Expand every ranking config over (m, K) and score it. Memoised on (question, top-m slots, K)."""
    memo = {}
    scored = []
    for cfg, per_q in zip(configs, rankings):
        for m, K in itertools.product(GRID["m"], GRID["K"]):
            rows = np.empty((len(questions), 5))
            for qi, (q, rq) in enumerate(zip(questions, per_q)):
                key = (qi, rq["top"][:m], K)
                if key not in memo:
                    memo[key] = qa_row(predict_answer(q, rq["top"][:m], K, cache)[0], q)
                rows[qi] = memo[key]
            scored.append({"config": {**cfg, "m": m, "K": K}, "metrics": summarize(rows)})
    return scored


def select(scored: list[dict]) -> dict:
    """Best F1 (first in grid order on ties), and best HHR among configs keeping HHR_FLOOR of its Hits@Any."""
    best_f1 = max(scored, key=lambda x: x["metrics"]["F1"])
    floor = HHR_FLOOR * best_f1["metrics"]["Hits@Any"]
    eligible = [x for x in scored if x["metrics"]["Hits@Any"] >= floor]
    best_hhr = max(eligible, key=lambda x: (x["metrics"]["HHR"], x["metrics"]["F1"]))
    return {"F1": best_f1, "HHR": best_hhr}


# ----------------------------------------------------------------------------------------------------------
# Test-time reporting
# ----------------------------------------------------------------------------------------------------------

def by_group(questions, values) -> dict:
    out = {}
    for group in GROUPS:
        idx = [i for i, q in enumerate(questions) if group == "all" or q.group == group]
        out[group] = values(idx)
    return out


def report_config(questions, space, cfg, cache, out_path: Path) -> dict:
    """Score one configuration on `questions`, write its predictions in evaluation.py's schema."""
    per_q = rank_all(questions, space, [cfg])[0]
    rows, records = [], []
    for q, rq in zip(questions, per_q):
        slots = rq["top"][:cfg["m"]]
        pred, detail = predict_answer(q, slots, cfg["K"], cache)
        rows.append(qa_row(pred, q))
        records.append({
            "id": q.id, "question": q.question, "prediction": sorted(pred),
            "ground_truth": sorted(q.answers), "ground_truth_entities": sorted(q.hard),
            "e_topic": q.e, "group": q.group,
            "chosen_rd": [space.name(s) for s in slots],
            "oracle_rd": [space.name(s) for s in q.oracle],
            "winner_sources": list(rq["winner_sources"]),
            "ranked_candidates": [space.name(s) for s in rq["top"]],
            "answer_detail": {"|".join(space.name(s)): d for s, d in detail.items()},
        })
    rows = np.array(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=1))

    def selection(idx):
        sub = [per_q[i] for i in idx]
        return {k: float(np.mean([x[k] for x in sub])) if sub else None
                for k in ("rd@1", "rd@3", "r@1", "r@3", "oracle_in_C")}

    def winner_sources(idx):
        counts = Counter()
        for i in idx:
            hit = "hit" if per_q[i]["rd@1"] else "miss"
            counts[f"{'+'.join(per_q[i]['winner_sources'])} ({hit})"] += 1
        return dict(counts.most_common())

    return {
        "config": cfg,
        "qa": by_group(questions, lambda idx: summarize(rows[idx])),
        "selection": by_group(questions, selection),
        "winner_sources": by_group(questions, winner_sources),
        "predictions": str(out_path),
    }


def oracle_rows(questions, cache, K_values) -> dict:
    out = {}
    for K in K_values:
        rows = np.array([qa_row(predict_answer(q, q.oracle, K, cache)[0], q) for q in questions])
        out[K] = by_group(questions, lambda idx: summarize(rows[idx]))
    return out


# ----------------------------------------------------------------------------------------------------------

def load_questions(args, kg, space, out_dir):
    path = out_dir / "questions.pkl"
    if path.exists() and not args.rebuild:
        return pickle.loads(path.read_bytes())
    ri = RelationInferenceInterface(kg, model_path=args.prior_path)
    questions = {split: build_questions(args.dataset, split, kg, ri, space) for split in ("val", "test")}
    path.write_bytes(pickle.dumps(questions))
    return questions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DEFAULT_PRIOR))
    parser.add_argument("--stages", default="oracle,text,presence,full")
    parser.add_argument("--prior_path", default=None)
    parser.add_argument("--out_dir", default="results/baseline")
    parser.add_argument("--rebuild", action="store_true", help="recompute question features (cos, prior, presence)")
    args = parser.parse_args()
    args.prior_path = args.prior_path or DEFAULT_PRIOR[args.dataset]
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    prior_tag = Path(args.prior_path).stem
    out_dir = Path(args.out_dir) / args.dataset / prior_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    kg = KGInterface(args.dataset)
    space = SlotSpace(kg)
    questions = load_questions(args, kg, space, out_dir)
    cache = PredictCache(args.dataset, kg, space, Path(args.out_dir) / args.dataset / "gnn_cache.pkl", K_MAX)

    report_path = out_dir / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["prior_path"] = args.prior_path
    report["counts"] = {split: dict(Counter(q.group for q in qs), n=len(qs)) for split, qs in questions.items()}

    if "oracle" in stages:
        cache.fill((q.e, s) for split in ("val", "test") for q in questions[split] for s in q.oracle)
        report["oracle"] = {split: oracle_rows(questions[split], cache, GRID["K"]) for split in ("val", "test")}

    for stage in (s for s in stages if s in STAGE_FIXED):
        configs = ranking_grid(stage)
        val_rankings = rank_all(questions["val"], space, configs)
        cache.fill((q.e, s) for per_q in val_rankings for q, rq in zip(questions["val"], per_q) for s in rq["top"])
        scored = evaluate_configs(questions["val"], configs, val_rankings, cache)
        chosen = select(scored)

        stage_report = {"n_configs": len(scored)}
        for target, pick in chosen.items():
            cfg = pick["config"]
            test_rankings = rank_all(questions["test"], space, [cfg])[0]
            cache.fill((q.e, s) for q, rq in zip(questions["test"], test_rankings) for s in rq["top"][:cfg["m"]])
            stage_report[target] = {
                "val": pick["metrics"],
                "test": report_config(questions["test"], space, cfg, cache,
                                      out_dir / f"{stage}_{target}_test_predictions.json"),
            }
        stage_report["top_val"] = sorted(scored, key=lambda x: -x["metrics"]["F1"])[:10]
        report[stage] = stage_report

    report_path.write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
