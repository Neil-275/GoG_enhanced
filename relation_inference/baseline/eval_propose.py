"""Evaluate the LLM-free Propose (and Propose -> Predict) on the protocol of notebooks/test_generate.ipynb.

    python -m relation_inference.baseline.eval_propose --dataset family --split test
    python -m relation_inference.baseline.eval_propose --dataset fb15k_237 --split test --n_propose 3

Case filter, as in `cal_propose_accuracy` / `test_propose_predict`: take the first hard answer, keep the question
only if exactly one dropped edge (G_complete - G_inc) goes q_entity -> hard_answer. The target (r, d) is that edge.
The proposal is the baseline's ranked slot list (`candidate_sources` + `rank_candidates`), truncated to
`--n_propose` slots, standing in for the LLM's `propose_action` output.

Metrics (intended semantics of the notebook, computed per question):
  Accuracy           target (r, d) is among the proposed slots
  Accuracy (rel)     target relation is among the proposed slots, any direction
  Hit@k              target (r, d) is in the top k of the full ranked list
  PP Accuracy        hard answer is in the GNN top-`--k_predict` of some proposed slot (test_propose_predict)
  PP Norm. precision 1 / len(concatenated GNN lists) when PP hits, else 0
"""
import argparse
import json
from pathlib import Path

import numpy as np

from GoG.kg_interface import KGInterface
from relation_inference.baseline import pipeline as P
from relation_inference.baseline import run_baseline as RB
from relation_inference.relation_inference_interface import RelationInferenceInterface

HITS_AT = (1, 3, 5)


def default_config(dataset: str, prior_path: str) -> dict:
    """Best-F1 `full` configuration chosen on val by run_baseline, if it has been run."""
    report = Path(f"results/baseline/{dataset}/{Path(prior_path).stem}/report.json")
    if report.exists():
        return dict(json.loads(report.read_text())["full"]["F1"]["test"]["config"])
    return {"T": 0.02, "lam": 1, "gamma": 2, "k_b": 3, "k_p": 3, "m": 1, "K": 3}


def target_slot(kg: KGInterface, space: P.SlotSpace, row: dict):
    """The benchmark's target: the single dropped edge q_entity -> first hard answer, else None."""
    e, hard = row["q_entity"][0], str(row["hard_answer"][0])
    inc = kg.incomplete_kg
    complete = kg.complete_kg
    edges = complete[(complete["head"] == e) & (complete["tail"] == hard)]
    kept = inc[(inc["head"] == e) & (inc["tail"] == hard)]
    dropped = set(edges["relation"]) - set(kept["relation"])
    return (space.slot(dropped.pop(), "outgoing"), hard) if len(dropped) == 1 else (None, hard)


def evaluate(questions, rows, kg, space, cfg, n_propose, k_predict, cache):
    records = []
    for q, row in zip(questions, rows):
        target, hard = target_slot(kg, space, row)
        if target is None:
            continue
        sources = P.candidate_sources(q, space, cfg["k_b"], cfg["k_p"])
        ranked, _ = P.rank_candidates(q, space, np.array(sorted(sources), dtype=np.int64),
                                      cfg["T"], cfg["lam"], cfg["gamma"])
        ranked = [int(s) for s in ranked]
        proposed = ranked[:n_propose]
        records.append({"q": q, "hard": hard, "target": target, "ranked": ranked, "proposed": proposed})

    cache.fill({(r["q"].e, s) for r in records for s in r["proposed"]})

    predictions, scores = [], []
    for r in records:
        q, target, ranked, proposed = r["q"], r["target"], r["ranked"], r["proposed"]
        per_slot = {s: cache.get(q.e, s, k_predict) for s in proposed}
        predicted = [x for s in proposed for x in per_slot[s]]
        rank = ranked.index(target) + 1 if target in ranked else None
        pp_hit = r["hard"] in predicted
        predictions.append({
            "id": q.id, "question": q.question, "e_topic": q.e,
            "ranked_candidates": [space.name(s) for s in ranked],
            "proposed": [space.name(s) for s in proposed],
            "predicted": {"|".join(space.name(s)): v for s, v in per_slot.items()},
        })
        scores.append({
            "id": q.id, "hard_answer": r["hard"], "target": space.name(target), "rank": rank,
            "acc": target in proposed,
            "acc_rel": space.rel_of(target) in {space.rel_of(s) for s in proposed},
            **{f"hit@{k}": rank is not None and rank <= k for k in HITS_AT},
            "pp_acc": pp_hit,
            "pp_norm_precision": 1 / len(predicted) if pp_hit else 0.0,
            "n_predicted": len(predicted),
        })
    return predictions, scores


def summarize(out: list[dict]) -> dict:
    keys = ["acc", "acc_rel", *(f"hit@{k}" for k in HITS_AT), "pp_acc", "pp_norm_precision"]
    return {"total": len(out), **{k: float(np.mean([o[k] for o in out])) if out else 0.0 for k in keys}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(RB.DEFAULT_PRIOR))
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--prior_path", default=None)
    parser.add_argument("--n_propose", type=int, default=None, help="proposed slots per question (default: config m)")
    parser.add_argument("--k_predict", type=int, default=10, help="GNN top-k per proposed slot, as in the notebook")
    for name in ("T", "lam", "gamma"):
        parser.add_argument(f"--{name}", type=float, default=None)
    for name in ("k_b", "k_p"):
        parser.add_argument(f"--{name}", type=int, default=None)
    parser.add_argument("--out_prefix", default=None,
                        help="writes <prefix>_predictions.json and <prefix>_evaluation.json "
                             "(default: results/baseline/<ds>/eval_propose_<split>)")
    args = parser.parse_args()
    args.prior_path = args.prior_path or RB.DEFAULT_PRIOR[args.dataset]

    cfg = default_config(args.dataset, args.prior_path)
    cfg.update({k: getattr(args, k) for k in ("T", "lam", "gamma", "k_b", "k_p") if getattr(args, k) is not None})
    n_propose = args.n_propose or cfg["m"]
    if args.k_predict > RB.K_MAX:
        parser.error(f"--k_predict must be <= {RB.K_MAX} (GNN cache depth)")

    kg = KGInterface(args.dataset)
    space = P.SlotSpace(kg)
    ri = RelationInferenceInterface(kg, model_path=args.prior_path)
    cache = P.PredictCache(args.dataset, kg, space, Path(f"results/baseline/{args.dataset}/gnn_cache.pkl"), RB.K_MAX)

    questions = P.build_questions(args.dataset, args.split, kg, ri, space)
    rows = P._read_split(args.dataset, args.split).to_dict(orient="records")
    predictions, scores = evaluate(questions, rows, kg, space, cfg, n_propose, args.k_predict, cache)
    stats = summarize(scores)

    print(f"dataset={args.dataset} split={args.split} prior={Path(args.prior_path).stem}")
    print(f"config: T={cfg['T']} lam={cfg['lam']} gamma={cfg['gamma']} k_b={cfg['k_b']} k_p={cfg['k_p']} "
          f"n_propose={n_propose} k_predict={args.k_predict}")
    print(f"Valid cases: {stats['total']} / {len(questions)}")
    print(f"Accuracy (r, d):        {stats['acc']:.4f}")
    print(f"Accuracy (relation):    {stats['acc_rel']:.4f}")
    for k in HITS_AT:
        print(f"Hit@{k} (r, d):           {stats[f'hit@{k}']:.4f}")
    print(f"Propose->Predict acc:   {stats['pp_acc']:.4f}")
    print(f"Normalized precision:   {stats['pp_norm_precision']:.4f}")

    prefix = args.out_prefix or f"results/baseline/{args.dataset}/eval_propose_{args.split}"
    config = {"dataset": args.dataset, "split": args.split, "prior_path": args.prior_path,
              **cfg, "n_propose": n_propose, "k_predict": args.k_predict}
    pred_path, eval_path = Path(f"{prefix}_predictions.json"), Path(f"{prefix}_evaluation.json")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(json.dumps({"config": config, "predictions": predictions}, indent=1))
    eval_path.write_text(json.dumps({"config": config, "summary": stats, "per_question": scores}, indent=1))
    print(f"wrote {pred_path}\nwrote {eval_path}")


if __name__ == "__main__":
    main()
