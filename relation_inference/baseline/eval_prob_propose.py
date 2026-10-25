"""Evaluate the LLM-free probabilistic Propose (GoG/prob_propose.py) end to end, without the agent.

    python -m relation_inference.baseline.eval_prob_propose --dataset family --split test

Prediction set per question = kept predicted entities (P(e) >= theta, at least the top one)
                              U the first `--n_observed` observed far ends of every selected slot.
QA metrics are run_baseline's (evaluation.py semantics), split by present / absent group. On top:
  pred_MRR / pred_Hits@1   rank of the best hard answer in the full predicted ranking (before theta)
  oracle_in_C / oracle_in_S a gold slot is among the candidates / the selected slots
  top_p / rd@1              mean P(r, d) of the top slot vs. how often it is a gold slot (calibration check)
  n_slots                   GNN calls per question
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from GoG.gnn_interface import OneShotInterface
from GoG.kg_interface import KGInterface
from GoG.prob_propose import MASS, THETA, ProbabilisticPropose
from relation_inference.baseline import pipeline as P
from relation_inference.baseline import run_baseline as RB

EXTRA = ("pred_MRR", "pred_Hits@1", "oracle_in_C", "oracle_in_S", "top_p", "rd@1", "n_slots", "n_pred")


def evaluate(prop: ProbabilisticPropose, questions, n_observed: int):
    qa_rows, extra_rows, records = [], [], []
    for q in questions:
        res = prop.propose_question(q)
        observed = res.observed(n_observed)
        pred = {e for e, _, _ in res.kept} | {x for v in observed.values() for x in v}
        ranked = [e for e, _, _ in res.ranked]
        hard_ranks = [ranked.index(h) + 1 for h in q.hard if h in ranked]
        best = min(hard_ranks) if hard_ranks else None
        oracle = set(q.oracle)
        selected = {s.slot for s in res.slots}
        top_slot = prop.space.slot(*res.candidates[0][:2]) if res.candidates else None

        qa_rows.append(RB.qa_row(pred, q))
        extra_rows.append([
            1 / best if best else 0.0, float(best == 1),
            float(any(prop.space.slot(r, d) in oracle for r, d, _ in res.candidates)),
            float(bool(oracle & selected)),
            res.candidates[0][2] if res.candidates else 0.0,
            float(top_slot in oracle) if top_slot is not None else 0.0,
            len(res.slots), len(res.kept),
        ])
        records.append({
            "id": q.id, "question": q.question, "e_topic": q.e, "group": q.group,
            "prediction": sorted(pred), "ground_truth": sorted(q.answers), "ground_truth_entities": sorted(q.hard),
            "oracle_rd": [prop.space.name(s) for s in q.oracle],
            "candidates": [[r, d, round(p, 4)] for r, d, p in res.candidates],
            "slots": [{"rd": [s.relation, s.direction], "p": round(s.p, 4), "observed": observed[s.slot],
                       "gnn_top": [[e, round(p, 4)] for e, p in s.predicted]} for s in res.slots],
            "ranked": [[e, round(p, 4), list(prop.space.name(s))] for e, p, s in res.ranked],
            "kept": [e for e, _, _ in res.kept],
        })
    return np.array(qa_rows), np.array(extra_rows), records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(RB.DEFAULT_PRIOR))
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--max_slots", type=int, default=3, help="as --max_selected_relations in GoG")
    parser.add_argument("--k_predict", type=int, default=3, help="as --predict_topk in GoG")
    parser.add_argument("--n_observed", type=int, default=5, help="as --n_related_triples in GoG")
    parser.add_argument("--out_dir", default="results/prob_propose")
    args = parser.parse_args()

    kg = KGInterface(args.dataset)
    gnn = OneShotInterface(args.dataset, kg.n_ent, kg.n_rel)
    gnn.assign_graph(kg)
    prop = ProbabilisticPropose(kg, gnn, args.dataset, max_slots=args.max_slots, k_predict=args.k_predict)

    out_dir = Path(args.out_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "predict_cache.pkl"
    if cache_path.exists():
        prop.predict_cache = pickle.loads(cache_path.read_bytes())

    questions = P.build_questions(args.dataset, args.split, kg, prop.ri, prop.space)
    qa, extra, records = evaluate(prop, questions, args.n_observed)
    cache_path.write_bytes(pickle.dumps(prop.predict_cache))

    def summary(idx):
        out = RB.summarize(qa[idx])
        out.update({k: float(v) for k, v in zip(EXTRA, extra[idx].mean(axis=0))} if len(idx) else {})
        return out

    config = {"dataset": args.dataset, "split": args.split, "mass": MASS, "theta": THETA,
              "max_slots": args.max_slots, "k_predict": args.k_predict, "n_observed": args.n_observed,
              "T_propose": prop.T_propose, "T_predict": prop.T_predict}
    report = {"config": config, "metrics": RB.by_group(questions, summary)}
    (out_dir / f"{args.split}_predictions.json").write_text(json.dumps(records, indent=1))
    (out_dir / f"{args.split}_report.json").write_text(json.dumps(report, indent=1))

    print(json.dumps(config))
    cols = ("n", "Hits@Any", "Hits@Hard", "F1", "HHR") + EXTRA
    print("group    " + " ".join(f"{c:>11}" for c in cols))
    for group, m in report["metrics"].items():
        print(f"{group:<8} " + " ".join(f"{m.get(c, 0):>11.3f}" for c in cols))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
