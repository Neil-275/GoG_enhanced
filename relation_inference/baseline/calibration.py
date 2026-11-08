"""Temperature calibration for LLM-free Propose (GoG/prob_propose.py), for any dataset and GNN sampler.

Python version of relation_inference/baseline/calibrate.ipynb (cells 6-10, and the wikidata5m section 34-38),
which stays as it was. One call fits, for one loaded KG and GNN:

  T_predict  on the GNN scores of the gold (relation, direction): pool of PREDICT_POOL, existence-filtered,
             fitted on questions whose hard answer is in the pool. The GNN does not depend on the prior, so
             every variant shares it.
  T_propose  on the Propose-Predict baseline score over C, per prior variant (finetune, pretrain, noprior),
             fitted on questions whose gold slot is in C.

Questions are those with exactly one dropped edge q_entity -> hard_answer. Each T is the minimum of the mean
NLL over the notebook's 50-point grid, refined on a 201-point grid between the two coarse neighbours of that
minimum. Files go to prob_propose.calibration_path(), named by dataset, variant and sampler.

Run from the repo root, for family or fb15k_237 (wikidata5m goes through notebooks/run_eval_bundle.py):
    python -m relation_inference.baseline.calibration --dataset family [--local_ppr]
"""
import argparse
import datetime
import pickle
from types import SimpleNamespace

import numpy as np
from scipy.special import log_softmax

from GoG.prob_propose import (
    PREDICT_POOL,
    PRIOR_VARIANTS,
    baseline_cfg,
    calibration_path,
    expected_calibration,
    prior_model_path,
    sampler_signature,
)
from relation_inference.baseline import pipeline as P
from relation_inference.relation_inference_interface import RelationInferenceInterface

COARSE_GRID = np.logspace(-1.3, 1.3, 50)   # the notebook's grid
FINE_POINTS = 201
CALIBRATION_SPLITS = {"family": "val", "fb15k_237": "val", "wikidata5m": "val_calibration"}


def gold_slot(case: P.Question, drop_edges):
    """(direction, relation) of the single dropped edge q -> hard answer, or None. Notebook cell 7."""
    hard_answer = sorted(case.hard)[0]
    edges = drop_edges[(drop_edges["head"] == case.e) & (drop_edges["tail"] == hard_answer)]
    if len(edges) != 1:
        return None
    direction = "outgoing" if edges["head"].item() == case.e else "incoming"
    return direction, edges["relation"].values[0]


def propose_data(questions, drop_edges, space, cfg):
    """Baseline scores over C and the gold position, for questions whose gold slot is in C. Notebook cell 8."""
    scores, golds, valid = [], [], 0
    for case in questions:
        gold = gold_slot(case, drop_edges)
        if not gold:
            continue
        valid += 1
        direction, relation = gold
        sources = P.candidate_sources(case, space, cfg["k_b"], cfg["k_p"])
        ranked, score = P.rank_candidates(case, space, np.array(sorted(sources), dtype=np.int64),
                                          cfg["T"], cfg["lam"], cfg["gamma"])
        pos = np.flatnonzero(ranked == space.slot(relation, direction))
        if pos.size:
            scores.append(score)
            golds.append(int(pos[0]))
    return scores, golds, {"valid": valid, "gold_in_C": len(scores)}


def predict_data(questions, drop_edges, gnn):
    """GNN pool scores for the gold slot and the hard answer's position, when it is in the pool. Cell 9."""
    scores, golds, valid = [], [], 0
    for case in questions:
        gold = gold_slot(case, drop_edges)
        if not gold:
            continue
        valid += 1
        direction, relation = gold
        out = gnn.predict_topk(case.e, relation, direction, return_scores=True, known=False, k=PREDICT_POOL)
        gnn_scores, cand_ids = out if out else ([], [])
        hard_answer = sorted(case.hard)[0]
        if hard_answer in cand_ids:
            scores.append(gnn_scores)
            golds.append(cand_ids.index(hard_answer))
    return scores, golds, {"valid": valid, "gold_in_pool": len(scores)}


def mean_nll(scores, golds, T):
    return float(np.mean([-log_softmax(np.asarray(s, dtype=float) / T)[g] for s, g in zip(scores, golds)]))


def fit_temperature(scores, golds):
    """Coarse grid as in the notebook, then a fine grid between the coarse minimum's neighbours."""
    if not scores:
        raise ValueError("no question to fit on")
    coarse = [mean_nll(scores, golds, T) for T in COARSE_GRID]
    i = int(np.argmin(coarse))
    lo, hi = COARSE_GRID[max(i - 1, 0)], COARSE_GRID[min(i + 1, len(COARSE_GRID) - 1)]
    fine_grid = np.logspace(np.log10(lo), np.log10(hi), FINE_POINTS)
    fine = [mean_nll(scores, golds, T) for T in fine_grid]
    j = int(np.argmin(fine))
    return {"T": float(fine_grid[j]), "nll": float(fine[j]),
            "coarse_T": float(COARSE_GRID[i]), "coarse_nll": float(coarse[i]),
            # A minimum on the grid's edge means the true one may lie outside it.
            "at_grid_edge": i in (0, len(COARSE_GRID) - 1),
            "curve": list(zip(COARSE_GRID.tolist(), coarse)), "n": len(scores)}


def calibrate(dataset, kg, gnn, split=None, variants=PRIOR_VARIANTS, write=True):
    """Fit T_predict once and T_propose per prior variant; write one file per variant.

    Returns {"predict": fit, "variants": {variant: {"fit": fit, "path": str}}, "split": split}.
    """
    split = split or CALIBRATION_SPLITS[dataset]
    space = P.SlotSpace(kg)
    drop_edges = kg.drop_edges
    sampler = sampler_signature(gnn.args)

    any_ri = RelationInferenceInterface(kg, model_path=prior_model_path(dataset, "finetune"))
    base_questions = P.build_questions(dataset, split, kg, any_ri, space)
    pred_S, pred_G, pred_counts = predict_data(base_questions, drop_edges, gnn)
    predict_fit = {**fit_temperature(pred_S, pred_G), **pred_counts, "questions": len(base_questions)}
    print(f"[calibration] {dataset} predict: T={predict_fit['T']:.4f} nll={predict_fit['nll']:.4f} "
          f"({pred_counts})")

    result = {"split": split, "sampler": sampler, "predict": predict_fit, "variants": {}}
    for variant in variants:
        # noprior loads the finetune weights too (prior_model_path), and lam = k_p = 0 ignores them.
        questions = base_questions
        if prior_model_path(dataset, variant) != prior_model_path(dataset, "finetune"):
            ri = RelationInferenceInterface(kg, model_path=prior_model_path(dataset, variant))
            questions = P.build_questions(dataset, split, kg, ri, space)
        prop_S, prop_G, prop_counts = propose_data(questions, drop_edges, space, baseline_cfg(variant))
        propose_fit = {**fit_temperature(prop_S, prop_G), **prop_counts, "questions": len(questions)}
        print(f"[calibration] {dataset} {variant} propose: T={propose_fit['T']:.4f} "
              f"nll={propose_fit['nll']:.4f} ({prop_counts})")
        path = calibration_path(dataset, variant, gnn.args)
        record = {"T_propose": propose_fit["T"], "best_nll_propose": propose_fit["nll"],
                  "T_predict": predict_fit["T"], "best_nll_predict": predict_fit["nll"],
                  **expected_calibration(dataset, variant, gnn.args),
                  "fit": {"split": split, "propose": propose_fit, "predict": predict_fit,
                          "gnn_checkpoint": getattr(gnn.args, "checkpoint", None),
                          "fitted_at": datetime.datetime.now().isoformat(timespec="seconds")}}
        if write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(record))
            print(f"[calibration] wrote {path}")
        result["variants"][variant] = {"fit": propose_fit, "path": str(path)}
    return result


def main():
    from gnn_config import one_shot_subgraph_config
    from GoG.gnn_interface import OneShotInterface
    from GoG.kg_interface import KGInterface

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(CALIBRATION_SPLITS))
    parser.add_argument("--split", help="question split under pipeline.DATASET_DIRS; default per dataset")
    parser.add_argument("--variants", nargs="+", default=list(PRIOR_VARIANTS), choices=PRIOR_VARIANTS)
    parser.add_argument("--local_ppr", action=argparse.BooleanOptionalAction, default=None,
                        help="unset keeps gnn_config.py")
    parser.add_argument("--dry_run", action="store_true", help="fit and print, write nothing")
    args = parser.parse_args()

    kg = KGInterface(args.dataset, args={"build_nx_graphs": False})
    gnn = OneShotInterface(args.dataset, kg.n_ent, kg.n_rel, local_ppr=args.local_ppr)
    gnn.assign_graph(kg)
    print("sampler:", sampler_signature(gnn.args), "| config:",
          {k: one_shot_subgraph_config[args.dataset]["args"].get(k) for k in ("checkpoint",)})
    calibrate(args.dataset, kg, gnn, split=args.split, variants=args.variants, write=not args.dry_run)


if __name__ == "__main__":
    main()
