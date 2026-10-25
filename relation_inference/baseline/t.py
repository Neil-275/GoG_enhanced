from pathlib import Path

import numpy as np

from GoG.kg_interface import KGInterface
from relation_inference.relation_inference_interface import RelationInferenceInterface
from relation_inference.baseline import pipeline as P
from relation_inference.baseline import run_baseline as RB

DATASET = "family"
CFG = {"T": 0.02, "lam": 1, "gamma": 2, "k_b": 3, "k_p": 3, "m": 1, "K": 3}  # config full-F1 đã chọn trên val

# ---- 1. Khởi tạo: KG, không gian slot, prior, cache GNN -------------------------------------------
kg = KGInterface(DATASET)
space = P.SlotSpace(kg)
ri = RelationInferenceInterface(kg, model_path=RB.DEFAULT_PRIOR[DATASET])
cache = P.PredictCache(DATASET, kg, space, Path(f"results/baseline/{DATASET}/gnn_cache.pkl"), k_max=RB.K_MAX)


# ---- 2. Trả lời một câu hỏi --------------------------------------------------------------------
def answer(q: P.Question, cfg: dict) -> dict:
    sources = P.candidate_sources(q, space, cfg["k_b"], cfg["k_p"])              # tập C
    ranked, scores = P.rank_candidates(q, space, np.array(sorted(sources), dtype=np.int64),
                                       cfg["T"], cfg["lam"], cfg["gamma"])        # xếp hạng (r, d)
    chosen = ranked[:cfg["m"]]                                                   # top-m slot
    cache.fill([(q.e, int(s)) for s in chosen])                                  # chạy GNN nếu cache thiếu
    pred, detail = RB.predict_answer(q, chosen, cfg["K"], cache)                 # observed ∪ GNN top-K
    return {"chosen": [space.name(s) for s in chosen], "prediction": sorted(pred),
            "detail": {space.name(s): d for s, d in detail.items()}}


# ---- 2a. Câu hỏi có sẵn trong dataset ----------------------------------------------------------
test = P.build_questions(DATASET, "test", kg, ri, space)
q = test[2]
out = answer(q, CFG)
print(q.question, "| e =", q.e)
print("  chọn      :", out["chosen"], "| gold:", [space.name(s) for s in q.oracle])
print("  dự đoán   :", out["prediction"], "| đáp án:", sorted(q.answers))
print("  chi tiết  :", out["detail"])
print("  qa_row    :", dict(zip(["hits_any", "hits_hard", "P", "R", "F1"], RB.qa_row(set(out["prediction"]), q))))


# ---- 2b. Câu hỏi tự nhập (không có đáp án, không có oracle) ------------------------------------
q2 = P.make_question(kg, ri, space, "Who is 1474's son?", "1474")
out2 = answer(q2, CFG)
print("\n", q2.question)
print("  chọn    :", out2["chosen"])
print("  dự đoán :", out2["prediction"])


# ---- 3. Chấm cả split với một config ----------------------------------------------------------
res = RB.report_config(test, space, CFG, cache, Path("results/baseline/demo_test_predictions.json"))
print("\nTest (all):", {k: round(res["qa"]["all"][k], 3) for k in ("Hits@Any", "Hits@Hard", "F1", "HHR")})
print("Chọn relation:", {k: round(v, 3) for k, v in res["selection"]["all"].items()})
