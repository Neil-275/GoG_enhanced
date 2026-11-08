"""Run the Propose evaluation of docs/eval_plan_wikidata5m.md on family, fb15k_237 and wikidata5m in one process.

Per dataset, in order, with the local-push sampler and the gnn_config.py parameters:
    1. calibration    T_predict, and T_propose for the finetune, pretrain and noprior priors (written to
                      results/calibration_results/*_localpush.pkl)
    2. relation prior top-3 slot hit, pretrain and finetune
    3. predict        GNN given the gold slot (MRR, Hits@k) with subgraph coverage
    4. prob_propose   LLM-free Propose then Predict, finetune prior
    5. ablation       step 4 for pretrain and noprior
KG and GNN are loaded once per dataset. A failing step writes its traceback to the note and the run goes on.
The note results/eval_bundle/NOTE_<timestamp>.md is appended after every step, so a crash keeps what finished.

Run from the repo root:
    python -m notebooks.run_eval_bundle [--datasets family fb15k_237 wikidata5m]
"""
import argparse
import datetime
import gc
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch

from GoG.gnn_interface import OneShotInterface
from GoG.kg_interface import KGInterface
from GoG.prob_propose import PRIOR_VARIANTS, ProbabilisticPropose, prior_model_path, sampler_signature
from notebooks import eval_pipeline as E
from relation_inference.baseline import pipeline as P
from relation_inference.baseline.calibration import CALIBRATION_SPLITS, calibrate

DATASETS = ["family", "fb15k_237", "wikidata5m"]  # small first, so code errors show before the 5-min load
NOTE_DIR = Path("results/eval_bundle")

# Matrix-sampler numbers on val, from the summary cell of notebooks/test_generate.ipynb (not rerun).
MATRIX_REFERENCE = {
    "family": {
        "valid": 179, "median_nodes": 301, "median_edges": 5227, "coverage": "1.0000",
        "mrr": "0.6049", "hit@1": "73/179", "hit@3": "135/179", "hit@5": "163/179", "hit@10": "177/179",
        "propose": "159/179", "predict": "111/179", "avg_slots": "2.8715", "avg_predicted": "2.4916",
        "normalized_precision": "0.4320",
    },
    "fb15k_237": {
        "valid": 506, "median_nodes": 1454, "median_edges": 37928, "coverage": "0.9862",
        "mrr": "0.6195", "hit@1": "258/506", "hit@3": "350/506", "hit@5": "396/506", "hit@10": "434/506",
        "propose": "465/506", "predict": "289/506", "avg_slots": "2.3419", "avg_predicted": "1.7984",
        "normalized_precision": "0.4605",
        "prior_pretrain": "0.3933", "prior_finetune": "0.6976",
    },
}
# The checkpoint stores no training args, so the sampler values are the train_auto.py defaults.
WD5M_CHECKPOINT_VAL_MRR = 0.799


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.rstrip()
    except OSError as error:
        return f"({error})"


def ram_gb():
    try:
        import psutil
        return f"{psutil.virtual_memory().total / 2**30:.1f}"
    except ImportError:
        return "?"


def fmt_ratio(r):
    return f"{r['hit']}/{r['total']} = {r['rate']:.4f}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


class Note:
    """Markdown note, appended and closed after every write so a crash keeps what finished."""

    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, *blocks):
        with open(self.path, "a", encoding="utf-8") as note:
            for block in blocks:
                note.write(block.rstrip("\n") + "\n\n")
        print("\n".join(blocks))


def header(note, datasets):
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "không có"
    note.write(
        f"# Eval bundle {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        "## Lần chạy\n\n"
        f"- Lệnh: `{' '.join([Path(sys.executable).name] + sys.argv)}`\n"
        f"- Dataset: {', '.join(datasets)}\n"
        f"- Git commit: `{run(['git', 'rev-parse', 'HEAD'])}` (branch `{run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])}`)\n"
        f"- Python {platform.python_version()}, torch {torch.__version__}, CUDA {torch.version.cuda}, GPU {gpu}, "
        f"RAM {ram_gb()} GB, host `{platform.node()}`",
        "`git status --short`:\n\n```\n" + (run(["git", "status", "--short"]) or "(sạch)") + "\n```",
        "## Ghi chú chung\n\n"
        "- Mỗi phép đo chỉ tính câu có đúng một cạnh bị drop nối `q_entity` -> `hard_answer`.\n"
        "- Sampler local push cho cả ba dataset, tham số giữ nguyên `gnn_config.py`, không quét.\n"
        "- Family/FB15K-237 fit calibration và báo cáo trên cùng tập `val`; calibration chỉ fit 2 nhiệt độ nên "
        "rò rỉ nhỏ. Wikidata5m fit trên `val_calibration` (val trừ smoke), báo cáo trên `smoke_val`.\n"
        "- Cấu hình sampler wikidata5m (`epsilon 1e-6`, 1024 node, 10000 cạnh) là mặc định của "
        "`one_shot_subgraph/train_auto.py`; checkpoint không lưu tham số train nên chưa xác nhận là cấu hình lúc "
        f"train. Mốc so sánh: ValMRR {WD5M_CHECKPOINT_VAL_MRR} trong tên checkpoint gốc.\n"
        "- Số relation prior của wikidata5m bị lạc quan: `smoke_val` nằm trong val, mà val là tập early-stopping "
        "khi finetune prior.\n"
        "- `Normalized precision` giữ định nghĩa của `test_generate.ipynb`: 1 / số entity dự đoán *của slot chứa* "
        "hard answer khi trúng, trung bình trên mọi câu. Cột `(mọi slot)` chia cho tổng entity dự đoán của mọi "
        "slot được chọn, đúng như mô tả trong kế hoạch.\n"
        "- Biến thể `noprior`: `lam = 0`, `k_p = 0` (prior không cộng điểm, không thêm slot).",
    )


def dataset_setup(note, dataset):
    args = E.sampler_config(dataset, {"local_ppr": True})
    calib_split = CALIBRATION_SPLITS[dataset]
    priors = "\n".join(f"  - {v}: `{prior_model_path(dataset, v)}`" + (" (không dùng: lam = k_p = 0)"
                                                                     if v == "noprior" else "")
                       for v in PRIOR_VARIANTS)
    note.write(
        f"## {dataset}",
        "### Cấu hình\n\n"
        f"- Sampler: `{sampler_signature(args)}`\n"
        f"- GNN checkpoint: `{args['checkpoint']}`, cache PPR: `{args['cache_dir']}`\n"
        f"- Câu hỏi fit calibration: `{P.DATASET_DIRS[dataset]}/{calib_split}.tsv`\n"
        f"- Câu hỏi báo cáo: `{E.DEFAULT_QUESTIONS[dataset]}`\n"
        f"- Prior:\n{priors}",
    )
    return args


def step(note, title, fn):
    """Run one step; on failure write the traceback and return None."""
    print(f"\n===== {title} =====")
    started = time.time()
    try:
        out = fn()
    except Exception:  # noqa: BLE001 - the bundle goes on
        note.write(f"### {title}\n\n**LỖI** sau {time.time() - started:.0f}s:\n\n```\n{traceback.format_exc()}```")
        return None
    return out, time.time() - started


def write_calibration(note, result, seconds):
    rows = []
    fits = [("predict (dùng chung)", result["predict"], result["predict"]["gold_in_pool"])]
    fits += [(f"propose {v}", out["fit"], out["fit"]["gold_in_C"]) for v, out in result["variants"].items()]
    for name, fit, used in fits:
        edge = " **(cực tiểu ở biên lưới)**" if fit["at_grid_edge"] else ""
        rows.append([name, f"{fit['T']:.4f}{edge}", f"{fit['nll']:.4f}", f"{fit['coarse_T']:.4f}",
                     f"{fit['coarse_nll']:.4f}", f"{used}/{fit['valid']}"])
    curves = [result["predict"]["curve"]] + [out["fit"]["curve"] for out in result["variants"].values()]
    curve_rows = [[f"{curves[0][i][0]:.4f}"] + [f"{c[i][1]:.4f}" for c in curves] for i in range(len(curves[0]))]
    note.write(
        f"### 1. Calibration ({seconds:.0f}s, split `{result['split']}`)",
        table(["Nhiệt độ", "T (lưới mịn)", "NLL", "T (lưới thô)", "NLL thô", "câu dùng để fit / hợp lệ"], rows),
        "File: " + ", ".join(f"`{out['path']}`" for out in result["variants"].values()),
        "<details><summary>Đường NLL theo T (lưới thô 50 điểm)</summary>\n\n"
        + table(["T", "predict"] + [f"propose {v}" for v in result["variants"]], curve_rows) + "\n\n</details>",
    )


def write_prior(note, result, seconds, dataset):
    rows = [[stage, fmt_ratio(result[stage]), f"`{result[stage]['model']}`", len(result[stage]["errors"])]
            for stage in ("pretrain", "finetune")]
    blocks = [f"### 2. Relation prior, hit trong top-3 ({seconds:.0f}s)",
              f"{result['questions']} câu, {result['q_entities']} q entity, {result['valid']} câu hợp lệ.",
              table(["Prior", "Top-3 hit", "Model", "Lỗi"], rows)]
    ref = MATRIX_REFERENCE.get(dataset, {})
    if "prior_finetune" in ref:
        blocks.append(f"Số cũ trong `test_generate.ipynb`: pretrain {ref['prior_pretrain']}, "
                      f"finetune {ref['prior_finetune']}.")
    errors = [e for stage in ("pretrain", "finetune") for e in result[stage]["errors"]]
    if errors:
        blocks.append("Lỗi (tính là trượt):\n\n```\n" + "\n".join(errors[:20]) + "\n```")
    note.write(*blocks)


def write_predict(note, result, seconds, dataset):
    ref = MATRIX_REFERENCE.get(dataset)
    metrics = [("Câu hợp lệ", result["valid"], "valid"),
               ("Hard answer nằm trong subgraph", fmt_ratio(result["coverage"]), "coverage"),
               ("Node trong subgraph (trung vị)", result["median_nodes"], "median_nodes"),
               ("Cạnh trong subgraph (trung vị)", result["median_edges"], "median_edges"),
               ("MRR", f"{result['mrr']:.4f}", "mrr")]
    metrics += [(f"Hit@{k}", fmt_ratio(result[f"hit@{k}"]), f"hit@{k}") for k in (1, 3, 5, 10)]
    header_row = ["Chỉ số", "Local push"] + (["Matrix (số cũ)"] if ref else [])
    rows = [[name, value] + ([ref.get(key, "")] if ref else []) for name, value, key in metrics]
    blocks = [f"### 3. GNN Predict khi biết slot đúng, kèm độ phủ subgraph ({seconds:.0f}s)",
              table(header_row, rows)]
    if dataset == "wikidata5m":
        blocks.append(f"MRR {result['mrr']:.4f} so với ValMRR {WD5M_CHECKPOINT_VAL_MRR} của checkpoint. Nếu thấp "
                      "hơn nhiều, nghi ngờ cấu hình sampler lúc train khác cấu hình đang dùng.")
    note.write(*blocks)


PROB_METRICS = [("Câu hợp lệ", lambda r: r["valid"], "valid"),
                ("Chọn đúng slot", lambda r: fmt_ratio(r["propose"]), "propose"),
                ("Hard answer trong entity dự đoán", lambda r: fmt_ratio(r["predict"]), "predict"),
                ("Số slot TB", lambda r: f"{r['avg_slots']:.4f}", "avg_slots"),
                ("Số entity dự đoán TB", lambda r: f"{r['avg_predicted']:.4f}", "avg_predicted"),
                ("Normalized precision", lambda r: f"{r['normalized_precision']:.4f}", "normalized_precision"),
                ("Normalized precision (mọi slot)", lambda r: f"{r['normalized_precision_all']:.4f}", None),
                ("Lỗi", lambda r: len(r["errors"]), None)]


def write_prob(note, title, results, seconds, dataset, with_reference):
    ref = MATRIX_REFERENCE.get(dataset) if with_reference else None
    variants = list(results)
    header_row = ["Chỉ số"] + variants + (["Matrix, finetune (số cũ)"] if ref else [])
    rows = [[name] + [fn(results[v]) for v in variants] + ([ref.get(key, "") if key else ""] if ref else [])
            for name, fn, key in PROB_METRICS]
    blocks = [f"### {title} ({seconds:.0f}s)", table(header_row, rows)]
    errors = [e for v in variants for e in results[v]["errors"]]
    if errors:
        blocks.append("Lỗi:\n\n```\n" + "\n".join(errors[:20]) + "\n```")
    note.write(*blocks)


def run_dataset(note, dataset):
    args = dataset_setup(note, dataset)
    missing = E.missing_for_kg_env(dataset, need_calibration=False)
    missing += [f"relation prior {prior_model_path(dataset, v)}" for v in PRIOR_VARIANTS
                if not Path(prior_model_path(dataset, v)).exists()]
    if missing:
        note.write("**Bỏ qua dataset**, thiếu:\n\n" + "\n".join(f"- {m}" for m in sorted(set(missing))))
        return

    loaded = step(note, "Load KG và GNN", lambda: _load(dataset))
    if loaded is None:
        return
    (kg, gnn), seconds = loaded
    note.write(f"KG và GNN load trong {seconds:.0f}s: {kg.n_ent} entity, {kg.n_rel} relation, "
               f"{len(kg.incomplete_kg)} cạnh trong G_inc, {len(kg.drop_edges)} cạnh bị drop.")

    out = step(note, "1. Calibration", lambda: calibrate(dataset, kg, gnn))
    if out:
        write_calibration(note, *out)

    report_path = E.DEFAULT_QUESTIONS[dataset]
    questions = pd.read_csv(report_path, sep="\t")
    out = step(note, "2. Relation prior", lambda: E.evaluate_relation_inference(dataset, questions))
    if out:
        write_prior(note, *out, dataset)

    env_out = step(note, "Khởi tạo KGEnv (prior finetune)",
                   lambda: E.load_kg_env(dataset, report_path, "finetune", kg=kg, gnn=gnn))
    if env_out is None:
        return
    kg_env = env_out[0]
    drop_edges = kg.drop_edges

    out = step(note, "3. GNN Predict", lambda: E.evaluate_predict(kg_env, questions, drop_edges))
    if out:
        write_predict(note, *out, dataset)

    prob = {}
    out = step(note, "4. Propose xác suất", lambda: E.evaluate_prob_propose(kg_env, questions, drop_edges))
    if out:
        prob["finetune"] = out[0]
        write_prob(note, "4. Propose xác suất rồi Predict, prior finetune", {"finetune": out[0]}, out[1],
                   dataset, with_reference=True)

    agent_args = kg_env.args
    started = time.time()
    for variant in ("pretrain", "noprior"):
        def evaluate(variant=variant):
            kg_env.prob_propose = ProbabilisticPropose(
                kg, gnn, dataset, max_slots=agent_args.max_selected_relations,
                k_predict=agent_args.predict_topk, prior_variant=variant)
            return E.evaluate_prob_propose(kg_env, questions, drop_edges)
        out = step(note, f"5. Ablation prior: {variant}", evaluate)
        if out:
            prob[variant] = out[0]
    if prob:
        write_prob(note, "5. Ablation relation prior (Propose xác suất rồi Predict)", prob,
                   time.time() - started, dataset, with_reference=False)


def _load(dataset):
    kg = KGInterface(dataset, args={"build_nx_graphs": False})
    gnn = OneShotInterface(dataset, kg.n_ent, kg.n_rel, local_ppr=True)
    gnn.assign_graph(kg)
    return kg, gnn


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    args = parser.parse_args()
    datasets = [d for d in DATASETS if d in args.datasets]

    note = Note(NOTE_DIR / f"NOTE_{datetime.datetime.now():%Y%m%d_%H%M%S}.md")
    header(note, datasets)
    for dataset in datasets:
        started = time.time()
        try:
            run_dataset(note, dataset)
        except Exception:  # noqa: BLE001 - next dataset
            note.write(f"**LỖI ngoài các bước** ở {dataset}:\n\n```\n{traceback.format_exc()}```")
        note.write(f"_{dataset}: {time.time() - started:.0f}s_")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    note.write("_Xong._")
    print(f"\nNote: {note.path}")


if __name__ == "__main__":
    main()
