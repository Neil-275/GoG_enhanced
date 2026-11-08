"""Pack the gitignored files notebooks/run_eval_bundle.py reads into one tar, with a manifest.

The list is built from the paths the code reads, not from whole folders:
    gnn_config.data_config_map                KG files (incomplete, complete, relations, labels, entities)
    gnn_config.one_shot_subgraph_config       GNN checkpoint per dataset
    relation_inference.baseline.calibration   calibration question splits (pipeline.DATASET_DIRS)
    notebooks.eval_pipeline.DEFAULT_QUESTIONS report question files
    GoG.prob_propose.prior_model_path         relation prior weights, finetune and pretrain
    results/calibration_results/*_matrix.pkl  old matrix calibration, kept for reference
Nothing under data_for_LP/id_processed/ is read: the local-push sampler writes its PPR cache there lazily.
sampled_args/*.pkl are tracked by git, so the clone has them.

Writes vast/data_manifest.txt (committed; run.sh checks it) and, unless --manifest_only, the tar with the
manifest inside. Run from the repo root:
    python -m vast.pack_data [--out PATH] [--manifest_only]
"""
import argparse
import glob
import hashlib
import os
import tarfile
from pathlib import Path

DATASETS = ("family", "fb15k_237", "wikidata5m")
MANIFEST = Path("vast/data_manifest.txt")


def required_files():
    from gnn_config import data_config_map, one_shot_subgraph_config
    from GoG.prob_propose import CALIBRATION_DIR, PRIOR_VARIANTS, prior_model_path
    from notebooks.eval_pipeline import DEFAULT_QUESTIONS
    from relation_inference.baseline import pipeline as P
    from relation_inference.baseline.calibration import CALIBRATION_SPLITS

    files = []
    for dataset in DATASETS:
        files += [path for key, path in data_config_map[dataset].items() if key.endswith("_path")]
        files.append(one_shot_subgraph_config[dataset]["args"]["checkpoint"])
        files.append(f"{P.DATASET_DIRS[dataset]}/{CALIBRATION_SPLITS[dataset]}.tsv")
        files.append(DEFAULT_QUESTIONS[dataset])
        files += [prior_model_path(dataset, variant) for variant in PRIOR_VARIANTS]
    files += sorted(glob.glob(str(CALIBRATION_DIR / "*_matrix.pkl")))
    return sorted({Path(f).as_posix() for f in files})


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="gog_eval_data.tar.gz", help="tar path (outside the repo is fine)")
    parser.add_argument("--manifest_only", action="store_true")
    args = parser.parse_args()

    files = required_files()
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        raise SystemExit("missing locally:\n  " + "\n  ".join(missing))

    lines = ["# size_bytes  sha256  path  (written by vast/pack_data.py; run.sh checks each path and size)"]
    for f in files:
        lines.append(f"{os.path.getsize(f)}  {sha256(f)}  {f}")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = sum(os.path.getsize(f) for f in files)
    print(f"{len(files)} files, {total / 2**30:.2f} GiB uncompressed -> {MANIFEST}")
    if args.manifest_only:
        return

    # gzip: the KG tsv files are most of the size and compress about 4x, which shortens the upload.
    with tarfile.open(args.out, "w:gz", compresslevel=6) as tar:
        for f in files + [MANIFEST.as_posix()]:
            tar.add(f, arcname=f)
            print("  +", f)
    print(f"tar: {args.out} ({os.path.getsize(args.out) / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
