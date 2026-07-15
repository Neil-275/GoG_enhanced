"""Demo: inference from in-memory entity/relation/edge lists.

This demo does NOT read entities/relations/edges from files.
You provide them as Python lists, then run top-k prediction.

Usage (example):

    source C:/Users/MTBH/anaconda3/etc/profile.d/conda.sh && conda activate hybkgqa
    python demo_inference_inmemory.py

Notes:
- You must set WEIGHT_PATH to a checkpoint that matches MODEL_KWARGS.
- First run may generate PPR caches under CACHE_DIR.
"""

import types

from inference_engine import InferenceEngine


def main():
    # ----- User-provided KG (example toy graph) -----
    entities = ["alice", "bob", "carol", "dave"]
    relations = ["likes", "knows"]

    # edges: (h, r, t) using names (can also be ints)
    edges = [
        ("alice", "likes", "bob"),
        ("bob", "likes", "carol"),
        ("carol", "knows", "dave"),
        ("dave", "knows", "alice"),
    ]

    # ----- Model / sampler config -----
    WEIGHT_PATH = "CHANGE_ME.pt"  # e.g. "data/FB15k_237/saveModel/topk_0.1_layer_8_ValMRR_0.296.pt"
    CACHE_DIR = "./infer_cache/"

    # This args object mimics the training args; values MUST match your checkpoint.
    args = types.SimpleNamespace(
        gpu=0,
        # sampler behavior
        topk=1.0,   # 1.0 means sample all nodes in this tiny toy graph
        topm=-1,
        add_manual_edges=False,
        use_gpu_ppr=True,
        # model hyperparams
        n_layer=6,
        hidden_dim=64,
        attn_dim=32,
        dropout=0.0,
        act="relu",
        initializer="binary",
        concatHidden=False,
        shortcut=False,
        readout="linear",
    )

    engine = InferenceEngine(
        args=args,
        entities=entities,
        relations=relations,
        edges=edges,
        weight_path=WEIGHT_PATH,
        cache_dir=CACHE_DIR,
        add_inverse_edges=True,
        add_idd_edges=False,
    )

    scores, tail_ids = engine.predict_topk(head="alice", relation="likes", topk=3)
    print("Top-3 tail ids:", tail_ids.tolist())
    print("Top-3 scores:", scores.tolist())


if __name__ == "__main__":
    main()
