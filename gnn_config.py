import torch

nbfnet_config: dict = {
    "family":{
        "config": "NBFNet_PyG/config/transductive/local_family.yaml",
        "checkpoint": "data_for_LP/id_processed/family/saveModel/topk_0.1_layer_8_ValMRR_0.462.pt",
    },
    "fb15k_237": {
        "config": "NBFNet_PyG/config/transductive/fb15k237_custom.yaml",
        "checkpoint": "data_for_LP/id_processed/fb15k_237/finetuneModel/epoch_3_ValMRR_0.568.pt"
    },
    "wikidata5m": {
        "config": "config/wikidata5m.yaml",
        "checkpoint": "checkpoints/wikidata5m/best.pt", 
    },
}

one_shot_subgraph_config: dict = {
    "family":{
        "args": {
            "device": "cuda:0",
            "gpu": "0",
            "topk": 0.1,
            "topm": -1,
            # "n_ent": 
            "add_manual_edges": False,
            "use_gpu_ppr": True,
            "n_layer": 5,
            "hidden_dim": 64,
            "attn_dim": 4,
            "dropout": 0.0,
            "act": "relu",
            "initializer": "binary",
            "concatHidden": True,
            "shortcut": False,
            "readout": "linear",
            "scoring_mode": "global",
            # Off keeps the reported matrix-sampler numbers; --local_ppr turns on these values,
            # chosen on val so that hard answer coverage and MRR match the matrix sampler.
            "local_ppr": False,
            "max_nodes_per_query": 1024,
            "max_edges_per_query": 10000,
            "local_ppr_alpha": 0.85,
            "local_ppr_epsilon": 1e-7,
            "cache_dir": "data_for_LP/id_processed/family/",
            "add_inverse_edges": True,
            "add_idd_edges": True,
            # "checkpoint": "data_for_LP/id_processed/family/saveModel/topk_0.1_layer_8_ValMRR_0.462.pt",
            # "checkpoint": "data_for_LP/id_processed/family/finetuneModel/epoch_3_ValMRR_0.746.pt",
            "checkpoint": "data_for_LP/weights/family/brink_only_layer_5.pt"
        },
    },
    "fb15k_237": {
        "args": {
            "device": "cuda:0",
            "gpu": "0",
            "topk": 0.1,
            "topm": -1,
            "add_manual_edges": False,
            "use_gpu_ppr": True,
            "n_layer": 8,
            "hidden_dim": 64,
            "attn_dim": 4,
            "dropout": 0.2,
            "act": "relu",
            "initializer": "binary",
            "concatHidden": True,
            "shortcut": False,
            "readout": "linear",
            "scoring_mode": "global",
            # Off keeps the reported matrix-sampler numbers; --local_ppr turns on these values,
            # chosen on val so that hard answer coverage and MRR match the matrix sampler.
            "local_ppr": False,
            "max_nodes_per_query": 1454,
            "max_edges_per_query": 30000,
            "local_ppr_alpha": 0.85,
            "local_ppr_epsilon": 1e-8,
            "cache_dir": "data_for_LP/id_processed/fb15k_237/",
            "add_inverse_edges": True,
            "add_idd_edges": True,
            # "checkpoint": "data_for_LP/id_processed/fb15k_237/saveModel/topk_0.1_layer_8_ValMRR_0.423.pt"
            # "checkpoint": "data_for_LP/id_processed/fb15k_237/finetuneModel/epoch_3_ValMRR_0.568.pt"
            "checkpoint": "data_for_LP/weights/fb15k_237/brink_only_layer_5.pt"
        },
    },
    "wikidata5m": {
        "args": {
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "gpu": "0",
            "topk": 0.1,  # ignored in local-push mode
            "topm": -1,
            "add_manual_edges": False,
            "use_gpu_ppr": True,
            # The checkpoint stores no training args. These four are the code defaults; the real
            # values are unknown, so check predict MRR against the checkpoint's ValMRR_0.799.
            "local_ppr": True,
            "max_nodes_per_query": 1024,
            "max_edges_per_query": 10000,
            "local_ppr_alpha": 0.85,
            "local_ppr_epsilon": 1e-6,
            "n_layer": 5,
            "hidden_dim": 64,
            "attn_dim": 4,
            "dropout": 0.0,
            "act": "relu",
            "initializer": "binary",
            "concatHidden": True,
            "shortcut": False,
            "readout": "linear",
            "scoring_mode": "global",
            "cache_dir": "data_for_LP/id_processed/wikidata5m/",
            "add_inverse_edges": True,
            "add_idd_edges": True,
            "checkpoint": "data_for_LP/weights/wikidata5m/brink_only_layer_5.pt"
        },
    },
}