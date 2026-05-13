nbfnet_config: dict = {
    "family":{
        "config": "NBFNet_PyG/config/transductive/local_family.yaml",
        "checkpoint": "NBFNet_weight/family_2026-05-04-01-57-16/model_epoch_10.pth",
    },
    "fb15k_237": {
        "config": "NBFNet_PyG/config/transductive/fb15k237_custom.yaml",
        # "checkpoint": "NBFNet_PyG/config/transductive/fb15k_237_custom.yaml",
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
            "n_layer": 6,
            "hidden_dim": 64,
            "attn_dim": 32,
            "dropout": 0.0,
            "act": "relu",
            "initializer": "binary",
            "concatHidden": False,
            "shortcut": False,
            "readout": "linear",
            "cache_dir": "data_for_LP/id_processed/family/",
            "add_inverse_edges": True,
            "add_idd_edges": True,
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
            "attn_dim": 32,
            "dropout": 0.0,
            "act": "relu",
            "initializer": "binary",
            "concatHidden": False,
            "shortcut": False,
            "readout": "linear",
            "cache_dir": "data_for_LP/id_processed/fb15k_237/",
            "add_inverse_edges": True,
            "add_idd_edges": True,
        },
    },
    "wikidata5m": {
    },
}