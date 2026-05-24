nbfnet_config: dict = {
    "family":{
        "config": "NBFNet_PyG/config/transductive/local_family.yaml",
        "checkpoint": "data_for_LP/id_processed/family/saveModel/topk_0.1_layer_8_ValMRR_0.462.pt",
    },
    "fb15k_237": {
        "config": "NBFNet_PyG/config/transductive/fb15k237_custom.yaml",
        "checkpoint": "data_for_LP/id_processed/fb15k_237/saveModel/topk_0.1_layer_8_ValMRR_0.423.pt"
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
            "n_layer": 8,
            "hidden_dim": 64,
            "attn_dim": 4,
            "dropout": 0.0,
            "act": "relu",
            "initializer": "binary",
            "concatHidden": True,
            "shortcut": False,
            "readout": "linear",
            "cache_dir": "data_for_LP/id_processed/family/",
            "add_inverse_edges": True,
            "add_idd_edges": True,
            "checkpoint": "data_for_LP/id_processed/family/saveModel/topk_0.1_layer_8_ValMRR_0.462.pt",
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
            "cache_dir": "data_for_LP/id_processed/fb15k_237/",
            "add_inverse_edges": True,
            "add_idd_edges": True,
            "checkpoint": "data_for_LP/id_processed/fb15k_237/saveModel/topk_0.1_layer_8_ValMRR_0.423.pt"
        },
    },
    "wikidata5m": {
    },
}