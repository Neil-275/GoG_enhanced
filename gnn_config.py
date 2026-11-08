import torch


class FB15k_237_Config:
    """Configuration for FB15k-237 dataset"""



class FamilyConfig:
    """Configuration for Family dataset"""
    incomplete_path = "brink_dataset/family/knowledge_graph_incomplete.tsv"
    complete_path = "brink_dataset/family/knowledge_graph_complete.tsv"
    rel_path = "brink_dataset/family/relations.txt"
    ent_path = "brink_dataset/family/entities.txt"


class Wikidata5mConfig:
    """Configuration for Wikidata5m dataset"""
    incomplete_path = "brink_dataset/wikidata5m/incomplete_facts.tsv"
    complete_path = "brink_dataset/wikidata5m/facts.tsv"
    rel_path = "brink_dataset/wikidata5m/relations.txt"
    # Line-aligned with relations.txt: the first alias of each relation id (relations_alias.txt)
    rel_label_path = "brink_dataset/wikidata5m/relation_labels.txt"
    ent_path = "brink_dataset/wikidata5m/entities.txt"


data_config_map = {
    "fb15k_237": {
            "incomplete_path": "brink_dataset/fb15k_237/private_id/knowledge_graph_incomplete.tsv",
            "complete_path": "brink_dataset/fb15k_237/private_id/knowledge_graph_complete.tsv",
            "rel_path": "brink_dataset/fb15k_237/private_id/relations.txt",  # For translation from BRINK ID to GNN ID
            "ent_path": "brink_dataset/fb15k_237/private_id/entities.txt"
        },
    "family":{
        "incomplete_path": "brink_dataset/family/knowledge_graph_incomplete.tsv",
        "complete_path": "brink_dataset/family/knowledge_graph_complete.tsv",
        "rel_path": "brink_dataset/family/relations.txt",
        "ent_path": "brink_dataset/family/entities.txt"
    },
    "wikidata5m": {
        "incomplete_path": "brink_dataset/wikidata5m/incomplete_facts.tsv",
        "complete_path": "brink_dataset/wikidata5m/facts.tsv",
        "rel_path": "brink_dataset/wikidata5m/relations.txt",
        "rel_label_path": "brink_dataset/wikidata5m/relation_labels.txt",
        "ent_path": "brink_dataset/wikidata5m/entities.txt"
    }
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