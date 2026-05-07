from NBFNet_PyG.nbfnet.inference import *
from NBFNet_PyG.nbfnet import util as gnn_util
import os

gnn_model_config: dict = {
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


class GNNInterface:
    def __init__(self, dataset_name: str):
        
        assert dataset_name in gnn_model_config, f"Dataset {dataset_name} not found in config"
        cfg_file = gnn_model_config[dataset_name]["config"]
        
        # Build context for Jinja2 template rendering in config file
        vars_in_cfg = gnn_util.detect_variables(cfg_file)
        context = {}
        if vars_in_cfg:
            # Provide default values for template variables
            if "data_root" in vars_in_cfg:
                # Point to the processed data directory for link prediction
                context["data_root"] = f"data_for_LP/id_processed/{dataset_name}"
            if "gpus" in vars_in_cfg:
                context["gpus"] = [0]  # Default to first GPU
            if "checkpoint" in vars_in_cfg:
                context["checkpoint"] = gnn_model_config[dataset_name].get("checkpoint")
                
        
        cfg = gnn_util.load_config(cfg_file, context=context)
        dataset_class = cfg.dataset.get("class")
        is_inductive = bool(dataset_class) and str(dataset_class).startswith("Ind") 

        self.dataset = gnn_util.build_dataset(cfg)
        cfg.model.num_relation = self.dataset.num_relations
        self.model = gnn_util.build_model(cfg)
        self.device = gnn_util.get_device(cfg)
        self.model = self.model.to(self.device)
    
    def assign_graph(self, graph):
        self.graph = graph.to(self.device)
        
    def predict_topk(self, entity: int, relation: int,
                      k: int = 10, predict: str = "tail",
                        filter_known: bool = True, filter_graph=None, promote_known: bool = False):   
        # relation_id = self.dataset.rel2id[relation]
        print(f"Predicting top-{k} for entity {entity} and relation {relation}...")
        return predict_topk(
            self.model,
            graph=self.graph,
            entity=entity,
            relation=relation,
            k=k,
            predict=predict,
            filter_known=filter_known,
            filter_graph=filter_graph,
            promote_known=promote_known,
        )