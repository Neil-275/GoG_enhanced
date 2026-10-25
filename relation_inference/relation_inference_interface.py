from pathlib import Path

import torch

from GoG.kg_interface import KGInterface
from relation_inference.relation_inference_model import RelationInferenceModel, make_args

MODEL_DIR = Path(__file__).resolve().parent / "model"


class RelationInferenceInterface:
    """Predicts which relations an entity is likely missing, from its local relation counts.

    Output index layout (same as training): 0..n_rel-1 = outgoing (entity is head),
    n_rel..2*n_rel-1 = incoming (entity is tail).
    """

    def __init__(self, kg: KGInterface, model_path=None, args=None, kg_type: str = "incomplete"):
        self.kg = kg
        self.kg_type = kg_type
        self.id2rel = {i: rel for rel, i in kg.rel2id.items()}
        self.args = args or make_args(kg.dataset_name, 2 * kg.n_rel)
        model_path = Path(model_path) if model_path else MODEL_DIR / f"{kg.dataset_name}.pt"

        self.model = RelationInferenceModel(self.args)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.to(self.args.device)
        self.model.eval()

    def create_input_feat(self, entity: str) -> tuple:
        """Count vector of the entity's outgoing and incoming relations in the KG."""
        input_feat = [0] * self.kg.n_rel * 2
        neighbors = self.kg.get_1hop_triples(entity, kg_type=self.kg_type)
        outgoing = neighbors[neighbors["head"] == entity]["relation"].value_counts().to_dict()
        incoming = neighbors[neighbors["tail"] == entity]["relation"].value_counts().to_dict()
        for rel, count in outgoing.items():
            input_feat[self.kg.rel2id[rel]] = count
        for rel, count in incoming.items():
            input_feat[self.kg.rel2id[rel] + self.kg.n_rel] = count
        return tuple(input_feat)

    @torch.no_grad()
    def query(self, entity: str, k: int = 3) -> list[dict]:
        """Top-k predicted missing relations for `entity`, best first.

        Each item: {"relation": name, "direction": "outgoing" | "incoming", "score": probability}.
        "outgoing" means (entity, relation, ?); "incoming" means (?, relation, entity).
        """
        x = torch.tensor([self.create_input_feat(entity)], dtype=torch.float32, device=self.args.device)
        probs = self.model(x).softmax(dim=1)[0]
        top = probs.topk(min(k, probs.numel()))
        results = []
        for score, idx in zip(top.values.tolist(), top.indices.tolist()):
            outgoing = idx < self.kg.n_rel
            results.append({
                "relation": self.id2rel[idx if outgoing else idx - self.kg.n_rel],
                "direction": "outgoing" if outgoing else "incoming",
                "score": score,
            })
        return results
