"""Deterministic, LLM-free Propose-Predict baseline: the 1-hop upper reference for the agent.

For each question (q, e_topic) on G_inc:
  candidates C = bi-encoder top-k_b relations (both directions) U prior top-k_p slots U slots present at e_topic
  score(r, d)  = log softmax_C(cos(q, r) / T) + lam * log p_prior(r, d) + gamma * 1[(r, d) present at e_topic]
  answer       = U over the top-m slots of: observed far ends in G_inc  U  GNN top-K (existence-filtered)

A "slot" is an index into the prior's layout: s < n_rel is (rel s, outgoing), s >= n_rel is (rel s - n_rel,
incoming). Every per-question signal is stored in that layout so candidate sets, scores and oracles are all
plain index arrays.
"""
import pickle
from ast import literal_eval
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import util

from GoG.kg_interface import KGInterface, embed_model
from GoG.utils import shorten_relation
from relation_inference.relation_inference_interface import RelationInferenceInterface

DATASET_DIRS = {
    "family": "brink_dataset/family",
    "fb15k_237": "brink_dataset/fb15k_237/private_id",
    "wikidata5m": "brink_dataset/wikidata5m",
}
DIRECTIONS = ("outgoing", "incoming")
PRIOR_FLOOR = 1e-6
SOURCES = ("bienc", "prior", "local")


@dataclass
class Question:
    id: str
    question: str
    e: str
    answers: set
    hard: set
    oracle: list            # oracle slots: dropped edges between e and a hard answer, in G_complete - G_inc
    cos: np.ndarray         # [n_rel] cosine(question, relation label)
    log_prior: np.ndarray   # [2 n_rel] log p_prior(slot | N(e)), floored at PRIOR_FLOOR
    present: np.ndarray     # [2 n_rel] bool, slot has an edge at e in G_inc
    observed: dict = field(default_factory=dict)  # slot -> sorted far-end entities in G_inc

    @property
    def group(self) -> str:
        """'present' if any oracle slot already has an edge at e_topic in G_inc, else 'absent'."""
        return "present" if any(self.present[s] for s in self.oracle) else "absent"


class SlotSpace:
    """Maps between slots and (relation, direction), with a deterministic name order for tie-breaking."""

    def __init__(self, kg: KGInterface):
        self.relations = list(kg.relations)
        self.n_rel = len(self.relations)
        self.rel2id = kg.rel2id
        names = [(r, d) for d in DIRECTIONS for r in self.relations]
        order = sorted(range(2 * self.n_rel), key=lambda s: names[s])
        self.name_rank = np.empty(2 * self.n_rel, dtype=np.int64)
        self.name_rank[order] = np.arange(2 * self.n_rel)
        self.rel_name_rank = np.argsort(np.argsort(np.array(self.relations, dtype=object)))

    def slot(self, relation: str, direction: str) -> int:
        return self.rel2id[relation] + (self.n_rel if direction == "incoming" else 0)

    def rel_of(self, slot) -> int:
        return slot % self.n_rel

    def name(self, slot: int) -> tuple:
        return self.relations[self.rel_of(slot)], DIRECTIONS[int(slot >= self.n_rel)]


def _read_split(dataset: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(f"{DATASET_DIRS[dataset]}/{split}.tsv", sep="\t", dtype={"id": str})
    for col in ("q_entity", "answer", "hard_answer"):
        df[col] = df[col].apply(literal_eval)
    return df


def question_text_scores(kg: KGInterface, questions: list[str]) -> np.ndarray:
    """Cosine of each question against every relation label, as in KGInterface.get_best_relation_match."""
    query_embeddings = embed_model.encode([shorten_relation(q) for q in questions], convert_to_tensor=True)
    return util.cos_sim(query_embeddings, kg.rel_emb).cpu().numpy()


@torch.no_grad()
def prior_log_probs(ri: RelationInferenceInterface, entities: list[str]) -> np.ndarray:
    """log p_prior over all 2 n_rel slots for each entity, from its G_inc relation-count profile."""
    x = torch.tensor([ri.create_input_feat(e) for e in entities], dtype=torch.float32, device=ri.args.device)
    probs = ri.model(x).softmax(dim=1).cpu().numpy()
    return np.log(np.maximum(probs, PRIOR_FLOOR))


def build_questions(dataset: str, split: str, kg: KGInterface, ri: RelationInferenceInterface,
                    space: SlotSpace) -> list[Question]:
    df = _read_split(dataset, split)
    rows = df.to_dict(orient="records")
    entities = [row["q_entity"][0] for row in rows]
    # Only question entities are looked up, and grouping all of wikidata5m does not fit in memory.
    inc = kg.incomplete_kg
    outgoing = inc[inc["head"].isin(entities)]
    incoming = inc[inc["tail"].isin(entities)]
    out_index = outgoing.groupby(["head", "relation"])["tail"].apply(lambda s: sorted(set(s))).to_dict()
    in_index = incoming.groupby(["tail", "relation"])["head"].apply(lambda s: sorted(set(s))).to_dict()
    dropped_by_pair = defaultdict(set)
    for h, r, t in zip(kg.drop_edges["head"], kg.drop_edges["relation"], kg.drop_edges["tail"]):
        dropped_by_pair[(h, t)].add(space.slot(r, "outgoing"))
        dropped_by_pair[(t, h)].add(space.slot(r, "incoming"))

    cos = question_text_scores(kg, [row["question"] for row in rows])
    log_prior = prior_log_probs(ri, entities)

    questions = []
    for i, row in enumerate(rows):
        e = entities[i]
        present = np.zeros(2 * space.n_rel, dtype=bool)
        observed = {}
        for rel in space.relations:
            for direction, index in (("outgoing", out_index), ("incoming", in_index)):
                far = index.get((e, rel))
                if far:
                    s = space.slot(rel, direction)
                    present[s] = True
                    observed[s] = far
        oracle = sorted(set().union(*(dropped_by_pair.get((e, h), set()) for h in row["hard_answer"])))
        questions.append(Question(
            id=str(row["id"]), question=row["question"], e=e,
            answers=set(map(str, row["answer"])), hard=set(map(str, row["hard_answer"])),
            oracle=oracle, cos=cos[i], log_prior=log_prior[i], present=present, observed=observed,
        ))
    return questions


def make_question(kg: KGInterface, ri: RelationInferenceInterface, space: SlotSpace, text: str, e: str,
                  qid: str = "custom") -> Question:
    """A Question for free text at `e`: no answers, no oracle. Same features as build_questions."""
    inc = kg.incomplete_kg
    present, observed = np.zeros(2 * space.n_rel, dtype=bool), {}
    for direction, here, far in (("outgoing", "head", "tail"), ("incoming", "tail", "head")):
        for rel, group in inc[inc[here] == e].groupby("relation"):
            s = space.slot(rel, direction)
            present[s], observed[s] = True, sorted(set(group[far]))
    return Question(id=qid, question=text, e=e, answers=set(), hard=set(), oracle=[],
                    cos=question_text_scores(kg, [text])[0], log_prior=prior_log_probs(ri, [e])[0],
                    present=present, observed=observed)


# ----------------------------------------------------------------------------------------------------------
# Candidate set and ranking
# ----------------------------------------------------------------------------------------------------------

def candidate_sources(q: Question, space: SlotSpace, k_b: int, k_p: int) -> dict[int, tuple]:
    """Slots in C, each mapped to the sources that put it there. Ties at the top-k cut break by name."""
    n = space.n_rel
    sources = defaultdict(list)
    for r in np.lexsort((space.rel_name_rank, -q.cos))[:k_b]:
        sources[int(r)].append("bienc")
        sources[int(r) + n].append("bienc")
    for s in np.lexsort((space.name_rank, -q.log_prior))[:k_p]:
        sources[int(s)].append("prior")
    for s in np.flatnonzero(q.present):
        sources[int(s)].append("local")
    return {s: tuple(v) for s, v in sources.items()}


def rank_candidates(q: Question, space: SlotSpace, slots: np.ndarray, T: float, lam: float,
                    gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Order `slots` (the candidate set C) by fused score, best first; ties break by (relation, direction) name."""
    text = q.cos[slots % space.n_rel] / T
    text = text - np.logaddexp.reduce(text)  # log softmax over C
    score = text + lam * q.log_prior[slots] + gamma * q.present[slots]
    order = np.lexsort((space.name_rank[slots], -score))
    return slots[order], score[order]


def relation_order(ranked_slots: np.ndarray, n_rel: int) -> list[int]:
    """Relations in order of their best-ranked direction (max over directions)."""
    seen, rels = set(), []
    for s in ranked_slots:
        r = int(s) % n_rel
        if r not in seen:
            seen.add(r)
            rels.append(r)
    return rels


# ----------------------------------------------------------------------------------------------------------
# GNN Predict, cached per (entity, slot)
# ----------------------------------------------------------------------------------------------------------

class PredictCache:
    """Top-`k_max` existence-filtered GNN candidates per (entity, slot), persisted to disk.

    `known=False` in OneShotInterface.predict_topk removes entities already linked to `e` by that
    (relation, direction) in G_inc, so predicted and observed sets are disjoint.
    """

    def __init__(self, dataset: str, kg: KGInterface, space: SlotSpace, path: Path, k_max: int = 10):
        self.dataset, self.kg, self.space, self.path, self.k_max = dataset, kg, space, Path(path), k_max
        self.table = pickle.loads(self.path.read_bytes()) if self.path.exists() else {}
        self._gnn = None

    def _model(self):
        if self._gnn is None:
            from GoG.gnn_interface import OneShotInterface
            self._gnn = OneShotInterface(self.dataset, self.kg.n_ent, self.kg.n_rel)
            self._gnn.assign_graph(self.kg)
        return self._gnn

    def fill(self, keys, save_every: int = 500):
        from tqdm import tqdm
        missing = sorted(set(keys) - self.table.keys())
        for i, (e, s) in enumerate(tqdm(missing, desc=f"GNN {self.dataset}")):
            rel, direction = self.space.name(s)
            self.table[(e, s)] = list(self._model().predict_topk(e, rel, direction, k=self.k_max, known=False))
            if (i + 1) % save_every == 0:
                self.save()
        if missing:
            self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(pickle.dumps(self.table))

    def get(self, e: str, s: int, K: int) -> list[str]:
        return self.table[(e, s)][:K] if K > 0 else []
