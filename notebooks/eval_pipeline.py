"""Pipeline evaluation on wikidata5m, family or fb15k_237, with the test_generate.ipynb metrics.

Stages:
    relation_inference  hit in top 3 of the (relation, direction) prior, pretrain and finetune
    predict             MRR and Hits@k of the GNN, given the gold relation and direction
    prob_propose        Accuracy (propose) = gold slot selected
                        Accuracy (predict) = hard answer among the predicted candidates, which is
                        the end-to-end prob_propose -> predict number, with its normalized precision

relation_inference runs on SubsetKG, which holds only the edges touching the question entities, as
KGInterface needs about 19 GB on wikidata5m. predict and prob_propose need the full KGEnv and GNN.

Run from the repo root:
    python -m notebooks.eval_pipeline [--dataset wikidata5m] [--questions PATH] [--stages ...]
"""
import argparse
import pickle
from ast import literal_eval
from pathlib import Path

import pandas as pd

from gnn_config import one_shot_subgraph_config
from GoG.GoG_env_tools import KGEnv
from GoG.kg_interface import DATASET_CONFIG_MAP
from GoG.prob_propose import CALIBRATION_DIR, PRIOR_PATHS
from notebooks.test_generate_utilities import (
    cal_predict_verify_accuracy,
    cal_prob_propose_accuracy,
    cal_relation_inference_performance,
    get_edges,
)
from relation_inference.relation_inference_interface import RelationInferenceInterface

MODEL_DIR = Path("relation_inference/model")
STAGES = ["relation_inference", "predict", "prob_propose"]
DEFAULT_QUESTIONS = {
    "wikidata5m": "brink_dataset/wikidata5m/smoke_val.tsv",
    "family": "brink_dataset/family/val.tsv",
    "fb15k_237": "brink_dataset/fb15k_237/private_id/val.tsv",
}


def read_lines(path):
    with open(path, encoding="utf-8") as lines:
        return [line.strip() for line in lines]


def load_edges_touching(path, entities):
    rows = []
    with open(path, encoding="utf-8") as facts:
        for line in facts:
            head, relation, tail = line.rstrip("\n").replace('"', "").split("\t")
            if head in entities or tail in entities:
                rows.append((head, relation, tail))
    return pd.DataFrame(rows, columns=["head", "relation", "tail"])


class SubsetKG:
    """What RelationInferenceInterface and the eval read from KGInterface, for `entities` only."""

    def __init__(self, dataset_name, entities):
        config = DATASET_CONFIG_MAP[dataset_name]
        self.dataset_name = dataset_name
        relations = read_lines(config.rel_path)
        if dataset_name == "family":
            relations = [relation + "_of" for relation in relations]  # same fix as KGInterface

        self.incomplete_kg = load_edges_touching(config.incomplete_path, entities)
        complete_kg = load_edges_touching(config.complete_path, entities)

        rel_label_path = getattr(config, "rel_label_path", None)
        if rel_label_path:
            labels = read_lines(rel_label_path)
            assert len(labels) == len(relations), "relation labels must be line-aligned with relations"
            id2label = dict(zip(relations, labels))
            self.incomplete_kg["relation"] = self.incomplete_kg["relation"].map(id2label)
            complete_kg["relation"] = complete_kg["relation"].map(id2label)
            relations = labels

        self.n_rel = len(relations)
        self.rel2id = {relation: index for index, relation in enumerate(relations)}
        self.drop_edges = pd.concat([complete_kg, self.incomplete_kg]).drop_duplicates(keep=False)

    def get_1hop_triples(self, entity, kg_type="incomplete"):
        assert kg_type == "incomplete", "SubsetKG keeps only the incomplete graph for lookups"
        edges = self.incomplete_kg
        # Concatenated like KGInterface, which returns a self-loop twice: the model's input depends on it.
        return pd.concat([edges[edges["head"] == entity], edges[edges["tail"] == entity]]).reset_index(drop=True)


def count_valid_cases(questions, drop_edges):
    """Cases the eval keeps: exactly one dropped edge q -> hard_answer."""
    total = 0
    for case in questions.to_dict(orient="records"):
        q_entity = literal_eval(case["q_entity"])[0]
        hard_answer = literal_eval(case["hard_answer"])[0]
        total += len(get_edges(drop_edges, q_entity, hard_answer)) == 1
    return total


def evaluate_relation_inference(dataset, questions):
    q_entities = {literal_eval(value)[0] for value in questions["q_entity"]}
    kg = SubsetKG(dataset, q_entities)
    print(f"{len(questions)} questions, {len(q_entities)} q entities")
    print(f"edges kept: {len(kg.incomplete_kg)}, dropped: {len(kg.drop_edges)}")
    print(f"valid cases: {count_valid_cases(questions, kg.drop_edges)}/{len(questions)}")

    for stage in ["pretrain", "finetune"]:
        print(f"\n{stage}")
        interface = RelationInferenceInterface(kg, model_path=MODEL_DIR / f"{dataset}_{stage}.pt")
        corrupts = cal_relation_inference_performance(None, questions, interface, kg.drop_edges)
        print(f"Corrupts: {len(corrupts)}")


def missing_for_kg_env(dataset):
    """What KGEnv would fail on. Checked first, since the failure comes after a long graph load."""
    missing = []
    gnn_args = one_shot_subgraph_config[dataset].get("args")
    if not gnn_args:
        missing.append(f'one_shot_subgraph_config["{dataset}"]["args"] in gnn_config.py')
    elif not Path(gnn_args["checkpoint"]).exists():
        missing.append(f"GNN checkpoint {gnn_args['checkpoint']}")
    if dataset not in PRIOR_PATHS:
        missing.append(f'PRIOR_PATHS["{dataset}"] in GoG/prob_propose.py')
    else:
        mode = Path(PRIOR_PATHS[dataset]).stem.split("_")[-1]
        calibration = CALIBRATION_DIR / f"calibration_results_{dataset}_{mode}.pkl"
        if not calibration.exists():
            missing.append(f"calibration file {calibration}")
    return missing


def load_kg_env(dataset, questions_path):
    # All datasets share these agent settings, so wikidata5m can reuse the fb15k_237 file.
    args_path = Path(f"sampled_args/sample_args_finetune_{dataset}_2.pkl")
    if not args_path.exists():
        args_path = Path("sampled_args/sample_args_finetune_fb15k_237_2.pkl")
    with open(args_path, "rb") as file:
        args = pickle.load(file)
    args.dataset = Path(questions_path).as_posix()
    return KGEnv(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="wikidata5m", choices=sorted(DEFAULT_QUESTIONS))
    parser.add_argument("--questions", help="question TSV; defaults to the dataset's smoke or val file")
    parser.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    args = parser.parse_args()

    questions_path = args.questions or DEFAULT_QUESTIONS[args.dataset]
    questions = pd.read_csv(questions_path, sep="\t")
    print(f"{args.dataset}: {questions_path}")

    if "relation_inference" in args.stages:
        print("\n===== relation_inference =====")
        evaluate_relation_inference(args.dataset, questions)

    env_stages = [stage for stage in args.stages if stage != "relation_inference"]
    if not env_stages:
        return
    missing = missing_for_kg_env(args.dataset)
    if missing:
        raise SystemExit(f"\nCannot run {env_stages} on {args.dataset}, missing:\n  " + "\n  ".join(missing))

    kg_env = load_kg_env(args.dataset, questions_path)
    drop_edges = kg_env.kg.drop_edges
    if "predict" in env_stages:
        print("\n===== predict =====")
        cal_predict_verify_accuracy(kg_env, questions, drop_edges, verify=False)
    if "prob_propose" in env_stages:
        print("\n===== prob_propose -> predict =====")
        cal_prob_propose_accuracy(kg_env, questions, drop_edges)


if __name__ == "__main__":
    main()
