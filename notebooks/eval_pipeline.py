"""Pipeline evaluation on wikidata5m, family or fb15k_237, with the test_generate.ipynb metrics.

Stages:
    relation_inference  hit in top 3 of the (relation, direction) prior, pretrain and finetune
    predict             MRR and Hits@k of the GNN, given the gold relation and direction, plus subgraph
                        coverage: share of questions whose hard answer is in the sampled subgraph, and the
                        median node and edge counts of that subgraph
    prob_propose        Accuracy (propose) = gold slot selected
                        Accuracy (predict) = hard answer among the predicted candidates, which is
                        the end-to-end prob_propose -> predict number, with its normalized precision

Every stage counts only questions with exactly one dropped edge q_entity -> hard_answer, and returns its
numbers as a dict (notebooks/run_eval_bundle.py writes them to a note). The loops follow
notebooks/test_generate_utilities.py, so the numbers match test_generate.ipynb.

relation_inference runs on SubsetKG, which holds only the edges touching the question entities, as
KGInterface needs about 19 GB on wikidata5m. predict and prob_propose need the full KGEnv and GNN.

Run from the repo root:
    python -m notebooks.eval_pipeline [--dataset wikidata5m] [--questions PATH] [--stages ...]
"""
import argparse
import pickle
import statistics
import traceback
from ast import literal_eval
from pathlib import Path

import pandas as pd

from gnn_config import data_config_map, one_shot_subgraph_config
from GoG.GoG_env_tools import KGEnv
from GoG.prob_propose import DEFAULT_PRIOR_VARIANT, PRIOR_VARIANTS, calibration_path, prior_model_path
from notebooks.test_generate_utilities import get_edges
from relation_inference.relation_inference_interface import RelationInferenceInterface

STAGES = ["relation_inference", "predict", "prob_propose"]
DEFAULT_QUESTIONS = {
    "wikidata5m": "brink_dataset/wikidata5m/smoke_val.tsv",
    "family": "brink_dataset/family/val.tsv",
    "fb15k_237": "brink_dataset/fb15k_237/private_id/val.tsv",
}
SAMPLER_ARGS = ("local_ppr", "local_ppr_epsilon", "max_nodes_per_query", "max_edges_per_query")


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
        config = data_config_map[dataset_name]
        self.dataset_name = dataset_name
        relations = read_lines(config["rel_path"])
        if dataset_name == "family":
            relations = [relation + "_of" for relation in relations]  # same fix as KGInterface

        self.incomplete_kg = load_edges_touching(config["incomplete_path"], entities)
        complete_kg = load_edges_touching(config["complete_path"], entities)

        rel_label_path = config.get("rel_label_path")
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


def valid_cases(questions, drop_edges):
    """(case, q_entity, hard_answer, relation, direction) for each question with one dropped edge q -> hard."""
    for case in questions.to_dict(orient="records"):
        q_entity = literal_eval(case["q_entity"])[0]
        hard_answer = literal_eval(case["hard_answer"])[0]
        edges = get_edges(drop_edges, q_entity, hard_answer)
        if len(edges) != 1:
            continue
        direction = "outgoing" if edges["head"].item() == q_entity else "incoming"
        yield case, q_entity, hard_answer, edges["relation"].values[0], direction


def ratio(hit, total):
    return {"hit": hit, "total": total, "rate": hit / total if total else float("nan")}


def evaluate_relation_inference(dataset, questions):
    """Top-3 slot hit of the prior, per stage. A query that raises counts as a miss and is reported."""
    q_entities = {literal_eval(value)[0] for value in questions["q_entity"]}
    kg = SubsetKG(dataset, q_entities)
    cases = list(valid_cases(questions, kg.drop_edges))
    print(f"{len(questions)} questions, {len(q_entities)} q entities, valid cases: {len(cases)}")
    result = {"questions": len(questions), "q_entities": len(q_entities), "valid": len(cases)}
    for stage in ["pretrain", "finetune"]:
        model_path = prior_model_path(dataset, stage)
        interface = RelationInferenceInterface(kg, model_path=model_path)
        hit, errors = 0, []
        for case, q_entity, _, relation, direction in cases:
            try:
                cands = interface.query(q_entity, k=3)
            except Exception as error:  # noqa: BLE001 - counted and reported
                errors.append(f"{case['id']}: {error!r}")
                continue
            hit += any(c["relation"] == relation and c["direction"] == direction for c in cands)
        result[stage] = {**ratio(hit, len(cases)), "model": model_path, "errors": errors}
        print(f"{stage}: top-3 hit {hit}/{len(cases)}, errors {len(errors)}")
    return result


def subgraph_stats(gnn, q_entity, hard_answer):
    """Nodes and edges of the subgraph the sampler gives the GNN for `q_entity`, and whether it holds the answer."""
    hid = gnn.entity2id[q_entity]
    subgraph = gnn.sampler.getOneSubgraph(int(hid))
    if gnn.sampler.ppr_method == "local_push":
        nodes, edges, _ = subgraph
    else:
        _, nodes, _, edges = subgraph
    nodes = [int(n) for n in (nodes.tolist() if hasattr(nodes, "tolist") else nodes)]
    return len(nodes), int(edges.shape[0]), gnn.entity2id[hard_answer] in set(nodes)


def evaluate_predict(kg_env, questions, drop_edges):
    """GNN top-10 given the gold slot (cal_predict_verify_accuracy, verify=False), plus subgraph coverage."""
    total = mrr = 0
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    n_nodes, n_edges, covered = [], [], 0
    for _, q_entity, hard_answer, relation, direction in valid_cases(questions, drop_edges):
        total += 1
        cands = kg_env.gnn.predict_topk(q_entity, relation, direction, k=10, known=False)
        if hard_answer in cands:
            rank = cands.index(hard_answer) + 1
            mrr += 1 / rank
            for k in hits:
                hits[k] += rank <= k
        nodes, edges, inside = subgraph_stats(kg_env.gnn, q_entity, hard_answer)
        n_nodes.append(nodes)
        n_edges.append(edges)
        covered += inside
    result = {
        "valid": total, "mrr": mrr / total if total else float("nan"),
        **{f"hit@{k}": ratio(v, total) for k, v in hits.items()},
        "coverage": ratio(covered, total),
        "median_nodes": statistics.median(n_nodes) if n_nodes else None,
        "median_edges": statistics.median(n_edges) if n_edges else None,
    }
    print(f"predict: valid {total}, MRR {result['mrr']:.4f}, "
          + ", ".join(f"Hit@{k} {v}/{total}" for k, v in hits.items())
          + f", coverage {covered}/{total}, median nodes {result['median_nodes']}, "
            f"median edges {result['median_edges']}")
    return result


def evaluate_prob_propose(kg_env, questions, drop_edges):
    """LLM-free Propose then Predict, as cal_prob_propose_accuracy.

    normalized_precision follows test_generate.ipynb: 1 / (predicted entities of the slot holding the hard
    answer), averaged over all valid questions. normalized_precision_all divides by the predicted entities
    of every selected slot instead.
    """
    total = acc_prop = acc_pred = 0
    n_slots = n_pred = 0
    precision = precision_all = 0.0
    errors = []
    for case, q_entity, hard_answer, relation, direction in valid_cases(questions, drop_edges):
        total += 1
        kg_env.records = [{"i": 1, "thought": "helo"}]
        kg_env.explored_entities = set()
        kg_env.explored_triples = []
        try:
            _, slots = kg_env.prob_propose_action(q_entity, case["question"])
        except Exception as error:  # noqa: BLE001 - e.g. q_entity missing from the graph
            errors.append(f"{case['id']}: {error!r}")
            continue
        n_slots += len(slots)
        all_predicted = sum(len(predicted) for predicted, _ in slots)
        matched_prop = matched_pred = False
        for predicted, proposal in slots:
            entities = [entity for entity, _ in predicted]
            n_pred += len(entities)
            if not matched_prop and proposal.relation == relation and proposal.direction == direction:
                acc_prop += 1
                matched_prop = True
            if not matched_pred and hard_answer in entities:
                acc_pred += 1
                precision += 1 / len(entities)
                precision_all += 1 / all_predicted
                matched_pred = True
    result = {
        "valid": total, "propose": ratio(acc_prop, total), "predict": ratio(acc_pred, total),
        "avg_slots": n_slots / total if total else float("nan"),
        "avg_predicted": n_pred / total if total else float("nan"),
        "normalized_precision": precision / total if total else float("nan"),
        "normalized_precision_all": precision_all / total if total else float("nan"),
        "errors": errors,
    }
    print(f"prob_propose: propose {acc_prop}/{total}, predict {acc_pred}/{total}, "
          f"avg slots {result['avg_slots']:.4f}, avg predicted {result['avg_predicted']:.4f}, "
          f"normalized precision {result['normalized_precision']:.4f}, errors {len(errors)}")
    return result


def sampler_config(dataset, overrides=None):
    """The dataset's GNN args in gnn_config.py, with `overrides` (None values ignored) written into it."""
    args = one_shot_subgraph_config[dataset]["args"]
    args.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return args


def missing_for_kg_env(dataset, prior_variant=DEFAULT_PRIOR_VARIANT, need_calibration=True):
    """What KGEnv would fail on. Checked first, since the failure comes after a long graph load."""
    missing = []
    gnn_args = one_shot_subgraph_config.get(dataset, {}).get("args")
    if not gnn_args:
        return [f'one_shot_subgraph_config["{dataset}"]["args"] in gnn_config.py']
    if not Path(gnn_args["checkpoint"]).exists():
        missing.append(f"GNN checkpoint {gnn_args['checkpoint']}")
    if dataset not in data_config_map:
        missing.append(f'data_config_map["{dataset}"] in gnn_config.py')
    else:
        missing += [f"KG file {path}" for key, path in data_config_map[dataset].items()
                    if key.endswith("_path") and not Path(path).exists()]
    model = Path(prior_model_path(dataset, prior_variant))
    if not model.exists():
        missing.append(f"relation prior {model}")
    calibration = calibration_path(dataset, prior_variant, gnn_args)
    if need_calibration and not calibration.exists():
        missing.append(f"calibration file {calibration}")
    return missing


def load_agent_args(dataset, questions_path, prior_variant=DEFAULT_PRIOR_VARIANT):
    # All datasets share these agent settings, so wikidata5m can reuse the fb15k_237 file.
    args_path = Path(f"sampled_args/sample_args_finetune_{dataset}_2.pkl")
    if not args_path.exists():
        args_path = Path("sampled_args/sample_args_finetune_fb15k_237_2.pkl")
    with open(args_path, "rb") as file:
        args = pickle.load(file)
    args.dataset = Path(questions_path).as_posix()
    args.build_nx_graphs = False
    args.return_raw_propose = True
    args.prior_variant = prior_variant
    return args


def load_kg_env(dataset, questions_path, prior_variant=DEFAULT_PRIOR_VARIANT, kg=None, gnn=None):
    return KGEnv(load_agent_args(dataset, questions_path, prior_variant), kg=kg, gnn=gnn)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="wikidata5m", choices=sorted(DEFAULT_QUESTIONS))
    parser.add_argument("--questions", help="question TSV; defaults to the dataset's smoke or val file")
    parser.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    parser.add_argument("--prior_variant", default=DEFAULT_PRIOR_VARIANT, choices=PRIOR_VARIANTS)
    # Unset keeps the value in gnn_config.py; these apply to this run only.
    parser.add_argument("--local_ppr", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--local_ppr_epsilon", type=float)
    parser.add_argument("--max_nodes_per_query", type=int)
    parser.add_argument("--max_edges_per_query", type=int, help="-1 = no cap")
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
    overrides = {name: getattr(args, name) for name in SAMPLER_ARGS if getattr(args, name) is not None}
    sampler_config(args.dataset, overrides)
    print(f"sampler overrides: {overrides}")
    missing = missing_for_kg_env(args.dataset, args.prior_variant)
    if missing:
        raise SystemExit(f"\nCannot run {env_stages} on {args.dataset}, missing:\n  " + "\n  ".join(missing))

    kg_env = load_kg_env(args.dataset, questions_path, args.prior_variant)
    drop_edges = kg_env.kg.drop_edges
    for stage, evaluate in (("predict", evaluate_predict), ("prob_propose", evaluate_prob_propose)):
        if stage in env_stages:
            print(f"\n===== {stage} =====")
            try:
                evaluate(kg_env, questions, drop_edges)
            except Exception:  # noqa: BLE001 - keep the other stage
                traceback.print_exc()


if __name__ == "__main__":
    main()
