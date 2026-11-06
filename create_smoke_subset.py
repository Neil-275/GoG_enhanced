"""Build the wikidata5m question files in the family/fb15k_237 schema, plus a smoke-test subset.

The benchmark splits one schema over two files: `{split}_cleaned_final_complete.tsv` holds every
answer, and `{split}_cleaned_final.tsv` holds only the hard one. GoG.py needs both in one row, so
this writes `answer` (all answers) and `hard_answer` together. Easy answers are answer - hard_answer.

Writes, next to the source files (the parent folder selects the KG in GoG.py):
    val.tsv, test.tsv  the full splits
    smoke_val.tsv      a subset of val only, so test stays unseen while experimenting

The subset is sampled systematically over val sorted by (target relation, rule, q direction,
q degree), so every stratum gets a proportional share. Same arguments give the same file.

Usage: python create_smoke_subset.py [--n 500] [--seed 42]
"""
import argparse
from ast import literal_eval
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "brink_dataset" / "wikidata5m"
DEGREE_BINS = [-1, 3, 10, 30, 100, 1000, np.inf]
DEGREE_LABELS = ["0-3", "4-10", "11-30", "31-100", "101-1000", ">1000"]
OUTPUT_COLUMNS = ["id", "rule_index", "question", "q_entity", "answer", "a_entity", "hard_answer", "rule", "grounding"]


def load_split(split):
    questions = pd.read_csv(DATA_DIR / f"{split}_cleaned_final_complete.tsv", sep="\t")
    hard_only = pd.read_csv(DATA_DIR / f"{split}_cleaned_final.tsv", sep="\t")
    assert (questions["id"] == hard_only["id"]).all(), f"{split}: the two files are not row-aligned"
    questions["hard_answer"] = hard_only["answer"]
    return questions


def add_strata(questions):
    q_entity = questions["q_entity"].map(lambda value: literal_eval(value)[0])
    grounding = questions["grounding"].map(lambda value: literal_eval(value)[0])
    conclusion = grounding.map(lambda value: value.split("=>")[1].split())

    strata = questions.assign(
        q=q_entity,
        target_relation=conclusion.str[1],
        direction=np.where(conclusion.str[0] == q_entity, "head", "tail"),
        hops=grounding.map(lambda value: value.split("=>")[0].count("|") + 1),
    )
    degrees = count_degrees(set(q_entity))
    strata["q_degree"] = q_entity.map(lambda entity: degrees[entity])
    strata["degree_bin"] = pd.cut(strata["q_degree"], DEGREE_BINS, labels=DEGREE_LABELS).astype(str)
    return strata


def count_degrees(entities):
    """Degree in the incomplete graph, the one the agent actually walks."""
    degrees = Counter()
    with open(DATA_DIR / "incomplete_facts.tsv", encoding="utf-8") as facts:
        for line in facts:
            head, _, tail = line.rstrip("\n").split("\t")
            if head in entities:
                degrees[head] += 1
            if tail in entities:
                degrees[tail] += 1
    return degrees


def sample_systematic(questions, n, seed):
    rng = np.random.default_rng(seed)
    shuffled = questions.iloc[rng.permutation(len(questions))]  # random tie-break inside a stratum
    ordered = shuffled.sort_values(
        ["target_relation", "rule_index", "direction", "q_degree"], kind="stable"
    )
    step = len(ordered) / n
    positions = (rng.uniform(0, step) + step * np.arange(n)).astype(int)
    return ordered.iloc[positions]


def share(column):
    return column.value_counts(normalize=True)


def total_variation(full, subset):
    return 0.5 * share(full).subtract(share(subset), fill_value=0).abs().sum()


def report(full, subset):
    print(f"\n{len(subset)} of {len(full)} val questions, {subset['id'].nunique()} distinct ids")
    print(f"rules covered: {subset['rule_index'].nunique()}/{full['rule_index'].nunique()}")
    print(f"target relations covered: {subset['target_relation'].nunique()}/{full['target_relation'].nunique()}")
    print(f"repeated question text: {subset['question'].duplicated().sum()}")
    print(f"{'feature':<16}{'TV distance (0 = same distribution)':>36}")
    for feature in ["target_relation", "rule_index", "hops", "direction", "degree_bin"]:
        print(f"{feature:<16}{total_variation(full[feature], subset[feature]):>36.3f}")

    table = pd.DataFrame({"full": share(full["degree_bin"]), "subset": share(subset["degree_bin"])})
    print("\n", table.reindex(DEGREE_LABELS).fillna(0).round(3).to_string(), sep="")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    val = load_split("val")
    test = load_split("test")
    assert not set(val["id"]) & set(test["id"]), "val and test ids collide, GoG.py keys results on id"

    val_strata = add_strata(val)
    subset = sample_systematic(val_strata, args.n, args.seed)
    report(val_strata, subset)

    val[OUTPUT_COLUMNS].to_csv(DATA_DIR / "val.tsv", sep="\t", index=False)
    test[OUTPUT_COLUMNS].to_csv(DATA_DIR / "test.tsv", sep="\t", index=False)
    subset[OUTPUT_COLUMNS].to_csv(DATA_DIR / "smoke_val.tsv", sep="\t", index=False)
    print(f"\nWrote val.tsv, test.tsv and smoke_val.tsv to {DATA_DIR}")


if __name__ == "__main__":
    main()
