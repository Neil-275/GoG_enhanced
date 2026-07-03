#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def convert_jsonl_to_json(jsonl_path: str | Path, output_path: str | Path | None = None) -> Path:
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Predictions JSONL file not found: {jsonl_path}")

    with jsonl_path.open("r", encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f if line.strip()]

    predictions = sorted(predictions, key=lambda item: item["index"])
    output_path = Path(output_path) if output_path else jsonl_path.with_suffix(".json")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    return output_path


def failed_cases_path(prediction_path: str | Path) -> Path:
    prediction_path = Path(prediction_path)
    stem = prediction_path.stem
    if "_predictions" in stem:
        filename = stem.replace("_predictions", "_failed_cases") + ".json"
    else:
        filename = stem.replace("predictions", "failed_cases") + ".json"
    return prediction_path.parent / filename


def evaluation_path(prediction_path: str | Path) -> Path:
    prediction_path = Path(prediction_path)
    return prediction_path.parent / (
        prediction_path.stem.replace("_predictions", "_evaluation") + ".json"
    )


def load_prediction_dicts(prediction_path: str | Path) -> list[dict]:
    prediction_path = Path(prediction_path)
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")

    with prediction_path.open("r", encoding="utf-8") as f:
        try:
            predictions = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            predictions = [json.loads(line) for line in f if line.strip()]

    if isinstance(predictions, dict):
        return [predictions]
    return predictions


def _as_set(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if item is not None}
    return {str(value)}


def _entry_sets(entry: dict) -> tuple[set[str], set[str], set[str]]:
    prediction = _as_set(entry.get("prediction"))
    answers = _as_set(entry.get("answers", entry.get("ground_truth")))
    hard_answer = _as_set(entry.get("hard_answer", entry.get("ground_truth_entities")))
    return prediction, answers, hard_answer


def evaluate_predictions(predictions: list[dict]) -> dict:
    total_hits_any = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_hits_hard = 0.0
    num_missing_predictions = 0

    for entry in predictions:
        prediction, answers, hard_answer = _entry_sets(entry)
        ground_truth = answers.union(hard_answer)
        overlap = prediction.intersection(ground_truth)

        if not prediction:
            num_missing_predictions += 1
            continue

        precision = len(overlap) / len(prediction) if prediction else 0.0
        recall = len(overlap) / len(ground_truth) if ground_truth else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        if overlap:
            total_hits_any += 1
        if prediction.intersection(hard_answer):
            total_hits_hard += 1
        total_precision += precision
        total_recall += recall
        total_f1 += f1_score

    num_entries = len(predictions)
    if num_entries == 0:
        return {
            "num_examples": 0,
            "num_missing_predictions": 0,
            "Hits@Any": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0,
            "Hits@Hard": 0.0,
            "HHR": 0.0,
        }

    results = {
        "num_examples": num_entries,
        "num_missing_predictions": num_missing_predictions,
        "Hits@Any": total_hits_any / num_entries,
        "Precision": total_precision / num_entries,
        "Recall": total_recall / num_entries,
        "F1": total_f1 / num_entries,
        "Hits@Hard": total_hits_hard / num_entries,
    }
    results["HHR"] = (
        results["Hits@Hard"] / results["Hits@Any"] if results["Hits@Any"] > 0 else 0.0
    )
    return results


def extract_failed_cases(prediction_path: str | Path, output_path: str | Path | None = None) -> Path:
    prediction_path = Path(prediction_path)
    output_path = Path(output_path) if output_path else failed_cases_path(prediction_path)

    failed_cases = []
    for entry in load_prediction_dicts(prediction_path):
        prediction, answers, hard_answer = _entry_sets(entry)
        overlap = prediction.intersection(answers.union(hard_answer))
        if overlap:
            continue

        case_dict = dict(entry)
        case_dict["overlap"] = []
        failed_cases.append(case_dict)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(failed_cases, f, indent=4)

    return output_path


def run_evaluation(prediction_path: str | Path, output_path: str | Path | None = None) -> Path:
    prediction_path = Path(prediction_path)
    output_path = Path(output_path) if output_path else evaluation_path(prediction_path)

    results = evaluate_predictions(load_prediction_dicts(prediction_path))
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f)

    return output_path


def postprocess_prediction_jsonl(jsonl_path: str | Path) -> dict[str, Path]:
    prediction_path = convert_jsonl_to_json(jsonl_path)
    failed_path = extract_failed_cases(prediction_path)
    eval_path = run_evaluation(prediction_path)
    return {
        "predictions": prediction_path,
        "failed_cases": failed_path,
        "evaluation": eval_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a prediction JSONL file, extract failed cases, and run evaluation."
    )
    parser.add_argument("prediction_jsonl", help="Path to the prediction JSONL file")
    args = parser.parse_args()

    outputs = postprocess_prediction_jsonl(args.prediction_jsonl)
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
