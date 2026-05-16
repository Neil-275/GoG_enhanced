from argparse import ArgumentParser
from pathlib import Path
import json
from json import JSONDecodeError
from pydantic import BaseModel, computed_field


current_dir = Path(__file__).parent


class PredictionEntry(BaseModel):
    index: int | str
    question: str
    prediction: set[str] | None = None
    answers: set[str]
    hard_answer: set[str]
    records: list | None = None
    error: str | None = None

    @computed_field
    def overlap(self) -> set[str]:
        return self.prediction.intersection(
            self.answers.union(self.hard_answer)
        ) if self.prediction is not None else set()

    @property
    def hits_any(self) -> bool:
        return bool(self.overlap)

    @property
    def hits_hard(self) -> bool:
        return bool(self.prediction.intersection(self.hard_answer))

    @property
    def precision(self) -> float:
        if not self.prediction:
            return 0.0
        return len(self.overlap) / len(self.prediction)

    @property
    def recall(self) -> float:
        if not self.answers.union(self.hard_answer):
            return 0.0
        return len(self.overlap) / len(self.answers.union(self.hard_answer))

    @property
    def f1_score(self) -> float:
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)


def _resolve_prediction_path(prediction_file: str) -> Path:
    prediction_file_path = Path(prediction_file)
    if prediction_file_path.is_absolute():
        return prediction_file_path

    relative_path = current_dir / prediction_file_path
    if relative_path.exists():
        return relative_path

    return prediction_file_path


def load_predictions(prediction_file: str) -> list[PredictionEntry]:
    prediction_file_path = _resolve_prediction_path(prediction_file)
    if not prediction_file_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {prediction_file_path}"
        )

    with open(prediction_file_path, "r") as f:
        try:
            raw_predictions = json.load(f)
        except JSONDecodeError:
            f.seek(0)
            raw_predictions = [json.loads(line) for line in f if line.strip()]

    if isinstance(raw_predictions, dict):
        raw_predictions = [raw_predictions]

    return [PredictionEntry.model_validate(entry) for entry in raw_predictions]


def evaluate(prediction_file: str):
    # Validate the prediction file path
    predictions = load_predictions(prediction_file)

    total_hits_any = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_hits_hard = 0.0
    num_missing_predictions = 0

    for entry in predictions:
        if not entry.prediction:
            num_missing_predictions += 1
            continue
        if entry.hits_any:
            total_hits_any += 1
        if entry.hits_hard:
            total_hits_hard += 1
        total_precision += entry.precision
        total_recall += entry.recall
        total_f1 += entry.f1_score

    num_entries = len(predictions)

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
        results["Hits@Hard"] / results["Hits@Any"]
        if results["Hits@Any"] > 0
        else 0.0
    )

    return results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--prediction_file",
        type=str,
        default="./predictions.json",
        help="Path to the prediction file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the evaluation results",
    )

    args = parser.parse_args()

    results = evaluate(args.prediction_file)
    print(f"Evaluation Results: {results}")

    if args.output_file:
        output_file_path = Path(args.output_file)
        if not output_file_path.parent.exists():
            raise FileNotFoundError(
                f"Output directory does not exist: {output_file_path.parent}"
            )
        with open(args.output_file, "w") as f:
            json.dump(results, f)
