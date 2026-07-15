from evaluation import PredictionEntry, load_predictions
import json
import argparse
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_path",
        type=str,
        default="results/gpt-4o-mini/family/1_10_3_0.7_test_predictions.json",
        help="Path to the predictions JSON file.",
    )
    args = parser.parse_args()
    
    prediction_path = Path(args.prediction_path)
    predictions = load_predictions(str(prediction_path))

    # Generate failed cases filename based on predictions filename
    prediction_filename = prediction_path.stem  # e.g., "1_10_3_0.7_test_predictions" or "predictions"
    if "_predictions" in prediction_filename:
        failed_cases_filename = prediction_filename.replace("_predictions", "_failed_cases") + ".json"
    else:
        failed_cases_filename = prediction_filename.replace("predictions", "failed_cases") + ".json"
    failed_cases_path = prediction_path.parent / failed_cases_filename

    print(f"Predictions path: {prediction_path}")
    print(f"Failed cases path: {failed_cases_path}")

    failed_cases: list[dict] = []
    for pred in predictions:
        if not pred.hits_any:
            case_dict = pred.model_dump()
            for k, v in case_dict.items():
                if isinstance(v, set):
                    case_dict[k] = list(v)
            failed_cases.append(case_dict)

    print(f"Found {len(failed_cases)} failed cases")
    
    with open(failed_cases_path, "w") as f:
        json.dump([case for case in failed_cases], f, indent=4)
    
    print(f"Failed cases saved to: {failed_cases_path}")
