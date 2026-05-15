from evaluation import PredictionEntry, load_predictions
import json


if __name__ == "__main__":
    prediction_path = "results/v2/family/prediction.json"
    predictions = load_predictions(prediction_path)

    failed_cases: list[dict] = []
    for pred in predictions:
        if not pred.hits_any:
            case_dict = pred.model_dump()
            for k, v in case_dict.items():
                if isinstance(v, set):
                    case_dict[k] = list(v)
            failed_cases.append(case_dict)

    with open("results/v2/family/failed_cases.json", "w") as f:
        json.dump([case for case in failed_cases], f, indent=4)
