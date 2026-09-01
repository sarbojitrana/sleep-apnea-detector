from src.data import FEATURES, build_features, load_dataset, split_data
from src.evaluate import evaluate_model
from src.model import train_tuned_model
from src.pipeline import predict


def main():
    X, y = load_dataset()
    assert len(X) == 374, f"expected 374 rows, got {len(X)}"
    assert list(X.columns) == FEATURES
    assert y.sum() == 78, f"expected 78 positive cases, got {y.sum()}"

    X_train, X_test, y_train, y_test = split_data(X, y)
    model, best_params = train_tuned_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    assert metrics["recall"] >= 0.75, f"recall regressed to {metrics['recall']:.3f}"
    assert metrics["accuracy"] >= 0.85, f"accuracy regressed to {metrics['accuracy']:.3f}"

    healthy = predict(model, {"age": 25, "sleep_duration": 8.0, "bmi": "Normal", "heart_rate": 65})
    at_risk = predict(model, {"age": 60, "sleep_duration": 5.0, "bmi": "Obese",
                              "heart_rate": 90, "stress": 9, "systolic": 145, "diastolic": 95})

    assert at_risk["probability"] > healthy["probability"]
    assert at_risk["risk_score"] > healthy["risk_score"]
    assert build_features({"age": 30}).shape == (1, len(FEATURES))

    print(f"best params      : {best_params}")
    print(f"test accuracy    : {metrics['accuracy']:.3f}")
    print(f"test recall      : {metrics['recall']:.3f}")
    print(f"test f1          : {metrics['f1']:.3f}")
    print(f"healthy profile  : {healthy['probability']:.1%} apnea probability")
    print(f"at-risk profile  : {at_risk['probability']:.1%} apnea probability")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
