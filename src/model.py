from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PARAM_GRID = {
    "svc__estimator__C": [0.1, 1, 10, 100],
    "svc__estimator__gamma": ["scale", 0.1, 0.01],
    "svc__estimator__kernel": ["rbf", "linear"],
}


def build_model():
    svm = SVC(class_weight="balanced", random_state=42)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", CalibratedClassifierCV(svm, method="sigmoid", cv=5, ensemble=False)),
    ])


def train_tuned_model(X_train, y_train, cv=5, scoring="f1"):
    grid = GridSearchCV(
        build_model(),
        PARAM_GRID,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring=scoring,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best_params = {k.replace("svc__estimator__", ""): v for k, v in grid.best_params_.items()}
    return grid.best_estimator_, best_params
