"""Small, class-balanced SVM; scaling fitted inside each group fold."""
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import profile_groups

PARAM_GRID = [
    {"svc__kernel": ["linear"], "svc__C": [0.01, 0.1, 1, 10]},
    {"svc__kernel": ["rbf"], "svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 0.01, 0.1]},
]
SCREENING_SCORER = make_scorer(fbeta_score, beta=2, zero_division=0)


def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(class_weight="balanced", random_state=42)),
    ])


def train_tuned_model(X_train, y_train, cv=3):
    grid = GridSearchCV(
        build_model(), PARAM_GRID,
        cv=StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring=SCREENING_SCORER, n_jobs=1, error_score="raise",
    )
    grid.fit(X_train, y_train, groups=profile_groups(X_train))
    best_params = {k.removeprefix("svc__"): v for k, v in grid.best_params_.items()}
    return grid.best_estimator_, best_params
