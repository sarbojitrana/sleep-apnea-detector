from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# def train_model(X_train, y_train):
#     model = SVC(kernel="rbf", C=1.0, gamma="scale")         
#     model.fit(X_train, y_train)
    
#     return model


def train_tuned_model(X_train, y_train):
    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", 0.1, 0.01],
        "kernel": ["rbf", "linear"]
    }

    grid = GridSearchCV(SVC(), param_grid, cv=5, scoring="recall")

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_