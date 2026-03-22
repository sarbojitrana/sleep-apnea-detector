
from src.data import load_data, preprocess_data, encode_data, prepare_data
from src.model import  train_tuned_model
from src.evalaute import evaluate_model

df = load_data("data/Sleep_health_and_lifestyle_dataset.csv")
df = preprocess_data(df)

df, encoders = encode_data(df)

X_train, X_test, y_train, y_test, scaler = prepare_data(df)


print("Train Shape: ", X_train.shape)
print("Test Shape: ", X_test.shape)

print("\nSample Features:\n", X_train[:5])

model, best_params = train_tuned_model(X_train, y_train)


print("\n Model trained successfully")

print("\n Best Params : ", best_params)


acc, report, matrix = evaluate_model(model, X_test, y_test)

print("\nAccuracy:", acc)
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", matrix)