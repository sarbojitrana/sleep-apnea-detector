
import pandas as pd

df = pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")

print(df.head())
print("\nColumns:\n", df.columns)
print("\nInfo:\n")
print(df.info())