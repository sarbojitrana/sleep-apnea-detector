
from src.data import load_data, clean_data, encode_data

df = load_data("data/Sleep_health_and_lifestyle_dataset.csv")
df = clean_data(df)

df, encoders = encode_data(df)

print("After Encoding: \n")
print(df.head())


print("\nData Types:\n")
print(df.dtypes)