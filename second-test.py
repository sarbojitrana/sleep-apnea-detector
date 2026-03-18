from src.data import load_data, clean_data

df = load_data("data/Sleep_health_and_lifestyle_dataset.csv")
df = clean_data(df)

print("After Cleaning:\n")
print(df.head())

print("\nShape:", df.shape)
print("\nTarget Distribution:\n", df["Sleep Disorder"].value_counts())