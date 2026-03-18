
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    
    df = df.drop(columns = ["Person ID"])
    
    df = df.dropna(subset = ["Sleep Disorder"])
    
    df["Sleep Disorder"] = df["Sleep Disorder"].apply(
        lambda x: 1 if x == "Sleep Apnea" else 0
    )
    
    bp_split = df["Blood Pressure"].str.split("/", expand =True)
    df["Systolic"] = bp_split[0].astype(int)
    df["Diastolic"] = bp_split[0].astype(int)
    
    
    df = df.drop(columns = ["Blood Pressure"])
    
    return df

from sklearn.preprocessing import LabelEncoder


def encode_data(df):
    df = df.copy()

    categorical_cols = ["Gender", "Occupation", "BMI Category"]

    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    df = df.astype(float)

    return df, encoders