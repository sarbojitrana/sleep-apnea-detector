
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


TARGET = "Sleep Disorder"

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    
    df = df.drop(columns = ["Person ID"])               # not required in mathematical context
    
    df = df.dropna(subset = [TARGET])         #target vector(y)
    
    bp_split = df["Blood Pressure"].str.split("/", expand =True)
    
    df["Systolic"] = bp_split[0].astype(int)
    df["Diastolic"] = bp_split[1].astype(int)
    
    # Pulse Pressure (important cardiovascular indicator)
    df["Pulse Pressure"] = df["Systolic"] - df["Diastolic"]
    
    # Sleep Efficiency Proxy
    df["Sleep Efficiency"] = df["Quality of Sleep"] / df["Sleep Duration"]
    
    # Activity Intensity
    df["Activity Ratio"] = df["Daily Steps"] / (df["Physical Activity Level"] + 1)
        
    df["Sleep Disorder"] = df["Sleep Disorder"].apply(
        lambda x: 1 if x == "Sleep Apnea" else 0
    )
    
    
    
    
    
    df = df.drop(columns = ["Blood Pressure"])
    
    return df


def encode_data(df):            #give numeric value to every parameter
    df = df.copy()

    categorical_cols = ["Gender", "Occupation", "BMI Category"]

    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    df = df.astype(float)

    return df, encoders


def prepare_data(df):
    X = df.drop(columns=["Sleep Disorder"])
    y = df["Sleep Disorder"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler