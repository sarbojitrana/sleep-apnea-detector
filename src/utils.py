def calculate_risk(df_row):
    score = 0

    if df_row["BMI Category"] > 1:
        score += 20

    if df_row["Stress Level"] > 7:
        score += 20

    if df_row["Sleep Duration"] < 6:
        score += 20

    if df_row["Heart Rate"] > 80:
        score += 20

    return score

def classify_severity(score):
    if score < 30:
        return "Low"
    elif score < 60:
        return "Moderate"
    else:
        return "High"