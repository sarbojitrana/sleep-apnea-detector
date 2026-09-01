RISK_RULES = [
    ("Obese", lambda f: f["BMI Category"] >= 2, 35),
    ("Overweight", lambda f: f["BMI Category"] == 1, 25),
    ("Short sleep (< 6h)", lambda f: f["Sleep Duration"] < 6, 20),
    ("Elevated resting heart rate (> 80)", lambda f: f["Heart Rate"] > 80, 15),
    ("High stress (> 7)", lambda f: f["Stress Level"] > 7, 15),
    ("Hypertensive range (>= 140/90)", lambda f: f["Systolic"] >= 140 or f["Diastolic"] >= 90, 15),
]

MAX_RISK = sum(points for _, _, points in RISK_RULES)


def calculate_risk(features):
    score = 0
    reasons = []

    for name, rule, points in RISK_RULES:
        if rule(features):
            score += points
            reasons.append(name)

    return min(score, 100), reasons


def classify_severity(score):
    if score < 30:
        return "Low"
    if score < 60:
        return "Moderate"
    return "High"
