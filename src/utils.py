"""Unscored symptom prompts, separate from the trained SVM."""
DISCLAIMER = (
    "Research screening prototype, not a diagnosis or an emergency monitor. "
    "A lower model signal cannot rule out sleep apnea. "
    "Discuss witnessed breathing pauses, persistent loud snoring, or daytime sleepiness "
    "with a clinician, regardless of the model result."
)
SYMPTOM_LABELS = {
    "loud_snoring": "Loud snoring reported",
    "witnessed_pauses": "Witnessed breathing pauses or gasping reported",
    "daytime_sleepiness": "Daytime sleepiness reported",
}


def reported_symptoms(record):
    return [label for key, label in SYMPTOM_LABELS.items() if record.get(key) is True]
