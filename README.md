# Sleep Apnea Detector

A lightweight sleep apnea **screener** built on a support vector machine. It takes the handful of
measurements a wearable or bedside IoT device already collects — sleep duration, resting heart rate,
step count, BMI, blood pressure — and returns a calibrated probability that the person has sleep
apnea, plus an explainable risk score.

The trained model is a **14 KB file** that predicts in **~2 ms** on CPU with no GPU, no network call,
and no runtime dependency beyond scikit-learn.

---

## What this solves

Sleep apnea is badly under-diagnosed. Confirming it requires polysomnography — an overnight sleep
study in a lab, which is expensive, has long waiting lists, and requires the patient to already
suspect something is wrong. Most people with apnea never get that far.

Meanwhile, a cheap fitness band already measures most of the things that correlate with apnea risk.
The gap is not sensing, it is triage: deciding *which* users should be told "this is worth getting
checked out."

This project fills that gap. It is designed to run **on the device**, so that:

- **No data leaves the device.** Sleep and vitals data is sensitive; keeping inference local avoids
  the entire problem of shipping it to a server.
- **It works offline.** A bedside monitor should not need connectivity to give a result.
- **It fits in a microcontroller's budget.** 81 support vectors and a 14-element feature vector is
  a few thousand multiply-accumulates — portable to C if needed.

What this is **not**: a diagnosis. It is a screening aid whose job is to catch people who should see
a doctor. That framing drives most of the design decisions below.

---

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Train the model once — this writes `models/apnea_svm.joblib`:

```bash
python app.py train
```

Then screen someone from the command line:

```bash
python app.py predict --age 58 --sleep 5.2 --bmi Obese --heart 88 --stress 8 \
                      --systolic 142 --diastolic 92
```

```
=== SLEEP APNEA SCREENING ===
Prediction  : LIKELY APNEA
Confidence  : 72.4%
Risk score  : 100/100 (High)
Risk factors:
  - Obese
  - Short sleep (< 6h)
  - Elevated resting heart rate (> 80)
  - High stress (> 7)
  - Hypertensive range (>= 140/90)
```

Every flag has a sensible default, so you only pass what you actually measured — `python app.py
predict --age 30 --bmi Overweight` is valid. Run `python app.py predict --help` for the full list.

Or use the web UI, which loads the saved model (and trains one automatically if none exists):

```bash
streamlit run ui.py
```

To verify everything still works after a change:

```bash
python test.py
```

---

## How it works

```
CSV  ──▶  load_dataset()  ──▶  StandardScaler  ──▶  SVC (RBF/linear)  ──▶  Platt calibration
                                                                                   │
device readings ──▶ build_features() ─────────────────────────────────────────────┤
                                │                                                  ▼
                                └──────────▶ rule-based risk score ──────▶  probability + risk
```

**1. Feature construction.** Eleven raw readings become fourteen features. Three are derived:

| Derived feature | Formula | Why |
|---|---|---|
| `Pulse Pressure` | systolic − diastolic | Arterial stiffness marker; apnea drives cardiovascular strain |
| `Sleep Efficiency` | quality ÷ duration | Separates "slept 8h badly" from "slept 5h well" |
| `Activity Ratio` | steps ÷ (active minutes + 1) | Distinguishes sustained exercise from incidental walking |

Critically, `build_features()` in [src/data.py](src/data.py) is the **single** implementation used
by both training and inference. Training-time and inference-time feature code that drifts apart is
the classic way an ML system silently breaks; here there is only one copy to get wrong.

**2. The model** is a scikit-learn `Pipeline`: `StandardScaler` → `SVC` → `CalibratedClassifierCV`.
Putting the scaler *inside* the pipeline means the fold-specific scaling is refit inside every
cross-validation fold, so the reported scores contain no leakage from test to train.

**3. The rule score** is a separate, hand-written checklist in [src/utils.py](src/utils.py) — points
for obesity, short sleep, elevated heart rate, high stress, and hypertension. It never touches the
model.

---

## Why these choices

**Why SVM.** With 374 samples and 14 features, this is a small-n problem. SVMs are built for exactly
that regime: the decision boundary depends only on the support vectors near the margin, which makes
them resistant to overfitting when data is scarce, unlike a deep network or a large gradient-boosted
ensemble that would happily memorise 374 rows. The trained model is also tiny and its inference path
is a fixed number of kernel evaluations — a predictable, bounded cost, which matters on a device.
`C` and `gamma` are grid-searched over `{rbf, linear}`; on this data the **linear** kernel with
`C=0.1` wins, which is a useful signal in itself: the classes are close to linearly separable and
extra model capacity is not needed.

**Why `class_weight="balanced"`.** Only 20.9% of the dataset has apnea. Left alone, an SVM can score
79% accuracy by predicting "no apnea" for everyone — a useless screener. Balancing the class weights
makes each missed apnea case cost proportionally more during fitting.

**Why tune on F1, not accuracy or raw recall.** Accuracy rewards the degenerate all-negative model.
Pure recall rewards the opposite degenerate model — flag everyone, recall 1.0. F1 forces the grid
search to find settings that catch cases *without* flagging half the healthy population.

**Why calibrated probabilities.** A bare SVM emits a distance from the hyperplane, which is not
interpretable. Wrapping it in `CalibratedClassifierCV` (Platt scaling) turns that into an actual
probability, so the device can act on the number: stay silent at 5%, suggest a check-up at 60%,
recommend a sleep study at 85%. A hard yes/no throws that gradient away.

**Why a rule score alongside the model.** Two reasons. It is **explainable** — "you are obese, sleep
under 6 hours, and are hypertensive" is actionable in a way that "72.4%" is not. And it is a
**safety net**: when the model says no but the rules say high risk, the app says so explicitly
rather than silently returning a negative. A screening tool that can only fail silently is worse
than one that admits uncertainty.

**Why `Occupation` was dropped.** In this dataset 61 of 73 nurses have apnea, versus 17 of 301
everyone else. Any model given that column learns "nurse ⇒ apnea," which is a sampling artifact of
how the data was collected, not physiology. It would not generalise, and a wearable does not know
your job anyway. Dropping it costs about 0.01 F1 and buys a model that is actually about the body.

**Why blood pressure is optional in practice.** Removing the three BP-derived features costs only
~0.02 F1 (0.874 → 0.853), so a band with no cuff is still viable. It is kept in the default model
because many users do have a home BP monitor and it is a genuine apnea correlate.

---

## Performance

Held-out stratified test set (75 people, 16 with apnea):

| Metric | Score |
|---|---|
| Accuracy | 0.933 |
| Precision | 0.762 |
| **Recall** | **1.000** |
| F1 | 0.865 |
| ROC AUC | 0.980 |
| 5-fold CV F1 | 0.872 ± 0.032 |

```
Confusion matrix          predicted
                        No apnea  Apnea
        actual No apnea    54       5
               Apnea        0      16
```

Zero missed cases, five false alarms. **That is the correct error profile for screening**: telling
five healthy people to see a doctor is a minor cost, missing someone with apnea is not. The single
held-out split is small, so the cross-validated F1 is the more trustworthy number.

Runtime: 14 KB artifact, 81 support vectors, 0.9 ms to load, ~2 ms per prediction.

---

## What changed in this update

The previous version worked, but had a defect that made its accuracy number misleading.

**The model was trained on the wrong population.** In the raw CSV, healthy people have an *empty*
`Sleep Disorder` cell. The old preprocessing called `dropna(subset=["Sleep Disorder"])`, which
deleted all **219 healthy people** and kept only the 155 who already had a diagnosed disorder. The
model was not learning "apnea vs. healthy" — it was learning "apnea vs. insomnia," among people
already known to be ill. Deployed on a wearable, where most users are healthy, it would have faced a
population it had never seen. Treating the blanks as "no disorder" recovers all 374 rows and makes
the task the one we actually want to solve.

**BMI was encoded in the wrong order.** `LabelEncoder` assigns codes alphabetically, which produced
`Normal=0, Normal Weight=1, Obese=2, Overweight=3` — telling the model that obese is *less* severe
than overweight. The dataset also splits an identical category into `Normal` and `Normal Weight`.
Both are fixed by an explicit ordinal map (`Normal=0, Overweight=1, Obese=2`) with the duplicate
merged.

**`Occupation` was a shortcut feature.** Removed, for the reasons above.

Nested cross-validation across the changes, tuning inside each fold:

| Configuration | n | Recall | F1 | AUC |
|---|---|---|---|---|
| Original (apnea vs. insomnia only) | 155 | 0.869 | 0.859 | 0.929 |
| \+ healthy rows recovered | 374 | 0.855 | 0.844 | 0.934 |
| \+ `Occupation` dropped | 374 | 0.836 | 0.837 | 0.936 |
| \+ ordinal BMI | 374 | 0.846 | 0.844 | 0.936 |
| \+ F1 tuning, balanced weights | 374 | **0.901** | **0.872** | **0.939** |

The first row is a different, easier-to-look-good-on task, so its accuracy was never comparable —
but the final model beats it on recall, F1 and AUC *while* solving the harder and more realistic
problem.

Alongside the correctness work:

- **The model is saved to disk.** `app.py train` fits once and writes `models/apnea_svm.joblib`;
  the CLI and UI load it. Previously every single prediction re-ran the entire grid search.
- **One feature-engineering implementation.** The three derived features were previously computed
  in three separate places that had to be kept in sync by hand.
- **The CLI is usable.** `train` / `predict` subcommands, every input exposed, sensible defaults,
  named BMI categories instead of raw integers. Previously five of twelve inputs were adjustable and
  the rest were hardcoded constants.
- **`test.py` is an actual test.** It asserts dataset shape, guards against accuracy and recall
  regressions, and checks that an at-risk profile scores above a healthy one. It was previously a
  training script that only printed numbers.
- **Added `requirements.txt`**, renamed `src/evalaute.py` to `src/evaluate.py`, and reported
  precision/recall/F1/AUC instead of accuracy alone.

---

## Project structure

```
app.py                  CLI: train and predict subcommands
ui.py                   Streamlit interface
test.py                 Smoke test and regression guard
requirements.txt
src/
  data.py               Loading, feature construction, encoding, splitting
  model.py              SVM pipeline and hyperparameter grid
  evaluate.py           Metrics and formatting
  pipeline.py           Train/save/load/predict orchestration
  utils.py              Rule-based risk score
data/
  Sleep_health_and_lifestyle_dataset.csv
models/
  apnea_svm.joblib      Written by `app.py train`, not tracked in git
```

---

## Limitations

- **374 people is a small dataset**, and a synthetic one. The numbers above are honest for this data
  but should not be assumed to transfer to a real clinical population without revalidation.
- **The target is self-reported**, not confirmed by polysomnography.
- **`Sleep Duration` and `Quality of Sleep` are self-reported** in the source data. A device would
  substitute its own measurements, which are distributed differently — retraining on device-collected
  data would be necessary before deployment.
- **`Age` and `Physical Activity Level` carry no signal here** (zero permutation importance), which
  is implausible for real apnea and suggests the dataset's generation process is simplified.
- **Not a medical device.** No regulatory validation. Screening aid only.
