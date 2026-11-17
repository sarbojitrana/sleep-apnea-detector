# SleepSense · IoT sleep apnea screening

An SVM-based **first-check research application** using measurements that can be entered at home or obtained from everyday sensors. The intended device is an affordable screening aid that encourages appropriate follow-up.

**New: overnight SpO₂ research mode.** A separate sensor SVM uses optical oxygen measurements from a verified Kaggle mirror of the published OSASUD dataset: 30 patients and approximately 267 hours. It uses no EEG/ECG inputs. On six held-out patients, it achieves 79.4% accuracy and 77.9% sensitivity on usable 60-second windows, with 77.3% coverage. This is a different task from the lifestyle model below. See [dataset selection and measured results](docs/dataset-selection.md) and the [full sensor evaluation](docs/osasud-evaluation.json).

```bash
python sensor.py download
python sensor.py train
# Then use the Overnight SpO₂ · research tab in the dashboard.
```

The lifestyle screening mode uses **six inputs**: age, the dataset's gender category, BMI category, sleep duration, resting pulse, and daily steps. It requires no EEG, ECG, blood-pressure cuff, stress rating, or subjective sleep-quality score. The independent oxygen mode uses only SpO₂ readings.

**Current capability:** screening for patterns associated with the *Sleep Apnea* label in the bundled Kaggle lifestyle dataset. The dataset contains no overnight waveforms, SpO₂, airflow, or event timestamps. This application cannot detect individual breathing pauses, compute AHI, diagnose apnea, or establish clinical severity. Real device screening needs separately validated data.

## Run locally

Python 3.12 is the tested environment. From the project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python app.py train
streamlit run ui.py
```

If using uv instead of pip:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.lock
.venv/bin/python app.py train
.venv/bin/streamlit run ui.py
```

Training is explicit. The dashboard and API **never train on startup**. Lifestyle training writes a versioned model to `models/apnea_svm.joblib` and a readable report to `models/apnea_svm.report.json`. Sensor training writes its separate `models/oximetry_svm.joblib` and report. Artifacts are local generated files excluded from Git. Old fourteen-feature artifacts must be retrained.

`requirements.txt` declares supported dependency ranges; `requirements.lock` pins the tested production environment, including transitive dependencies. The lock was generated on Linux with Python 3.12; regenerate and retest for another platform. Saved models require the exact scikit-learn version that trained them.

## Dashboard

- **Screening:** enter all six inputs, optionally report symptoms, and download a screening report. Example values are explicitly labelled.
- **Device import:** CSV or JSON, up to 100 daily summaries and 1 MB. Invalid rows receive individual errors. Download all results.
- **Session history:** the last 20 manual screenings, including whether example data was used; clear or download them.
- **Model & evidence:** source data audit, training support, held-out and nested validation metrics, confusion matrix, and full model report.

The app does not persist submitted readings or results to disk or a database. Session history lives in server memory while the session exists; exported downloads are controlled by the user. Hosted operation would transmit inputs to the host and needs its own privacy and access controls.

## Input contract

Every measurement is required; missing values are **never replaced by example values**.

| JSON / CSV field | Meaning | Accepted input |
| --- | --- | --- |
| `gender` | Source dataset category, entered in profile | `Female` or `Male` |
| `age` | Age in years | Whole number, 18–100 |
| `bmi` | Category based on measured height/weight | `Normal`, `Overweight`, `Obese` |
| `sleep_duration` | Main sleep period, hours/night | Finite number, 0.5–16 |
| `heart_rate` | Resting pulse measured optically or manually, bpm | Finite number, 30–220 |
| `daily_steps` | Full day's step total | Whole number, 0–100000 |

These broad bounds validate format and scope; they are not clinical normal ranges. The source does not document BMI category cutoffs. Do not invent numeric category cutoffs when converting source data.

Three optional Boolean fields—`loud_snoring`, `witnessed_pauses`, and `daytime_sleepiness`—prompt follow-up independently of the SVM. They are not trained features or a validated questionnaire score. Unchecked/omitted symptoms are not evidence of absence. CSV booleans use `true` or `false`.

Device input must be an **already processed daily summary**, not a stream of raw optical or accelerometer samples. Sleep duration estimation and resting-pulse quality control are responsibilities of the device or companion software. Source measurements and sensor estimates may have different distributions.

See [example JSON](examples/daily_summary.json) and [example CSV](examples/daily_summaries.csv). The CSV deliberately includes a profile outside training support to demonstrate abstention.

## Command line

```bash
python app.py predict --gender Female --age 40 --sleep 7 \
  --bmi Normal --heart 70 --steps 7000 --json

python app.py predict --gender Female --age 58 --sleep 6 \
  --bmi Overweight --heart 75 --steps 7000 --witnessed-pauses

python app.py batch --input examples/daily_summaries.csv
python app.py batch --input examples/daily_summaries.csv --output results.json
```

Use all six prediction flags. Optional symptom flags use hyphens. CLI errors exit with code 2; per-row batch failures are recorded as `invalid_input`.

## Interpreting a result

| Status | Meaning |
| --- | --- |
| `raised_signal` | SVM margin is at or above zero; follow-up is prompted |
| `lower_signal` | SVM margin is below zero; apnea is **not** ruled out |
| `insufficient_evidence` | A valid input falls outside the training support; the model withholds classification |
| `invalid_input` | An imported row is missing, malformed, or unsupported |

`screen_positive` is a model screening flag, **not disease presence**. It is `null` when the model abstains. `decision_margin` is a signed SVM score, **not a percentage probability, confidence, or severity**. It is also null when abstaining.

The bundled training data covers only ages **27–59**, sleep duration **5.8–8.5 hours**, pulse **65–86 bpm**, and steps **3000–10000/day**. The exact support is saved from development data in the artifact. An older person or a shorter night may be physiologically valid while still being unsupported by this dataset. In-range values do not prove a profile is represented.

Any reported symptom prompts follow-up even if the SVM signal is lower or unavailable. Discuss concerns with a clinician regardless of model output. This application is not an emergency monitor.

## What changed and why

The original app used fourteen features, filled missing readings with invented defaults, displayed an arbitrary weighted score as severity, and called the positive-class probability “confidence”. It split repeated profiles across training and testing.

The updated application:

1. Uses six accessible inputs, with one shared transformation and strict validation across all entry points.
2. Preserves the literal `None` dataset label. Blank or unknown targets raise errors; `Insomnia` belongs to the **other-label** class, not a “healthy” class.
3. Keeps identical six-feature profiles together, including conflicting labels, in both holdout and CV.
4. Fits `StandardScaler → SVC` inside every fold. Tunes linear/RBF kernels on **F2**, which weights recall more heavily than F1; uses balanced class weights.
5. Retunes inside each outer development fold for nested CV, while keeping a fixed holdout unseen by all tuning.
6. Uses a fixed zero-margin boundary for evaluation and serving. No unsupported clinical probability or severity is displayed.
7. Abstains outside training support and keeps symptom prompts independent.
8. Saves version, feature schema, dataset checksum, support bounds, model ID, and evaluation with the estimator. Writes artifacts atomically.
9. Adds bounded batch imports, a tested authenticated device API, and dashboard reports.

Repeated rows still contribute to fitting and row-level metrics. Grouping prevents exact-profile leakage but does not remove near-duplicate or source-generation bias. With this dataset, group isolation is a minimum safeguard, not independent clinical validation.

## Measured dataset performance

One run in the pinned environment uses 300 development rows and a **fixed holdout of 74 rows representing 22 distinct profiles**. It does not select a favourable split.

| Metric | Grouped holdout | Nested development CV, pooled |
| --- | ---: | ---: |
| Sensitivity / recall | 1.000 | 0.857 |
| Specificity | 0.881 | 0.920 |
| Precision | 0.682 | 0.740 |
| F1 | 0.811 | 0.794 |
| F2 | 0.915 | 0.831 |
| ROC AUC | 0.981 | 0.916 |
| Average precision (stepwise PR summary) | 0.872 | 0.794 |

Holdout matrix, actual rows / predicted columns, ordered other label then apnea label:

```text
[[52, 7],
 [ 0,15]]
```

The holdout caught 15/15 apnea-labelled rows with seven false flags. A descriptive row-level 95% sensitivity interval is approximately **79.6%–100%**, and duplicated rows further reduce independence. Nested development predictions missed **9 of 63** positive rows. Neither result establishes clinical sensitivity.

An all-negative classifier achieves about 79.7% accuracy on this holdout while detecting no positive rows. An all-positive classifier has perfect recall and only about 20.3% precision. Both baselines are included in the generated report.

The data contains **374 rows, 106 distinct six-input profiles, and 16 profiles with conflicting binary labels**. This sets a practical limit on what the available inputs can distinguish. Do not tune against the holdout after seeing its score. Regenerate the report whenever data, code, or dependency versions change.

## IoT architecture

```text
Optical pulse sensor → validated resting pulse ───────┐
Motion sensor / phone → daily steps, sleep estimate ─┤
One-time profile → age, gender category, BMI ─────────┤
                                                    ▼
                      daily JSON summary → Python gateway / local app
                                           validation → saved SVM
                                           screening + symptom follow-up
```

The Python implementation can run on a laptop, home gateway, or suitable Linux single-board computer. It does not run scikit-learn on an ESP32/Arduino. A microcontroller can send summaries to the gateway. Running the SVM directly on a microcontroller requires a separately implemented and numerically verified export of the scaler and kernel.

Pulse is obtained from **PPG (optical sensing)**, not ECG. A sensor module's ability to produce a heart-rate or oxygen estimate does not make those estimates clinically validated. SpO₂ may be useful in a future overnight-signal project, but it is **not present in the current dataset and is rejected as an unknown model input**.

See [the device and validation roadmap](docs/iot-roadmap.md) for the next stages.

## Local device API

The optional gateway API loads the saved artifact once and stores no requests. Start it locally:

```bash
export APNEA_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uvicorn api:app --host 127.0.0.1 --port 8000 --no-access-log
```

Keep the secret in your device/gateway configuration. Requests:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/model \
  -H "Authorization: Bearer $APNEA_API_KEY"
curl http://127.0.0.1:8000/v1/screen \
  -H "Authorization: Bearer $APNEA_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @examples/daily_summary.json
```

Interactive API docs: `http://127.0.0.1:8000/docs`. Screening/model requests require a bearer key of at least 24 characters. `/health` is public. Invalid schema returns 422, wrong content type 415, oversized screening body 413, and missing/invalid credentials 401. Maximum body size is 16 KB. Model path can be configured with `APNEA_MODEL_PATH`.

This is a **local/gateway API foundation**, not an internet deployment. Before public hosting, add HTTPS, request/time/rate limits at a reverse proxy, secret rotation, access controls for the dashboard, and an appropriate privacy/retention policy. The code does not add a database, alerts, or automatic deployment.

A [Render Blueprint](render.yaml) and [Dockerfile](Dockerfile) configure the dashboard deployment. The image installs the pinned environment and explicitly trains during the build; runtime only loads the artifact. The default mode serves Streamlit on the host-assigned `PORT`; `APP_MODE=api` serves the authenticated device API instead. See [deployment instructions](docs/deployment.md) for the Render settings, local container commands, and optional API service. Automatic redeploys are disabled.

## Verify

```bash
python -m pip install -r requirements-dev.txt
python app.py train
python test.py
# Equivalent:
python -m unittest discover -s tests -v
```

Tests cover preprocessing parity, missing/invalid measurements, literal labels, disjoint groups, artifact compatibility, abstention, symptom follow-up, batch imports, API authentication/validation, and dashboard interactions. UI tests need a trained artifact and are skipped if it is absent. CI explicitly trains before running them.

## Evidence and limitations

The dataset is from [Kaggle's Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset). Verified acquisition methods, source population, and independent clinical label confirmation are not bundled; do not assume these are confirmed diagnoses or independent real patients. Dataset licensing and clinical-use suitability must be checked before redistributing or deploying derived artifacts.

Clinical evaluation can include clinician-directed **home sleep apnea testing**, not only laboratory polysomnography. The [AASM diagnostic guideline](https://aasm.org/resources/clinicalguidelines/diagnostic-testing-osa.pdf) states that prediction tools alone do not establish an OSA diagnosis. The [FDA explains pulse-oximeter limitations](https://www.fda.gov/consumers/consumer-updates/pulse-oximeter-basics), including measurement inaccuracies under some conditions.

Actual device validation requires diverse recordings paired with appropriate reference labels, subject-separated evaluation, sensor-quality checks, and prospective evaluation of missed cases and false alarms. A lifestyle SVM cannot substitute for that evidence.

Only load model artifacts you created or trust: joblib/pickle files can execute code. The app accepts readings, never uploaded model files.
