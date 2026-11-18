# SleepSense · Combined IoT sleep apnea screening

**One form → one SVM → one screening result.**

Enter these four jointly recorded measurements:

| Input | Unit | Possible IoT source |
| --- | --- | --- |
| SpO₂ | % | Optical pulse oximeter |
| Heart rate | bpm | Optical pulse sensor |
| Breathing rate | breaths/min | Suitable wearable respiratory sensor |
| Perfusion index | % | Optical pulse oximeter / validated PPG processing |

Use **averages from the same completed 60-second measurement period**, ideally recorded during sleep. All four values are required and enter the same scaler and SVM. No separate oxygen-only mode or lifestyle model selection is presented in the main app.

The result is **Positive screening result** or **Negative screening result**. Missing/invalid measurements are rejected, and unsupported values produce **Unable to assess**. A positive screen is not a diagnosis, and a negative screen does not rule out sleep apnea.

**Current limitation:** the combined model is not reliable for health decisions. Its exploratory six-patient evaluation produced 52.7% accuracy and 20.8% sensitivity. The application displays this limitation prominently. Combining more scalar measurements did not improve performance on the available dataset.

## Run

Tested with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python combined.py train
streamlit run ui.py
```

Training downloads the pinned 6.4 MB public Kaggle file if needed, verifies its SHA-256, and writes `models/combined_svm.joblib` plus a readable report. No training runs when a user submits measurements.

CLI prediction:

```bash
python combined.py predict --spo2 97 --heart-rate 65 \
  --respiratory-rate 16 --perfusion-index 3.5
```

The main UI contains four numeric fields and a **Predict** button. Example measurements are explicitly labelled. Reports, measured performance, and session history are available in expanders. Submitted measurements are kept in server-side session memory; the application does not write them to a database.

## Dataset and measurement provenance

The model uses [OSASUD on Kaggle](https://www.kaggle.com/datasets/menes97/osasud-sleep-apnea-original-parquet), a format-converted mirror of the [published OSASUD dataset](https://doi.org/10.1038/s41597-022-01272-y). The [original publisher's data](https://doi.org/10.6084/m9.figshare.19214289.v1) is CC0.

SpO₂, HR, respiratory rate, and perfusion index are recorded together, with patient IDs, timestamps, and expert respiratory-event annotations. The training code never joins unrelated lifestyle rows or invents missing sensor measurements.

The **physiological quantities** are available to IoT devices, but the source HR/RR scalars were measured by clinical monitors using ECG-derived methods. No EEG or ECG waveform is a model input. Porting measurement methods to optical pulse and wearable respiration requires validation on the eventual device.

Age, BMI, steps, and sleep duration are omitted because they are not recorded jointly in this dataset; the user selected the supported joint sensor inputs for this combined form.

## What the model actually predicts

The training label means **at least one annotated apnea/hypopnea second occurred in the same 60-second interval**. It includes obstructive, central, mixed apnea, and hypopnea. It does not establish person-level OSA diagnosis, count events, calculate AHI, estimate severity, or predict future events.

A spot reading while awake is not equivalent to the training interval and cannot establish normal breathing throughout a night.

Each training channel needs at least 90% valid one-second samples, with no missing run longer than five seconds. No normal values are substituted for missing readings. For device integration, compute all four means from the same usable interval.

## Measured performance

The fixed split has 24 development patients and six evaluation patients. Tuning and nested CV keep patients separate and refit scaling inside folds.

| Metric | Combined model | SpO₂-mean-only control on the same usable windows |
| --- | ---: | ---: |
| Accuracy | 52.7% | 70.4% |
| Sensitivity | 20.8% | 63.6% |
| Specificity | 73.0% | 74.8% |
| Precision | 32.9% | 61.7% |
| F1 | 0.254 | 0.626 |
| ROC AUC | 0.436 | 0.723 |

There are 2,450 usable evaluation windows, with 73.8% coverage. The combined model missed 756 of 954 positive usable windows. An additional 252 positive windows failed quality requirements. These observations are clustered within six patients, not thousands of independent patients.

This cohort and split were previously inspected in the oxygen-only study. The combined model's evaluation is **exploratory**, not new external validation. Nested development sensitivity was also low at 25.0%, so the problem is not confined to one evaluation split.

The full measured report is [docs/combined-evaluation.json](docs/combined-evaluation.json). There is no demonstrated performance improvement from combining these four scalar averages. Better matched training data and/or richer overnight measurements are necessary before making a dependable screening claim.

The previous lifestyle SVM and oxygen-pattern SVM remain available as historical research baselines in code. Their prediction tasks, inputs, and sample units differ from this simple form. Their higher percentages are not valid performance claims for the combined predictor.

## Device API

```bash
export APNEA_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uvicorn api:app --host 127.0.0.1 --port 8000 --no-access-log
```

The main combined predictor is:

```http
POST /v2/screen
Authorization: Bearer <your key>
Content-Type: application/json

{
  "spo2": 97,
  "heart_rate": 65,
  "respiratory_rate": 16,
  "perfusion_index": 3.5
}
```

`GET /v2/model` returns its report. `GET /health` is public. Strict schema validation and the 16 KB request limit apply. Use `APNEA_COMBINED_MODEL_PATH` to override the artifact location.

The legacy `/v1/screen` lifestyle and `/v1/oximetry/screen` research routes remain for compatibility, independent of the main form. They return 503 if their optional artifacts are absent.

## Verify and deploy

```bash
python -m pip install -r requirements-dev.txt
# Historical artifacts are needed for the complete research regression suite:
python app.py train
python sensor.py download
python sensor.py train
python combined.py train
python test.py
```

Tests cover the joint input contract, shared training/inference transformation, every input reaching the same estimator, invalid measurements, sampling quality, API authentication, and the single-form UI, alongside the historical model checks.

The [Dockerfile](Dockerfile) trains artifacts explicitly at build time and serves the combined form by default. The [Render Blueprint](render.yaml) retains manual redeploys. Hosting is not clinical validation. See [deployment notes](docs/deployment.md).

## Research history

- [Oxygen-only dataset selection and experiment](docs/dataset-selection.md)
- [Original oxygen-pattern report](docs/osasud-evaluation.json)
- [Source verification and license](docs/datasets/osasud-source.json)
- [IoT validation roadmap](docs/iot-roadmap.md)

Raw Kaggle files and generated model artifacts remain excluded from Git. Only load model files you trust; joblib/pickle can execute code. The app never accepts uploaded model files.
