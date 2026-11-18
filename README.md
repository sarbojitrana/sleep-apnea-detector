# SleepSense · Combined IoT screening

One interface, six sensor parameters, optional sleep-duration context, and one **Predict** action.

| Field | Role |
| --- | --- |
| Average SpO₂ (%) | Trained sensor input |
| Lowest SpO₂ (%) | Trained sensor input |
| SpO₂ standard deviation (percentage points) | Trained sensor input |
| Heart rate (bpm) | Trained sensor input |
| Breathing rate (breaths/min) | Trained sensor input |
| Perfusion index (%) | Trained sensor input |
| Sleep duration (hours) | **Context only; not yet trained** |

The original four sensor quantities remain in one SVM. Lowest oxygen and oxygen variability preserve information lost by reducing the signal to an average. All six sensor summaries must cover the **same completed 60-second interval**. An IoT companion calculates the summary statistics from valid samples; a single awake spot reading is not equivalent.

Sleep duration is included as context at the user's explicit selection because the verified dataset does not contain actual sleep duration jointly with these sensors and apnea labels. It is saved with the result and explicitly excluded from the trained feature vector. No sleep value is invented or substituted with total recording length.

The main page contains parameter fields and Predict only. There are no model selectors, example toggles, histories, metrics panels, or download controls.

## Run

Tested on Linux with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python predictor.py train
streamlit run ui.py
```

Training downloads the pinned public Kaggle file if needed, verifies its SHA-256, and saves `models/predictor_svm.joblib` and `models/predictor_svm.report.json`. Training never occurs on application startup or a user request.

CLI:

```bash
python predictor.py predict --spo2 97 --spo2-min 95 --spo2-std 0.8 \
  --heart-rate 65 --respiratory-rate 16 --perfusion-index 3.5 \
  --sleep-duration 7
```

All six sensor parameters are required. Sleep duration is optional context. Missing or invalid readings are rejected; readings outside training support receive **Unable to assess** instead of a false negative.

Results are **positive or negative screening signals**, not confirmation or exclusion of sleep apnea.

## Performance improvement

Patient-separated nested development validation refits scaling, model selection, and threshold selection inside each outer training fold. The threshold targets 85% recall on out-of-fold development predictions; it is never selected using evaluation labels.

| Metric | Previous four-average model | Updated six-input model | Updated nested development |
| --- | ---: | ---: | ---: |
| Accuracy | 52.7% | **74.1%** | 67.5% |
| Sensitivity / recall | 20.8% | **89.6%** | 84.0% |
| Specificity | 73.0% | **64.2%** | 58.5% |
| Precision | 32.9% | **61.5%** | 52.4% |
| F1 | 0.254 | **0.730** | 0.646 |
| ROC AUC | 0.436 | **0.857** | 0.791 |

The fixed six-patient evaluation has 2,450 usable windows. The updated predictor missed **99/954 positive usable windows** and falsely flagged **535/1,496 negative usable windows**. Increased sensitivity comes with more false alarms.

Signal-quality coverage is **73.8%**. Another 872 windows, including 252 positive windows, were unusable. Across all positive windows including quality exclusions, 855/1,206 (70.9%) were flagged. Unusable data is insufficient evidence, never a negative result.

This cohort and evaluation split had already been examined in prior development. These are **exploratory research results, not new external or clinical validation**. Thousands of windows from six patients are not thousands of independent patients. The earlier lifestyle model's higher score used a different task/data and its original split shared duplicate profiles; that score does not establish clinical reliability.

See [the reliability update](docs/reliability-update.md) and [full measured report](docs/predictor-evaluation.json).

## Why sleep duration is not trained yet

[OSASUD's verified Kaggle mirror](https://www.kaggle.com/datasets/menes97/osasud-sleep-apnea-original-parquet) contains real matched oxygen, pulse, breathing-rate and perfusion measurements with respiratory-event annotations. It has no sleep-stage export or actual time-asleep column.

The larger Kaggle candidates inspected were synthetic, missing required measurements, or lacked verified clinical provenance. We do not generate oxygen/sleep values, join unrelated patients, use AHI/diagnosis as an input, or equate time recorded with time asleep to produce a higher accuracy number.

To learn sleep duration's contribution, obtain recordings with actual sleep duration and these sensor summaries measured for the same participants, with independent apnea labels. An input field alone cannot create that relationship.

## Dataset and device limitations

- Original data: [OSASUD publication](https://doi.org/10.1038/s41597-022-01272-y), [publisher's CC0 data](https://doi.org/10.6084/m9.figshare.19214289.v1).
- Pinned Kaggle scalar file: 6,338,669 bytes, SHA-256 `6a685bac8a65bd52f986ab8d66547f49d73082d7555250f4e54d69c00a8a44b9`.
- Only 30 stroke-unit patients; generalization to home screening and other populations is unestablished.
- The label is any annotated apnea/hypopnea second in the 60-second interval, including central/mixed events; it is not a person-level OSA diagnosis or severity.
- Clinical-monitor HR/RR scalars were measured using ECG-derived methods. No EEG/ECG waveform is an input. Optical pulse and wearable respiration estimates need validation against those reference quantities.
- Each source channel requires at least 90% valid samples and no missing run longer than five seconds. No normal values replace absent data.
- Mean, minimum and standard deviation are calculated only from valid oxygen readings in the current interval; no future readings or label fields enter features.
- Ancillary sensors retain a nonzero learned scale chosen by inner patient-group validation. More fields are not assumed to be more predictive.

Raw recordings and generated artifacts are excluded from Git. Only load trusted local model files; joblib/pickle can execute code. The web app accepts numeric parameters, never uploaded model artifacts.

## Device API

```bash
export APNEA_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uvicorn api:app --host 127.0.0.1 --port 8000 --no-access-log
```

Primary endpoint:

```http
POST /v3/screen
Authorization: Bearer <your key>
Content-Type: application/json

{
  "spo2": 97,
  "spo2_min": 95,
  "spo2_std": 0.8,
  "heart_rate": 65,
  "respiratory_rate": 16,
  "perfusion_index": 3.5,
  "sleep_duration": 7
}
```

The response includes `context` and `context_not_used_by_model` to preserve the sleep-duration distinction. `GET /v3/model` returns the model report. `GET /health` is public. Authentication, strict numeric validation, and a 16 KB body limit apply. Use `APNEA_PREDICTOR_PATH` for a custom artifact.

Historical `/v1` and `/v2` routes remain for research compatibility and return 503 when their optional artifacts are not installed. The default container installs only the current predictor, avoiding unnecessary baseline training.

## Verify and deploy

For the full historical regression suite:

```bash
python -m pip install -r requirements-dev.txt
python app.py train
python sensor.py download
python sensor.py train
python combined.py train
python predictor.py train
python test.py
```

Tests verify the six-feature contract, required measurements, genuinely separate sleep context, patient isolation, development-only operating threshold, shared API/UI inference, invalid readings, and compatibility with historical models.

The [Dockerfile](Dockerfile) explicitly trains the current predictor during build. Its unprivileged process uses the host-assigned port and a health check. [Render configuration](render.yaml) retains manual redeploys. Hosting is not medical validation.

Historical experiment reports remain in [docs](docs/reliability-update.md). They are not alternative modes in the main interface.
