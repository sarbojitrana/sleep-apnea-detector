# Sensor dataset selection and measured experiment

Selected **[OSASUD Sleep Apnea – Original (Parquet)](https://www.kaggle.com/datasets/menes97/osasud-sleep-apnea-original-parquet), version 1** from Kaggle. It is a format-converted mirror of the published [OSASUD clinical dataset](https://doi.org/10.1038/s41597-022-01272-y), with [original data on Figshare](https://doi.org/10.6084/m9.figshare.19214289.v1).

The original publisher's API confirms **CC0** and the original file MD5 stated by the mirror's converter. [Source metadata](datasets/osasud-source.json) records the URLs, license, author attribution, pinned Parquet SHA-256, and verification limits. The original pickle conversion has not been independently repeated; no downloaded pickle is executed.

## Why this candidate

| Candidate | Decision |
| --- | --- |
| OSASUD original Parquet mirror | Selected: real overnight recordings, SpO₂, preserved patient/time keys, expert respiratory-event labels, traceable clinical source |
| Sleep Disordered Breathing Detection (ziya07) | Not selected for this experiment: tabular diagnostic fields including AHI and ECG-derived measurements; insufficient acquisition detail for a sensor-window experiment. Feeding AHI into an apnea classifier would undermine the screening task. |
| Sleep Apnea (neha2611) | File listing contains a single a01 recording and annotations; cannot provide the needed independent-patient evaluation; license is unspecified |
| Existing Sleep Health and Lifestyle CSV | Retained as the separate lifestyle baseline; it has no overnight oxygen measurements |

The selected scalar file is only **6,338,669 bytes**, rather than the full approximately 400 MB waveform mirror. It contains **961,357 one-second rows, 30 patients, and approximately 267 hours**. Labels include normal breathing, hypopnea, and obstructive, central, and mixed apnea.

There are 13.8% missing oxygen readings. Patient/timestamp pairs are unique and each patient's timestamps are contiguous at 1 Hz. Recorded nonmissing SpO₂ spans 60–100%. The mirror's scalar values and label counts agree with its supplied conversion report.

**Important sensor distinction:** OSASUD's HR, respiratory-rate, and PVC columns are ECG-derived. The new model reads only the SpO₂ signal, patient/time identifiers, and event labels. Patient IDs are used only for splitting; events only for targets. ECG/EEG, AHI, the anomaly target alias, and other PSG signals are never predictors. Future PPG or movement features need a separately documented extraction and comparison.

## Reproduce

```bash
python sensor.py download
python sensor.py train --report docs/osasud-evaluation.json
python test.py
```

The public Kaggle download requires no credential and pins version 1 plus its exact SHA-256. Raw data lives under ignored `data/osasud/`; model artifacts live under ignored `models/`. The tracked [evaluation report](osasud-evaluation.json) is an aggregate research result.

## Predeclared experiment

- **Target:** classify whether a completed, nonoverlapping 60-second window contains at least one expert-annotated apnea/hypopnea second. This is not person-level OSA diagnosis, event counting, AHI estimation, or forecasting.
- **Inputs:** nine SpO₂ summaries: mean, minimum, standard deviation, 10th/90th percentiles, fraction below 90%, range, mean adjacent absolute change, and linear trend. All use only the current completed window.
- **Quality:** at least 54/60 samples must be finite percentages in (0,100], with no missing run longer than five seconds. Do not impute missing readings as normal oxygen. Unusable windows receive insufficient evidence.
- **Split:** fixed random seed 42; 24 development patients and six held-out patients. The same patients never appear in development and test.
- **Tuning:** nested three-fold patient-separated CV on development data, class-balanced SVM, F2 selection, fixed zero-margin decision. Standardization is fitted inside every fold.
- **Compute bound:** up to 150 evenly spaced eligible training windows per patient, chosen without labels; final fit uses 3,568 windows. All eligible validation/test windows are scored.
- **Ablation:** independently tune a mean-SpO₂-only baseline and the nine-feature model under the same protocol. The main artifact is the predeclared nine-feature model; do not select or retune it using holdout scores.

## Results

| Metric | Mean-SpO₂-only holdout | Oxygen-pattern holdout | Oxygen-pattern nested development |
| --- | ---: | ---: | ---: |
| Accuracy | 71.3% | **79.4%** | 75.3% |
| Sensitivity / recall | 63.0% | **77.9%** | 73.7% |
| Specificity | 76.3% | **80.3%** | 76.2% |
| Precision | 61.9% | **70.7%** | 64.7% |
| F1 | 0.625 | **0.741** | 0.689 |
| ROC AUC | 0.728 | **0.869** | 0.817 |
| Average precision | 0.569 | **0.771** | 0.716 |

Oxygen-pattern holdout matrix, actual rows / predicted columns, negative then positive:

```text
[[1280, 315],
 [ 215, 759]]
```

These scores describe **2,569 usable windows from six patients**, not 2,569 independent patients. They are conditional on passing the signal-quality policy.

**Coverage matters:** 753 of 3,322 complete holdout windows (22.7%) were unusable. Those include 232 annotated positive windows. Of all 1,206 positive holdout windows including unusable ones, 759 (62.9%) were flagged. Abstention is reported separately and must never be displayed as a negative prediction.

The actual improvement supported by this experiment is over the mean-oxygen baseline on the **same dataset and held-out patients**: about +8.1 percentage points in accuracy and +14.9 points in sensitivity. This supports extracting oxygen dynamics, with substantial errors still present.

The lifestyle SVM's 90.5% holdout accuracy cannot be compared as a ranking: it predicts different labels on a different dataset at a different unit of observation. The new dataset is a more appropriate starting point for an oxygen-sensing device, not evidence that the device is ready for clinical use.

## Application and device format

The dashboard adds an **Overnight SpO₂ · research** tab, while the existing lifestyle model remains available. Both artifacts are separate. The container explicitly downloads the pinned scalar file and trains both models during build; neither model trains on a user request.

Upload a CSV with exactly these columns:

```csv
elapsed_seconds,spo2
0,97
1,97
2,
3,96
```

A real recording needs at least 60 rows at exactly 1 Hz, up to 12 hours and 1 MB. Keep a timestamp row with blank SpO₂ for each missing sample. Gaps, duplicate timestamps, invented resampling, and raw sensor ADC values are not equivalent to these inputs.

```bash
python sensor.py predict --input your_1hz_oxygen.csv
```

Output contains per-window signals, quality, abstentions, and unscored trailing samples. A flagged window is not an apnea count. The model's decision margin is not a probability or clinical confidence.

The authenticated gateway API also supports `POST /v1/oximetry/screen` with `{"spo2_samples": [...]}` containing exactly 60 consecutive one-second readings, using JSON `null` for missing values. `GET /v1/oximetry/model` returns the research report. Both require the existing bearer key. If the sensor artifact is absent, these routes return 503 and the lifestyle service remains available. Configure a custom artifact path with `APNEA_SENSOR_MODEL_PATH`.

## Limits and next validation

Only 30 stroke-unit patients are represented. Normal windows from these patients do not constitute a healthy-population control cohort. Clinical-monitor oxygen estimates can differ from a low-cost home sensor, including contact, motion, and pigmentation-related measurement effects. Validate actual device signals and evaluate on a new population before making screening claims.

Do not join unrelated Kaggle datasets by row or label, generate oxygen values for the old lifestyle rows, or include diagnostic labels/indices as predictors. Additional channels must be aligned recordings from the same patients.
