# Duration context and the improved combined predictor

The six-input predictor introduced in this update retains the four existing sensor quantities. The later [research/UI update](research-review.md) restores metrics, demo controls, and history without changing its fitted model. Two jointly recorded oxygen summaries were added because a mean alone discards desaturation dynamics:

- Average SpO₂, lowest SpO₂, and SpO₂ standard deviation over the same 60 seconds.
- Mean heart rate, breathing rate, and perfusion index over those 60 seconds.
- Sleep duration in hours, **context only**.

The user explicitly selected context-only handling for sleep duration after the joint-data limitation was explained. Sleep duration is returned with the result but excluded from the estimator. Changing this context value cannot change the classifier score. It is not filled from recording length or copied from an unrelated person's lifestyle row.

## Dataset audit

OSASUD has timestamps and full recording durations, but no sleep-stage annotations or actual time-asleep variable in the published export. Recording time includes wake and is not total sleep time.

The broader Kaggle search did not produce a verified cohort containing actual sleep duration, all four existing sensor values, and reliable apnea labels together:

| Candidate | Finding |
| --- | --- |
| [OSASUD original Parquet](https://www.kaggle.com/datasets/menes97/osasud-sleep-apnea-original-parquet) | Matched sensor values and respiratory-event labels; no true sleep duration |
| [Extended lifestyle data](https://www.kaggle.com/datasets/fortressenebe/sleep-health-and-lifestyle-dataset-extended) | Author explicitly describes synthetic augmentation; lacks matched four-channel sensor recordings |
| [Sleep Health & Daily Performance](https://www.kaggle.com/datasets/mohankrishnathalla/sleep-health-and-daily-performance-dataset) | Author describes 100,000 synthetic records; benchmark accuracy would not establish clinical reliability |
| [30,000-row lifestyle dataset](https://www.kaggle.com/datasets/gowtha69/sleep-health-and-lifestylesleep-disorder-dataset) | No verified clinical acquisition supplied and no matched PI/respiratory-rate columns documented; diagnostic RDI cannot substitute for respiratory rate |
| [NSRR learning sample](https://www.kaggle.com/datasets/rishitjakharia/nsrr-dataset) | Only three example sleep studies; not enough independent subjects for the required validation |

Larger clinical cohorts such as [SHHS](https://sleepdata.org/datasets/shhs) and [CPS](https://physionet.org/content/cps-dataset-sleep/1.0.0/) have access requirements. Their measurement availability and intended-use terms must be checked before assuming they contain every requested quantity. No restricted source was accessed through an unverified mirror.

The historical lifestyle model's higher accuracy does not demonstrate real-world reliability. Its original random split put identical input profiles in training and test. Adding generated columns or using clinical diagnostic indices as predictors would not repair the evidence.

## Development-only model and threshold selection

1. Keep the same predeclared patient split for exploratory comparison; do not move patients to improve the result.
2. Fit six sensor summaries from the same recording interval; never use patient identity, event labels, age/BMI guesses, or future intervals as predictors.
3. Refit scaling within each fold. Inner patient-group CV selects RBF SVM C and a nonzero ancillary-sensor scale by balanced accuracy. The scale can downweight patient-specific pulse, respiratory and perfusion baselines while retaining all original inputs.
4. Select the operating threshold from positive-class **out-of-fold development scores**, targeting 85% development recall. This is a selection objective, not a guaranteed sensitivity.
5. In every outer development fold, rerun hyperparameter selection and threshold selection using only that fold's training patients.
6. Fit the final estimator and threshold on development patients, then score the fixed six-patient evaluation set.
7. Compare the six-input model with a four-average model retrained under the same new protocol.

The evaluation cohort had already been inspected during earlier work, so these results remain exploratory. The next confirmation must use previously unseen people and actual device measurements. No new external clinical validation was performed.

## Measured results

| Metric | Previous four-average model | Updated six-input model | Updated nested development |
| --- | ---: | ---: | ---: |
| Accuracy | 52.7% | **74.1%** | 67.5% |
| Sensitivity | 20.8% | **89.6%** | 84.0% |
| Specificity | 73.0% | **64.2%** | 58.5% |
| Precision | 32.9% | **61.5%** | 52.4% |
| F1 | 0.254 | **0.730** | 0.646 |
| ROC AUC | 0.436 | **0.857** | 0.791 |

Evaluation matrix for the updated predictor (actual rows / predicted columns, negative then positive):

```text
[[961, 535],
 [ 99, 855]]
```

There are 2,450 eligible windows from six patients. The model missed 99/954 eligible positive windows and falsely flagged 535/1,496 eligible negative windows. More sensitivity comes with more false alarms than the old zero-margin operating point.

Signal-quality coverage is unchanged at **73.8%**. Another 872 windows failed quality requirements, including 252 positive windows. Across all 1,206 positive evaluation windows, including quality exclusions, 855 (70.9%) were flagged. An excluded window is insufficient evidence, never a negative result.

The control with only four averages, retrained using the new procedure, reached 99.1% sensitivity but only 11.5% specificity and 45.6% accuracy. This shows why raising recall alone is not enough. Oxygen dynamics retain substantially more discrimination at a sensitive operating point.

All results, folds, thresholds, selected parameters and quality coverage are in [predictor-evaluation.json](predictor-evaluation.json). Windows from the same patient are correlated; they are not thousands of independent validation subjects.

## Interface and use

The form has seven fields and one Predict action. The subsequent interface update adds example/random prefill buttons, measured model performance, and session history with CSV export. A short screening limitation remains. The metrics are read from the loaded model's report; demo data is never included in them.

```bash
python predictor.py train --report docs/predictor-evaluation.json
streamlit run ui.py
```

Devices compute the six summaries over the same 60-second interval, with the existing quality checks. Sleep duration is optional context and is explicitly labelled as not trained. The supported primary API is `POST /v3/screen`; older endpoints remain available only when their historical artifacts are installed.

This is improved research performance, not established clinical reliability. The current task is interval-level breathing-event screening, not confirming a person's diagnosis. Clinical-monitor HR/RR and optical/wearable device estimates are not automatically interchangeable.
