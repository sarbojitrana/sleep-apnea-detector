# Reference audit and application improvements

Reviewed 18 November 2026. This records what could be verified, the changes that followed, and what remains untested. It is a reference base for the future PPT, not a claim that every requested full text was accessible.

## Requested references and access

| Requested work | Verified identity and access | Use in this project |
| --- | --- | --- |
| Sara Mohan George, *Sleep Apnea Detection Using Wearable Sensors* | The [publisher's contents](https://www.routledge.com/Advances-in-Wearables-and-Nearables-for-Healthcare-Applications/Naik/p/book/9781032796543) identify Chapter 1, *Sleep Apnea Detection Using Wearable Sensors: Principles, Methods, and Challenges*, in Ganesh R. Naik's edited book. The listing shows copyright 2027. Only the publisher listing was accessible; chapter methods/results were not reviewed. | Background reference for wearable acquisition and the distinction between screening and clinical validation. No chapter-specific accuracy or algorithm claims are adopted. |
| Subasri R and colleagues, *Real-Time Sleep Apnea Detection System Using Multi-Sensor Fusion and Embedded Machine Learning* | [IEEE DOI 10.1109/ICVADV67766.2026.11470342](https://doi.org/10.1109/ICVADV67766.2026.11470342), Priyadharshini S, Subasri R, Sindhukavi S, ICVADV 2026, pp. 1289–1293. Publisher-indexed abstract available; full text not accessible. | Abstract describes synchronized multisensor acquisition, Arduino integration, and SVM/RF methods. This motivates a quality-checked device-input path, not a claim that adding every mentioned sensor improves our classifier. |
| *Obstructive Sleep Apnea Prediction Using Machine Learning*, attributed in the request to Jiangfan Yu | The supplied DOI [10.22214/ijraset.2023.53137](https://doi.org/10.22214/ijraset.2023.53137) resolves to **Balaji Bodkhe, Mandar Phirke, Sankalp Oimbe, Jayesh Khairnar, Amol Kengar**, IJRASET 11(V), May 2023, pp. 6113–6123. The [publisher HTML](https://www.ijraset.com/research-paper/obstructive-sleep-apnea-prediction) and [full PDF](https://www.ijraset.com/best-journal/obstructive-sleep-apnea-prediction-using-machine-learning) were reviewed. Yu's authorship was not verified. | Discusses preprocessing, multiple classifiers, and multiple evaluation metrics. Its result table reports RF accuracy 91% and SVC 56.1%. Available cohort/split details do not support reproducing those scores on our data. We tested the RF idea independently. |

The user supplied these SciSpace records:

- [OSA prediction](https://scispace.com/papers/obstructive-sleep-apnea-prediction-using-machine-learning-3c4yxz8b)
- [Wearable sensors](https://scispace.com/papers/sleep-apnea-detection-using-wearable-sensors-yc8vb59nz1f4)
- [Multisensor embedded ML](https://scispace.com/papers/real-time-sleep-apnea-detection-system-using-multi-sensor-6uuz3h0knpt8)

These exact pages did not expose their text through available web access. No SciSpace integration was available, so its chat was not used. No AI-generated paper summary has been represented as a verified result. Publisher sources above supply the supported evidence. Correct the citation identities before using them in presentation slides.

## Reproducible experiment: Random Forest versus the current SVM

The RF comparison was fixed before evaluation: 100 trees, six inner-CV combinations of depth (4, 8, unrestricted) and minimum leaf size (5, 15), balanced subsample weights. Both methods use the same six real sensor summaries, patient split, outer and inner patient-group fold seeds, training sample cap, balanced-accuracy selection objective, and development-only 85% recall threshold objective.

Only the existing **24 development patients** were used. The six previously examined evaluation patients were excluded. The SVM comparison reuses its saved nested results after checking identical outer validation patients. No random row split, synthetic training data, new patient-joining, or threshold adjustment using evaluation labels was performed.

| Nested development metric | Current SVM | Random Forest |
| --- | ---: | ---: |
| Accuracy | **67.5%** | 62.3% |
| Sensitivity | 84.0% | 84.2% |
| Specificity | **58.5%** | 50.3% |
| Precision | **52.4%** | 48.0% |
| F1 | **0.646** | 0.612 |
| ROC AUC | **0.791** | 0.774 |

RF produced more false positives at essentially the same sensitivity, so it does not justify replacing the SVM. This is an exploratory development comparison, not proof of universal SVM superiority. The application's fitted SVM, threshold, and reported six-patient evaluation metrics remain unchanged: 74.1% accuracy, 89.6% sensitivity, 64.2% specificity on usable intervals.

The recorded estimator serializations were about 151 KB for SVM and 269 KB for RF. Median single-interval prediction time on the development computer was about 0.86 ms versus 4.43 ms. This excludes sensor acquisition and network latency, and is **not an Arduino/ESP32 benchmark**. Scikit-learn serialization size is not microcontroller runtime memory.

Reproduce with `python -m src.research_benchmark --report docs/research-benchmark.json`. The [machine-readable report](research-benchmark.json) includes folds, scores, parameters, threshold selection, timings, and the unchanged primary artifact hash.

## Implemented improvements

1. **Visible evidence:** metrics are read from the loaded artifact, with nested validation, confusion matrix, signal coverage, and false-positive/false-negative counts. Interval-level accuracy is never presented as a person's diagnostic certainty.
2. **Consistent sensor processing:** authenticated `POST /v3/window` accepts all four synchronized 1 Hz channels over 60 seconds, applies the existing per-channel quality policy, and computes the six SVM inputs. Missing samples are excluded, never filled with normal measurements. Any failed channel yields insufficient evidence, not a negative screen.
3. **Traceable demonstrations:** example/random buttons fill editable parameters without submitting. Random oxygen summaries derive from one synthetic signal, preserving consistency between its mean, minimum and variation. Demo inputs never affect measured performance.
4. **Session history:** up to 100 submissions retain entered values, timestamp, data origin, output and model ID, with CSV export and clear. History is session-only; it is not a clinical longitudinal record or a multi-user database.

The sensor-window endpoint reduces feature-entry mismatch. It does not establish a better measured accuracy or verify the device's timestamps, calibration, contact, or optical-vs-clinical-monitor transfer.

## Improvements requiring new evidence

Actual sleep duration, age/BMI, motion, snoring, temperature, and blood-pressure channels should affect a trained predictor only when jointly recorded with the required channels and independent reference labels. More fields alone do not establish a learned contribution. Sleep duration remains explicitly context only.

Longer-window desaturation dynamics, motion-based artifact rejection, and temporal SVM features are reasonable future experiments after defining the device sampling contract and obtaining suitable training data. The 60-second interval target is not AHI, overnight severity, or a confirmed person-level OSA diagnosis. External patient validation and actual device validation remain the highest priorities.

For a PPT, distinguish **reported literature results**, **our measured SVM results**, **our RF development comparison**, and **future work**. Cite corrected authors/DOIs, state the 30-patient source population, and disclose that usable-window coverage is 73.8%.
