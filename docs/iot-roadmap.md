# From prototype to an accessible screening device

## Current build

The main application uses one form and one SVM with four jointly recorded quantities. It predicts an annotated breathing-event pattern in a completed measurement interval. Its measured sensitivity is poor, so it demonstrates the workflow rather than providing reliable health screening. It has no confirmed person-level diagnosis or emergency alarm.

| Input | Accessible source | Important boundary |
| --- | --- | --- |
| SpO₂ | Optical pulse oximeter | Validate calibration, contact and averaging |
| Heart rate | Optical pulse sensor | Source monitor used ECG-derived HR; device transfer needs validation |
| Breathing rate | Wearable respiratory sensor | Source monitor used ECG-derived RR; estimates are not automatically interchangeable |
| Perfusion index | Optical pulse oximeter / PPG processing | Use the reported percentage, not raw ADC values |

No EEG or ECG waveform is required as an input. All four quantities must describe the same 60-second period. Missing values are never replaced by normal defaults. The earlier lifestyle and oxygen-pattern models remain historical research baselines.

## Stage 1 · Connect the combined-measurement device

1. Choose a microcontroller that can transmit JSON, and an optical pulse sensor with a documented sampling interface.
2. Obtain reliable oxygen, pulse, breathing-rate, and perfusion measurements, including quality checks. These extraction/calibration algorithms are outside this repository.
3. Average all four quantities over the same completed 60-second interval.
4. Send `examples/combined_measurements.json` format to the authenticated `/v2/screen` endpoint over a trusted connection.
5. Handle `insufficient_evidence` and `null` screening flags explicitly. An unsupported or missing input is not a negative result.
6. Display the next step and model ID; preserve the distinction between example data and measured data.

Use the combined JSON example to exercise the interface before connecting hardware. A single spot measurement while awake is not equivalent to overnight breathing measurements.

A Python gateway allows offline operation on the local network. Remote hosting introduces connectivity and privacy requirements. An ESP32/Arduino cannot execute this scikit-learn artifact directly.

## Stage 2 · Obtain appropriate training evidence

Before offering real physiological screening, obtain a dataset containing inexpensive sensor channels actually available on the chosen device. Match each recording with independently verified reference labels and documented sensor units, sample rate, timing, and subject identifiers.

The current Kaggle OSASUD mirror contains joint clinical-monitor measurements and respiratory-event annotations, but only 30 stroke-unit patients. The four-average combined model did not generalize well. Future data should pair the actual IoT measurements with appropriate reference labels across a broader population. Motion, snoring and profile variables can be considered when recorded and labelled for the same participants; do not join unrelated datasets to fabricate that relationship.

Validate these points before implementing an overnight SVM:

- Separate people, not just windows or nights, between training, tuning, and testing.
- Extract features from fixed windows using training-only decisions. Never calculate baselines using future observations if predictions are meant to be real time.
- Preserve sensor-quality flags, missingness, sampling gaps, and artifacts. Do not turn absent samples into normal oxygen values.
- Use independently confirmed person-level labels for person-level screening, and timestamped event labels for event detection. The two tasks require different evaluation.
- Measure sensitivity, specificity, precision, coverage/abstention, and false alarms per recording period. Use uncertainty estimates that respect clustering by person.
- Test across age ranges, skin pigmentation, body types, comorbidities, and actual hardware conditions represented in the intended use population.

An SVM remains a reasonable small-data baseline after appropriate feature extraction. A more elaborate model cannot repair unsuitable labels or a mismatch between training and device measurements.

## Stage 3 · Validate and prepare deployment

Evaluate the exact sensor-to-feature-to-result pipeline prospectively. Define the intended screening population and how a user obtains follow-up. Set any decision threshold using development data, then freeze it for external testing.

For gateway deployment, keep training separate from startup, retain the artifact checksum/model ID and versioned input contract, protect transport and credentials, and explicitly handle outages and unsupported measurements. The repository includes a local API, a container recipe, and CI tests; it does not provide clinical validation or a production hosting service.

For microcontroller inference, export the fitted standardization parameters, support vectors, coefficients, intercept, and kernel parameters. Compare C/C++ decisions against Python on representative and boundary inputs before using the port. Quantization and memory use must be measured on the target hardware.

This is a screening research project. A raised signal should guide evaluation, and a lower signal must never be represented as excluding apnea. Clinician-directed home sleep apnea testing may offer a more accessible diagnostic pathway for suitable patients.

References: [AASM diagnostic testing guideline](https://aasm.org/resources/clinicalguidelines/diagnostic-testing-osa.pdf), [FDA pulse oximeter limitations](https://www.fda.gov/consumers/consumer-updates/pulse-oximeter-basics), [Kaggle source](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset).
