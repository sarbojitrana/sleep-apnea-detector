# From prototype to an accessible screening device

## Current build

The application predicts a person-level dataset label from six daily/profile inputs. It is suitable for demonstrating the software workflow and testing a local device-to-gateway connection. It has no real-time apnea event detector or alarm.

| Input | Accessible source | Important boundary |
| --- | --- | --- |
| Age and dataset gender category | User profile | Only two source categories; no validated extrapolation |
| BMI category | Measured weight/height plus an appropriate category | Source category cutoffs are not documented |
| Resting pulse | Optical PPG sensor, wearable, or manual measurement | Validate pulse extraction, reject movement and poor contact |
| Daily steps | Accelerometer or phone summary | Calibrate step counting; do not substitute night-only steps |
| Sleep duration | Manual sleep diary or validated wearable estimate | Quiet wakefulness is not necessarily sleep |

No EEG, ECG, cuff pressure, or stress sensor is required. Entry points require all six inputs so that device data loss never silently becomes an invented normal measurement.

## Stage 1 · Connect a daily-summary device

1. Choose a microcontroller that can transmit JSON, and an optical pulse sensor with a documented sampling interface.
2. Obtain a reliable resting-pulse summary, steps/day, and sleep hours from the sensor/companion workflow. These algorithms are outside this repository.
3. Keep profile data on the companion/gateway and combine it with the daily summary.
4. Send the exact example JSON schema to the authenticated local API over a trusted connection.
5. Handle `insufficient_evidence` and `null` screening flags explicitly. An unsupported or missing input is not a negative result.
6. Display the next step and model ID; preserve the distinction between example data and measured data.

The bundled CSV can exercise lower/raised signals, invalid rows, and unsupported profiles before connecting hardware. Do not send individual sensor samples to a model trained on daily summaries.

A Python gateway allows offline operation on the local network. Remote hosting introduces connectivity and privacy requirements. An ESP32/Arduino cannot execute this scikit-learn artifact directly.

## Stage 2 · Obtain appropriate training evidence

Before offering real physiological screening, obtain a dataset containing inexpensive sensor channels actually available on the chosen device. Match each recording with independently verified reference labels and documented sensor units, sample rate, timing, and subject identifiers.

For a future overnight model, possible inexpensive channels include optical pulse/SpO₂, motion, and a carefully validated snoring/acoustic signal. Which channels are useful must be established from data. The current Kaggle CSV provides no basis for training or testing those features.

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
