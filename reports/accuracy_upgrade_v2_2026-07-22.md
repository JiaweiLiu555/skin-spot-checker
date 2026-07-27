# Accuracy upgrade — Version 2

## Outcome

Version 2 keeps the MobileNetV3-Large browser architecture but fine-tunes its final
feature stages with both MILK10k clinical close-ups and PAD-UFES-20 phone photos.
PAD patients are separated across train, validation, and test before training.
The released operating thresholds are selected from validation predictions only
with 98% minimum validation targets for overall and melanoma sensitivity.

The phone-domain ranking and false-positive behavior improve materially. This is
not a clinical-validation result and does not support a 99.999% accuracy claim.

## Patient-separated phone data

| Split | Images | Patients | Higher concern | Lower concern | Melanoma |
|---|---:|---:|---:|---:|---:|
| Train | 636 | 331 | 437 | 199 | 14 |
| Validation | 159 | 82 | 110 | 49 | 4 |
| Test | 199 | 102 | 136 | 63 | 4 |

No patient occurs in more than one PAD split. The phone-test melanoma subgroup is
only n=4 and is too small for a stable melanoma-performance conclusion.

## Same-test comparison

| Model | Evaluation | n | Accuracy | Sensitivity | Specificity | Balanced accuracy | ROC-AUC | Melanoma sensitivity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Version 1 | PAD patient-separated phone test | 199 | 70.4% | 100.0% | 6.3% | 53.2% | 0.716 | 4/4 |
| **Version 2** | PAD patient-separated phone test | 199 | **72.4%** | 93.4% | **27.0%** | **60.2%** | **0.830** | 4/4 |
| Version 1 | MILK10k clinical test | 524 | 76.7% | 97.6% | 24.2% | 60.9% | 0.755 | 39/45 |
| **Version 2** | MILK10k clinical test | 524 | 70.6% | 97.6% | 2.7% | 50.1% | **0.797** | **45/45** |

Version 2 is optimized for the app's phone-photo use case. Its phone ROC-AUC,
accuracy, balanced accuracy, and specificity improve, while phone sensitivity
falls from 100.0% to 93.4%. On the clinical test, ranking and melanoma sensitivity
improve but specificity falls sharply. The model therefore remains a
sensitivity-heavy educational prototype.

Multiple candidate architectures and operating policies were compared during
development. These results should be treated as development evidence, not as a
prospective or final independent clinical validation.

## Reader-facing score

The app displays a 1–10 **pattern-concern score**:

- 1 means less similar to higher-concern training images.
- 10 means more similar to higher-concern training images.
- The score is relative to validation-selected review cutoffs.
- It is not a cancer probability and must not be interpreted as a diagnosis.

