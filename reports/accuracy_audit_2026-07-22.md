# Accuracy audit — July 22, 2026

## Decision

Keep the deployed checkpoint. Ship the score-interpretation, photo-picker, and conservative near-cutoff UI fixes. Do not claim that classifier accuracy improved.

## Reproduced deployed results

| Set | n | Sensitivity | Specificity | Balanced accuracy | ROC-AUC | Melanoma sensitivity |
|---|---:|---:|---:|---:|---:|---:|
| MILK10k untouched test | 524 | 97.6% | 24.2% | 60.9% | 0.755 | 86.7% (39/45) |
| PAD-UFES-20 phone OOD | 267 | 100.0% | 6.8% | 53.4% | 0.767 | 100.0% (13/13) |

The six missed in-domain melanomas confirm that the app cannot rule out cancer. The phone melanoma subgroup is only n=13 and does not establish real-world readiness.

## Candidate checks

1. **Older concern-only checkpoint — rejected for leakage.** Its old training split overlapped 434 lesions in the final validation split and 421 lesions in the final test split. Its stronger-looking results are invalid for the final evaluation.
2. **Clean frozen linear probes — rejected at the operating point.** The melanoma head's untouched-test ROC-AUC improved from 0.853 to 0.875, but the validation-selected 98%-sensitivity policy produced 97.8% melanoma sensitivity (44/45) with only 6.0% specificity. That is not a better usable classifier.
3. **Four-view mirrored averaging — rejected.** It improved validation discrimination but, using validation-selected thresholds, untouched-test melanoma sensitivity fell to 82.2% (37/45).
4. **Original-or-mirrored flag — rejected.** At the 90% validation melanoma target it reached 88.9% (40/45) test melanoma sensitivity, but balanced accuracy fell to 59.7% and specificity to 21.5%.

No candidate used the test or phone set for fitting or threshold selection. Those sets were read only after each policy was locked from validation data.

## Shipped safety and usability changes

- Removed the prominent percentage meter. Raw sigmoid outputs are uncalibrated and are not cancer probabilities.
- Leads with “Above cutoff,” “Below cutoff,” or “Near cutoff—flagged.”
- Routes a technically usable near-cutoff image to “Review recommended” instead of leaving it without a decision.
- Provides separate **Choose from Photos** and **Take Photo** controls on iPhone.
- Keeps retake requirements for images that are too dark, bright, small, narrow, or lacking visible detail.

## What is required for a genuinely better model

- A fresh lesion-level training run with repeated seeds and stable model selection
- More representative phone photos, especially benign moles and melanomas, divided by patient into train/validation/untouched external test sets
- Calibration on a separate validation set
- Subgroup evaluation with disclosed sample sizes
- Prospective clinical evaluation before any diagnostic or screening claim
