# Version 1.3 contour and skin-tone robustness upgrade

## Decision

Promote `models/experiments/contour_fair_seed2031_safety95.pt` to
`models/skin_lesion_mobilenet_v3.pt`.

The candidate was selected using `data/combined_v2/val.csv` only. The clinical
and phone test manifests were not opened during training, checkpoint selection,
or threshold selection.

## What changed

- Kept the ImageNet-pretrained MobileNetV3-Large RGB CNN.
- Added a fixed Sobel edge extractor and a small learned contour CNN.
- Fused RGB and contour features before the higher-concern and melanoma heads.
- Initialized the new model to exactly reproduce Version 1.2 before training.
- Added camera/lighting augmentation: brightness, contrast, saturation, hue,
  gamma, autocontrast, mild grayscale, blur, crop, rotation, and flips.
- Added capped skin-tone-aware sampling alongside domain/diagnosis balancing.
- Added validation checkpoint selection that includes eligible skin-tone-group
  AUCs. Groups require at least 20 images, 8 positives, and 8 negatives.
- Selected the release thresholds on validation data only, targeting at least
  95% overall sensitivity and 98% melanoma sensitivity.

## Same-test comparison

| Evaluation | Model | Accuracy | Sensitivity | Specificity | Balanced accuracy | ROC-AUC | Melanoma sensitivity |
|---|---|---:|---:|---:|---:|---:|---:|
| Phone test, n=199 | Version 1.2 RGB | 72.4% | 93.4% | 27.0% | 60.2% | 0.830 | 4/4 |
| Phone test, n=199 | Version 1.3 RGB + contour | 74.4% | 91.9% | 36.5% | 64.2% | 0.855 | 4/4 |
| Clinical test, n=524 | Version 1.2 RGB | 70.6% | 97.6% | 2.7% | 50.1% | 0.797 | 45/45 |
| Clinical test, n=524 | Version 1.3 RGB + contour | 72.1% | 98.1% | 6.7% | 52.4% | 0.791 | 45/45 |

Phone confusion matrices (TN/FP/FN/TP) changed from `17/46/9/127` to
`23/40/11/125`. The new policy catches fewer non-melanoma higher-concern phone
images, while substantially reducing false positives and improving threshold-free
ranking. This tradeoff must be stated rather than hidden.

Clinical confusion matrices changed from `4/145/9/366` to `10/139/7/368`.

## Skin-tone audit

Fairness is **not established**. Results below are descriptive and the groups
are uneven.

- Phone tone class 2 (n=103; 79 positive/24 negative): ROC-AUC improved from
  0.836 to 0.869.
- Phone tone class 3 (n=28; 19 positive/9 negative): ROC-AUC improved from
  0.731 to 0.795.
- Phone tone class 5 has only 3 images and all are positive. No specificity,
  balanced-accuracy, or ROC-AUC conclusion is possible.
- Clinical tone class 5 has n=38 (30 positive/8 negative): sensitivity improved
  from 93.3% to 100%, and ROC-AUC changed from 0.871 to 0.875. This is still a
  small retrospective subgroup.
- Some other clinical subgroup ROC-AUCs were flat or slightly worse. Tone-aware
  sampling and contour fusion therefore do not prove that bias was removed.

Full subgroup tables:

- `reports/experiments/contour_fair_seed2031_safety95_phone_test/skin_tone_metrics.csv`
- `reports/experiments/contour_fair_seed2031_safety95_clinical_test/skin_tone_metrics.csv`

## Stability and limitations

Only one completed contour-training seed is available. An independent repeat was
attempted but could not start because the local execution service reached its
current usage limit. No training-run variability claim is made.

The phone melanoma group is only n=4. Detecting 4/4 is encouraging but is not a
stable estimate. These are development holdouts after multiple model experiments,
not prospective clinical validation.

The contour design is a lightweight adaptation of the general RGB-plus-structure
idea described in *Reducing skin tone bias in dermatology AI via sketch-guided
multimodal fusion* (Scientific Reports, 2026). This project does not reproduce
that paper’s full dual-encoder, distillation, datasets, or fairness study.

