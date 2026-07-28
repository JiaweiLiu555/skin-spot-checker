# Version 1.5 specificity and scoring correction

## Why Version 1.5 was needed

Version 1.4 could show 7–9/10 for normal skin or an ordinary mole because the
display took the largest normalized distance above either model cutoff. The
dedicated melanoma cutoff was intentionally very low, so weak activation from
that safety head could create a dramatic-looking score. The number was never a
cancer probability, but the presentation overstated the strength of the
evidence.

Version 1.5 separates three concepts:

1. A center-versus-border gate checks whether the image appears to contain one
   visible centered spot.
2. The review decision uses validation-selected higher-concern and melanoma
   safety thresholds.
3. The 1–10 display uses only the three-model fused higher-concern output. The
   nonlinear transform reserves 9–10 for genuinely high ensemble consensus.

The melanoma safety head may still recommend follow-up when the displayed
ensemble score is low. The app explains this explicitly instead of inflating
the score.

## Threshold selection

Thresholds were selected using the existing grouped validation predictions
only. The search maximized specificity and then balanced accuracy while
requiring at least 90% overall validation sensitivity and 95% validation
melanoma sensitivity.

- Higher-concern threshold: 0.4477158209
- Dedicated melanoma threshold: 0.0045387894
- Validation sensitivity: 92.4%
- Validation specificity: 20.3%
- Validation balanced accuracy: 56.3%
- Validation melanoma sensitivity: 46/48

## Development-set comparison

| Evaluation | Version | Sensitivity | Specificity | Balanced accuracy | Melanoma sensitivity |
|---|---|---:|---:|---:|---:|
| Clinical development, n=524 | v1.4 | 97.9% | 6.7% | 52.3% | 45/45 |
| Clinical development, n=524 | v1.5 | 95.7% | 9.4% | 52.6% | 44/45 |
| Phone development, n=199 | v1.4 | 92.6% | 38.1% | 65.4% | 4/4 |
| Phone development, n=199 | v1.5 | 86.8% | 46.0% | 66.4% | 4/4 |

The phone specificity improvement is real on this development set, but it is
not large enough to make the app clinically reliable. The policy misses more
higher-concern examples than v1.4. The phone melanoma subgroup contains only
four images and cannot establish sensitivity.

## Lesion-presence gate

The gate threshold was chosen conservatively. It passes 98.8% of validation
lesion images, 97.9% of clinical development lesions, and 98.5% of phone
development lesions. At this threshold it rejected about 30% of synthetic
corner crops taken from lesion images.

Those corner crops are not a substitute for a real normal-skin dataset. The
gate may still accept normal skin with shadows or reject a subtle,
low-contrast lesion. Available skin-tone samples are too small and uneven to
establish fairness, so this check is capture guidance rather than a medical
lesion detector.

## Interpretation

- 1 means little fused higher-concern model evidence.
- 10 means strong fused higher-concern model evidence.
- Neither endpoint estimates cancer probability.
- A low score cannot rule out cancer.
- A safety-head flag is a conservative follow-up signal, not a diagnosis.
