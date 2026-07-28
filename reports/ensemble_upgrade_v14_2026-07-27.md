# Version 1.4 ensemble upgrade

## Release decision

Version 1.4 replaces the single deployed contour-aware model with a
three-member, sequentially executed browser ensemble:

1. contour-aware MobileNetV3-Large trained on MILK10k clinical close-ups and
   PAD-UFES-20 phone images;
2. EfficientNet-B0 trained on the MILK10k clinical development split;
3. EfficientNet-B0 retrained on the combined MILK10k and patient-separated
   PAD-UFES-20 development splits.

The higher-concern output uses an empirical-rank transform followed by
regularized logistic regression. The meta-learner was fit from base-model
validation predictions only. The dedicated contour-model melanoma head and its
validation-selected threshold remain the melanoma safety channel.

The larger EfficientNet-B3/B4 and transformer suggestions were not placed in
the phone bundle. The deployed ONNX members already total about 49 MB, and the
app previously experienced browser out-of-memory failures. A Swin/ViT teacher
remains a reasonable future experiment if it is distilled into a smaller
student and improves external phone testing.

## Data separation

- Combined training manifest: 4,828 images
- Combined validation manifest: 683 images
- Clinical development test: 524 MILK10k images, grouped by lesion
- Phone development test: 199 PAD-UFES-20 images, grouped by patient
- Meta-learner: five grouped folds within the validation predictions
- Thresholds: selected on validation predictions only

No clinical or phone test labels were read by the training or fusion-fitting
scripts. The development test sets were evaluated repeatedly while comparing
v1.4 release policies, so they are no longer a one-shot final test and must not
be described as independent clinical validation.

## Measured results

| Evaluation | Version | Accuracy | Sensitivity | Specificity | Balanced accuracy | ROC-AUC | Melanoma sensitivity |
|---|---|---:|---:|---:|---:|---:|---:|
| MILK10k clinical development test, n=524 | v1.3 | 72.1% | 98.1% | 6.7% | 52.4% | 0.791 | 45/45 |
| MILK10k clinical development test, n=524 | v1.4 | 71.9% | 97.9% | 6.7% | 52.3% | 0.809 | 45/45 |
| PAD-UFES-20 phone development test, n=199 | v1.3 | 74.4% | 91.9% | 36.5% | 64.2% | 0.855 | 4/4 |
| PAD-UFES-20 phone development test, n=199 | v1.4 | 75.4% | 92.6% | 38.1% | 65.4% | 0.858 | 4/4 |

The v1.4 release policy improves every reported phone-test headline metric and
preserves 4/4 phone melanoma flags. The phone melanoma subgroup is far too
small to establish melanoma performance. On the clinical development test,
ranking improves while the thresholded operating point is effectively flat and
slightly worse on accuracy and sensitivity. This is evidence of a modest phone
development-set improvement, not a claim of real-world accuracy.

## Browser verification

- All three ONNX exports were numerically checked against their PyTorch models.
- The app runs members sequentially and releases each inference session before
  loading the next member.
- A held-out PAD-UFES-20 image completed inference in the production build on
  the one-thread WASM path. WebGL stalled while compiling a later ensemble
  member in the deployed browser check, so it is retained only as an explicit
  troubleshooting option and is not the iPhone default.
- Photo-library upload, camera input, quality checks, the 1–10 non-probability
  score, and Version 1.4 labeling remain present.

## Remaining limitations

- No prospective, external clinical validation
- Development tests reused during version comparison
- Only one new RGB training seed
- Phone melanoma n=4
- A real iPhone Safari pass is still required because desktop browser
  verification cannot reproduce each device's memory limit
- Fairness remains unknown; subgroup sizes are too small and uneven
- High false-positive rate remains clinically unacceptable
- Three-model inference increases download size and latency
- Metadata, transformer, and explicit color-constancy members were not deployed
  because required user inputs or safe phone-memory evidence are not yet
  available

The app remains an educational screening prototype and must not diagnose,
confirm, or rule out cancer.
