# Model card: AI4ALL Skin Spot Checker

## Summary

AI4ALL Skin Spot Checker Version 1.6 is a classroom computer-vision prototype
with a learned visible-spot input gate and two concern-model outputs: a general
higher-concern output and a dedicated melanoma-pattern output. If no centered
spot is detected, it asks for a better-framed image instead of inventing a
concern score. For a technically usable lesion image it returns “review
recommended” or “no model flag—cancer is not ruled out.” It is not a medical
device, is not clinically validated, and cannot diagnose or rule out cancer.

The PWA presentation layer is Mega Version 2.0. It does not change the locked
concern ensemble or its thresholds.

On iPhone/iPad, Mega automatically uses a separately validation-fitted
single-member fusion policy around the contour-aware CNN to avoid Safari
WebAssembly out-of-memory failures. Its phone development-test ROC-AUC was
0.855 versus 0.864 for the full browser ensemble; sensitivity was 0.919 and
specificity 0.365. The full ensemble remains the default on desktop.

## Intended use

- AI education and research demonstrations
- Learning transfer learning, data splitting, evaluation, and responsible AI communication
- Comparing model performance on held-out MILK10k clinical close-up images

## Prohibited use

- Diagnosing a person
- Deciding whether to seek, delay, or change medical care
- Screening a population
- Replacing a clinician, dermoscopy, pathology, or biopsy
- Commercial use inconsistent with the dataset license

## Architecture and training

- Architecture: a three-member ensemble containing a contour-aware MobileNetV3-Large and two independently trained EfficientNet-B0 RGB members
- Fusion: empirical validation-rank features plus regularized logistic regression; the dedicated contour-model melanoma head remains the melanoma safety channel
- Initialization: ImageNet pretrained weights
- Input: 224 × 224 RGB phone or clinical close-up image
- Objective: weighted binary cross-entropy
- Optimizer: AdamW
- Domain adaptation: final feature stages fine-tuned with MILK10k clinical and PAD-UFES-20 phone images
- Robustness training: camera/lighting augmentation plus capped skin-tone-aware sampling
- Threshold selection: jointly maximize specificity while meeting 95% overall and 98% melanoma validation sensitivity targets when possible
- Splits: lesion-level MILK split and patient-level PAD train/validation/test split
- Input gate: ImageNet-pretrained MobileNetV3-Small trained on
  case-separated SCIN participant phone images plus patient-separated
  PAD-UFES-20 lesions and sampled SLICE-3D lesion crops

## Results

The checkpoint and numerical thresholds were fit from training/validation data. Multiple candidates and safety policies were compared during development, so the test results below are development holdout evidence rather than a one-shot final clinical validation.

| Evaluation | n | Accuracy | Sensitivity | Specificity | Balanced accuracy | Higher-concern ROC-AUC | Melanoma sensitivity | Confusion TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MILK10k in-domain | 524 | 0.719 | 0.979 | 0.067 | 0.523 | 0.809 | 1.000 (45/45) | 10/139/8/367 |
| PAD-UFES-20 patient-separated phone test | 199 | 0.754 | 0.926 | 0.381 | 0.654 | 0.858 | 1.000 (4/4) | 24/39/10/126 |

The dedicated melanoma head’s ROC-AUC is 0.864 in-domain and 0.905 on the patient-separated phone test. The phone melanoma subgroup is only n=4 and cannot establish melanoma performance.

Version 1.5 changes the operating policy without retraining the three base
models. On the phone development set, specificity increases from 0.381 to
0.460 and balanced accuracy from 0.654 to 0.664; sensitivity decreases from
0.926 to 0.868. On the clinical development set, specificity increases from
0.067 to 0.094 and sensitivity decreases from 0.979 to 0.957. Full details are
in `reports/v15_specificity_and_scoring_2026-07-27.md`.

The Version 1.6 input gate selected its threshold on a separate case/patient
validation set of 475 images. It reached sensitivity 0.955, specificity 0.950,
balanced accuracy 0.953, and ROC-AUC 0.979 for the narrow task “visible
centered lesion/growth versus participant-labeled looks healthy.” SCIN
`LOOKS_HEALTHY` is not pathology-confirmed normal skin, so these results do
not measure cancer detection and must not be interpreted that way.

A new EfficientNet-B0 concern candidate trained with sampled SLICE-3D data was
evaluated but rejected for release. It improved some operating-point metrics,
but replacing or adding it reduced phone-photo ensemble ROC-AUC (0.857 or
0.852 versus 0.864 for the retained three-member comparison). Its standalone
phone melanoma sensitivity was 2/4, so its melanoma head was not used.

These results are not clinically acceptable. Version 1.4 improves every
reported phone development-test headline metric over Version 1.3, but still
misses 10/136 higher-concern phone images and flags many lower-concern images.
The clinical thresholded operating point is effectively flat while clinical
ROC-AUC improves. The development tests were reused while comparing v1.4
policies, so they are not a one-shot final validation. Full evidence is under
`reports/experiments/` and
`reports/ensemble_upgrade_v14_2026-07-27.md`.

Only one new combined-data EfficientNet training seed was run. Repeated-seed
variability remains future work.

## Safety design

- Uses “review recommended” and “no model flag—cancer is not ruled out,” not diagnostic labels
- Always displays a medical disclaimer
- Rejects images below a minimum size or with an extreme aspect ratio
- Checks basic brightness and visible-detail proxies and offers capture guidance
- Runs a learned visible-lesion gate before the concern ensemble and omits the
  1–10 score when no centered spot is found; this is input routing, not diagnosis
- Gives every technically usable image an above-cutoff or below-cutoff screening flag. Images that fail basic quality checks still require a retake.
- Presents a 1–10 fused-evidence score with labeled endpoints. The dedicated
  melanoma safety head does not inflate this number. The score is explicitly
  not a cancer probability.
- Performs browser inference on-device in the Netlify PWA
- Loads and releases the three ONNX members sequentially to limit peak browser memory
- Uses only the contour-aware member on iPhone/iPad so a presentation can run
  within Safari’s smaller WebAssembly memory allowance
- Shows each real base member’s validation-relative rank in a model-evidence
  graph; none of the bars is a cancer probability
- Offers an optional 3 × 3 occlusion-sensitivity map for the real contour CNN.
  The map is an educational/debugging aid, not a clinical explanation.

## Limitations

- Retrospective dataset evaluation only
- Unknown performance on truly unconstrained phone photographs
- Potential subgroup performance differences
- Limited examples for some conditions and skin tones
- No prospective, external, or clinical validation
- Very high false-positive rate, especially on smartphone photos
- Final training-run variability has not been measured across repeated seeds
- The basic image-quality filter cannot detect every unusable or out-of-distribution image
- The learned gate can reject a real lesion or pass ordinary skin; its
  validation lesion sensitivity was 95.5%, not 100%
- SCIN “looks healthy” is participant-labeled and not pathology-confirmed
- Development tests were reused during v1.4 policy comparison
- The three-member bundle increases phone download size and latency

## July 22 accuracy and score-interpretation audit

A user correctly noticed that a known cancer image produced a melanoma-pattern output of 18%. The raw sigmoid output is uncalibrated and is not a cancer probability. The current app uses a labeled 1–10 pattern-concern score so it supplies useful gradation without pretending to estimate cancer probability.

The audit reproduced the final held-out results and confirmed six melanoma false negatives out of 45 in-domain melanoma images (86.7% sensitivity). Two candidate replacements were rejected rather than deployed:

- An older single-head checkpoint had stronger-looking metrics but was trained on a superseded split. Its training data overlapped 434 lesions in the final validation set and 421 lesions in the final test set, so its apparent improvement was leakage.
- Frozen-head refitting and mirrored-view averaging improved some threshold-free ROC-AUC values, but did not improve the complete untouched-test operating point. The most sensitivity-heavy candidate reached 44/45 melanoma detections while specificity fell to 6.0%; mirrored averaging reduced melanoma sensitivity to 37/45 at its validation-selected thresholds.

Version 1.4 replaces the prior single-model decision with rank-based fusion
across three members while retaining the dedicated melanoma head. Its current
comparisons are reported above. A genuinely high-accuracy successor still
requires substantially more benign and melanoma phone images, repeated seeds,
calibration, and prospective external evaluation.

## Fairness status

**Not established.** Every available subgroup count is disclosed in each
report’s `skin_tone_metrics.csv`, including groups marked “insufficient data.”
The phone test groups are n=35, 103, 28, 16, and 3, with 14 missing-tone
examples; the n=3 tone-5 group contains no negative examples. The clinical
tone-5 group is n=38. These retrospective, uneven samples cannot establish
parity or absence of bias. Tone-aware sampling, contour fusion, and the v1.4
ensemble are not evidence that bias was removed.

## Camp-deck context

The camp deck reports an educational small-data melanoma result of AUC 0.9426 and, more importantly, an out-of-distribution collapse from roughly 0.93 AUC on familiar images to roughly 0.56 on phone photos. Those results are not directly comparable benchmarks. They motivate this model’s separate phone OOD evaluation, melanoma-specific head, conservative near-cutoff routing, capture checks, and non-diagnostic wording.

Grad-CAM is intentionally not presented as a clinical explanation. The
optional occlusion map is used only to debug whether the network is sensitive
to obvious regions or artifacts. General-purpose vision-language models are
not used for lesion diagnosis.

Mega Version 2.0 implements that debugging goal with occlusion sensitivity
rather than a synthetic radial glow or an unverified Grad-CAM claim. It hides
one coarse region at a time and visualizes the signed change in the contour
model output. It is not evidence that a highlighted region is malignant.
