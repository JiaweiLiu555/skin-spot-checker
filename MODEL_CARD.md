# Model card: AI4ALL Skin Spot Checker

## Summary

AI4ALL Skin Spot Checker is a classroom computer-vision prototype with two outputs: a general higher-concern output and a dedicated melanoma-pattern output. For a technically usable image it returns “review recommended” or “no model flag—cancer is not ruled out.” It is not a medical device, is not clinically validated, and cannot diagnose or rule out cancer.

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

- Architecture: contour-aware MobileNetV3-Large with an RGB stream and a fixed-Sobel contour stream encoded by a small learned CNN
- Initialization: ImageNet pretrained weights
- Input: 224 × 224 RGB phone or clinical close-up image
- Objective: weighted binary cross-entropy
- Optimizer: AdamW
- Domain adaptation: final feature stages fine-tuned with MILK10k clinical and PAD-UFES-20 phone images
- Robustness training: camera/lighting augmentation plus capped skin-tone-aware sampling
- Threshold selection: jointly maximize specificity while meeting 95% overall and 98% melanoma validation sensitivity targets when possible
- Splits: lesion-level MILK split and patient-level PAD train/validation/test split

## Results

The checkpoint and numerical thresholds were fit from training/validation data. Multiple candidates and safety policies were compared during development, so the test results below are development holdout evidence rather than a one-shot final clinical validation.

| Evaluation | n | Accuracy | Sensitivity | Specificity | Balanced accuracy | Higher-concern ROC-AUC | Melanoma sensitivity | Confusion TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MILK10k in-domain | 524 | 0.721 | 0.981 | 0.067 | 0.524 | 0.791 | 1.000 (45/45) | 10/139/7/368 |
| PAD-UFES-20 patient-separated phone test | 199 | 0.744 | 0.919 | 0.365 | 0.642 | 0.855 | 1.000 (4/4) | 23/40/11/125 |

The dedicated melanoma head’s ROC-AUC is 0.864 in-domain and 0.905 on the patient-separated phone test. The phone melanoma subgroup is only n=4 and cannot establish melanoma performance.

These results are not clinically acceptable. Version 1.3 improves phone
specificity, balanced accuracy, and ranking, but still misses 11/136
higher-concern phone images and flags many lower-concern images. The clinical
operating point is especially sensitivity-heavy. Full evidence is under
`reports/experiments/` and
`reports/contour_fairness_upgrade_v13_2026-07-23.md`.

Only one final training seed was run. Bootstrap intervals quantify evaluation-sample uncertainty, not training-run variability; repeated-seed variability remains future work.

## Safety design

- Uses “review recommended” and “no model flag—cancer is not ruled out,” not diagnostic labels
- Always displays a medical disclaimer
- Rejects images below a minimum size or with an extreme aspect ratio
- Checks basic brightness and visible-detail proxies and offers capture guidance
- Gives every technically usable image an above-cutoff or below-cutoff screening flag. Images that fail basic quality checks still require a retake.
- Presents a 1–10 pattern-concern score with labeled endpoints. The score is explicitly not a cancer probability. Raw technical outputs remain hidden under an expandable section.
- Performs browser inference on-device in the Netlify PWA

## Limitations

- Retrospective dataset evaluation only
- Unknown performance on truly unconstrained phone photographs
- Potential subgroup performance differences
- Limited examples for some conditions and skin tones
- No prospective, external, or clinical validation
- Very high false-positive rate, especially on smartphone photos
- Final training-run variability has not been measured across repeated seeds
- The basic image-quality filter cannot detect every unusable or out-of-distribution image

## July 22 accuracy and score-interpretation audit

A user correctly noticed that a known cancer image produced a melanoma-pattern output of 18%. The raw sigmoid output is uncalibrated and is not a cancer probability. Version 1.1 removed the percentage meter; Version 2 adds a labeled 1–10 pattern-concern score so the app supplies useful gradation without pretending to estimate cancer probability.

The audit reproduced the final held-out results and confirmed six melanoma false negatives out of 45 in-domain melanoma images (86.7% sensitivity). Two candidate replacements were rejected rather than deployed:

- An older single-head checkpoint had stronger-looking metrics but was trained on a superseded split. Its training data overlapped 434 lesions in the final validation set and 421 lesions in the final test set, so its apparent improvement was leakage.
- Frozen-head refitting and mirrored-view averaging improved some threshold-free ROC-AUC values, but did not improve the complete untouched-test operating point. The most sensitivity-heavy candidate reached 44/45 melanoma detections while specificity fell to 6.0%; mirrored averaging reduced melanoma sensitivity to 37/45 at its validation-selected thresholds.

Version 2 replaces the Version 1 weights after patient-separated phone-domain training. On the same new phone test it improves accuracy, balanced accuracy, specificity, and ROC-AUC, but sensitivity falls to 93.4%. A genuinely high-accuracy successor still requires substantially more benign and melanoma phone images, repeated seeds, calibration, and prospective external evaluation.

## Fairness status

**Not established.** Every available subgroup count is disclosed in each
report’s `skin_tone_metrics.csv`, including groups marked “insufficient data.”
The phone test groups are n=35, 103, 28, 16, and 3, with 14 missing-tone
examples; the n=3 tone-5 group contains no negative examples. The clinical
tone-5 group is n=38. These retrospective, uneven samples cannot establish
parity or absence of bias. Tone-aware sampling and contour fusion improved some
descriptive subgroups and did not improve all of them, so they are not evidence
that bias was removed.

## Camp-deck context

The camp deck reports an educational small-data melanoma result of AUC 0.9426 and, more importantly, an out-of-distribution collapse from roughly 0.93 AUC on familiar images to roughly 0.56 on phone photos. Those results are not directly comparable benchmarks. They motivate this model’s separate phone OOD evaluation, melanoma-specific head, conservative near-cutoff routing, capture checks, and non-diagnostic wording.

Grad-CAM is intentionally not presented as a clinical explanation. A future heatmap may be used only to debug whether the network attends to obvious artifacts. General-purpose vision-language models are not used for lesion diagnosis.
