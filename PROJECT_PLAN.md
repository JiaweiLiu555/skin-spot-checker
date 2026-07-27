# Whole-project plan

## 1. Project question

Can an efficient transfer-learning model flag concerning patterns in clinical close-up lesion photographs while preserving melanoma sensitivity, recognizing uncertainty, and exposing the gap between familiar test images and smartphone photos?

## 2. Success criteria

- The complete pipeline runs from raw metadata to an evaluated model.
- No lesion or patient group appears in more than one data split.
- Results report sensitivity and specificity, not accuracy alone.
- The model keeps a dedicated melanoma output so aggregate results cannot hide melanoma failures.
- Thresholds are selected on validation data only; the in-domain and smartphone OOD test sets remain untouched.
- The app handles invalid uploads, basic photo-quality failures, and uncertain predictions without forcing a binary answer.
- The presentation discusses dataset imbalance, skin-tone representation, acquisition differences, false negatives, and false positives.

## 3. Build phases

### Phase 1 — Define and research

- Define the educational target as concerning-lesion patterns vs. other, with a dedicated melanoma-pattern head, on clinical close-up images.
- Record the dataset source, license, label definitions, and acquisition method.
- Create a model card describing intended use and prohibited use.

Deliverable: a one-page problem statement and dataset card.

### Phase 2 — Prepare and explore the data

- Download the properly licensed ISIC MILK10k clinical close-up release.
- Verify image files, duplicate identifiers, missing labels, and class counts.
- Explore class balance and available demographics.
- Split by lesion or patient group, never randomly by image alone.

Deliverable: reproducible manifests and exploratory charts.

### Phase 3 — Establish a baseline

- Train MobileNetV3-Large using ImageNet transfer learning.
- Use augmentation only on the training split.
- Address class imbalance with a positive-class loss weight.
- Save the best validation checkpoint and jointly selected validation thresholds for the two outputs.

Deliverable: baseline checkpoint and training history.

### Phase 4 — Evaluate and improve

- Evaluate once on the untouched in-domain test split and separately on PAD-UFES-20 smartphone images.
- Report sensitivity, specificity, precision, F1, balanced accuracy, ROC-AUC, confusion matrix, and ROC curve.
- Inspect false positives and false negatives.
- Compare one controlled change at a time, such as augmentation strength, frozen vs. unfrozen features, or MobileNet vs. EfficientNet.
- If metadata supports it, compare performance across skin-tone or demographic groups, disclose every subgroup sample size, and label fairness not established when evidence is insufficient.

Deliverable: results table, error analysis, and limitations section.

### Phase 5 — Build the demo

- Upload JPG/PNG images.
- Reject unreadable, tiny, or extreme-aspect-ratio inputs.
- Show a cautious “higher concern,” “lower concern,” or “unable to assess” output plus both model scores.
- Explain that the score is not a calibrated cancer probability.
- Keep the medical disclaimer visible.

Deliverable: on-device iPhone PWA for Netlify plus a Streamlit fallback demonstration.

### Phase 6 — Present responsibly

- Explain transfer learning and the train/validation/test split visually.
- Show the confusion matrix and emphasize the cost of false negatives.
- Demonstrate with held-out, non-identifying sample images only.
- End with limitations and realistic next steps: external validation, calibration, representative clinical-photo data, subgroup evaluation, and clinical oversight.

Deliverable: 5–8 minute presentation and live demo.

## 4. Suggested schedule

| Session | Goal | Output |
|---|---|---|
| 1 | Scope, safety, dataset | Problem statement |
| 2 | Data audit and split | Manifests and charts |
| 3 | Baseline training | First checkpoint |
| 4 | Evaluation and error analysis | Metrics and figures |
| 5 | Model improvement | Comparison table |
| 6 | App integration | Working demo |
| 7 | Slides and rehearsal | Final presentation |

## 5. Final folder structure

```text
AI4ALL Medical AI/
├── app.py                      # Streamlit user interface
├── train.py                    # Training entry point
├── evaluate.py                 # Test-set evaluation
├── requirements.txt
├── README.md
├── PROJECT_PLAN.md
├── src/
│   ├── data.py                 # Dataset and transforms
│   ├── inference.py            # Validation and prediction rules
│   ├── metrics.py              # Metrics and threshold selection
│   └── model.py                # MobileNetV3 construction/checkpoints
├── scripts/
│   ├── download_milk10k.py     # Official image download
│   ├── prepare_dataset.py      # Group-aware split generation
│   ├── prepare_pad_ood.py      # Smartphone OOD manifest/download
│   └── export_onnx.py          # Browser-model export and validation
├── tests/
│   ├── test_inference.py
│   └── test_metrics.py
├── data/
│   ├── raw/                    # Not committed
│   ├── processed/              # Generated in-domain manifests
│   └── ood/                    # PAD-UFES-20 smartphone set
├── models/                     # Generated checkpoints
├── reports/                    # In-domain, OOD, and baseline results
└── web/                        # Netlify-ready iPhone PWA
```

## 6. Main risks and mitigations

| Risk | Mitigation |
|---|---|
| Phone photos differ from curated clinical-closeup training images | Keep a separate PAD-UFES-20 smartphone OOD evaluation and limit claims |
| Multiple images of the same lesion leak across splits | Group by lesion/patient identifier |
| Overall metrics hide melanoma failures | Weighted two-head loss and melanoma-specific sensitivity |
| A score is mistaken for a diagnosis | Persistent disclaimer and careful labels |
| Overall performance hides subgroup failures | Evaluate subgroups when metadata/sample size permits |
| The model is forced to answer unusable or borderline photos | Quality checks plus an explicit abstention state |
| Small demographic groups are overinterpreted | Disclose subgroup counts and label fairness not established |

## 7. Optional extensions

- Grad-CAM visualization, clearly labeled as an explanatory aid rather than proof
- Probability calibration using the validation set
- Comparison with an ABCDE-inspired feature baseline
- A learned image-quality model beyond the current blur, lighting, distance, and framing checks
- Repeated-seed training with variability reported without reusing test results for model selection
- A size/latency-tested ensemble only if it remains practical in the browser
