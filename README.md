# AI4ALL Medical AI — Skin Spot Checker

An educational computer-vision project that analyzes a **clinical close-up image of one skin lesion** and returns a cautious “review recommended” or “no model flag—cancer is not ruled out” result for a technically usable image.

> **Important:** This project is not a medical device and cannot diagnose or rule out cancer. A real diagnosis requires evaluation by a qualified clinician and may require a biopsy. Do not use the result to make medical decisions.

## Included applications

1. `app.py`: a Python/Streamlit research interface.
2. `web/`: an iPhone-friendly progressive web app for Netlify. It runs the exported model locally in the browser through ONNX Runtime Web, so the app does not intentionally upload the selected image.

## Model and data

- Architecture: contour-aware MobileNetV3-Large CNN with ImageNet transfer learning
- Image streams: RGB features plus fixed-Sobel contours encoded by a small learned CNN
- Training source: MILK10k clinical close-ups plus patient-separated PAD-UFES-20 phone photos
- Input: 224 × 224 RGB image
- Outputs: a general higher-concern score plus a dedicated melanoma-pattern score
- Decision policy: jointly selected validation thresholds; every image that passes the basic quality checks receives a screening flag
- Split: lesion-level stratified 80% training / 10% validation / 10% test
- Higher-concern labels: AKIEC, BCC, MAL_OTH, MEL, SCCKA
- Lower-concern labels: BEN_OTH, BKL, DF, INF, NV, VASC

See `DATASET.md` for the license, citation, mapping, and limitations. See `MODEL_CARD.md` before quoting or demonstrating results.

## Folder structure

```text
AI4ALL Medical AI/
├── app.py
├── train.py
├── evaluate.py
├── netlify.toml
├── DATASET.md
├── MODEL_CARD.md
├── PROJECT_PLAN.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   └── ood/
├── models/
├── reports/
├── scripts/
│   ├── download_milk10k.py
│   ├── prepare_dataset.py
│   ├── prepare_pad_ood.py
│   └── export_onnx.py
├── src/
├── tests/
└── web/
    ├── src/
    ├── public/model/
    └── package.json
```

## Python setup

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run tests:

```bash
PYTHONPATH=. python -m pytest -q
```

## Reproduce the dataset

Download the official MILK10k metadata and ground truth from the ISIC Challenge data page, then run:

```bash
python scripts/download_milk10k.py \
  --metadata data/raw/MILK10k_Training_Metadata.csv \
  --output data/raw/milk10k_clinical

python scripts/prepare_dataset.py \
  --metadata data/raw/MILK10k_Training_Metadata.csv \
  --ground-truth data/raw/MILK10k_Training_GroundTruth.csv \
  --images data/raw/milk10k_clinical \
  --output data/processed
```

The download is resumable. Existing verified image files are skipped.

## Train and evaluate

```bash
python train.py \
  --data-dir data/processed \
  --epochs 10 \
  --batch-size 32 \
  --min-sensitivity 0.90 \
  --min-melanoma-sensitivity 0.90 \
  --seed 137

python evaluate.py \
  --checkpoint models/skin_lesion_mobilenet_v3.pt \
  --manifest data/processed/test.csv
```

The two checkpoint thresholds are selected jointly using validation data only. Final evaluation writes sensitivity, specificity, balanced accuracy, ROC-AUC, melanoma-specific sensitivity, a confusion matrix, an ROC curve, bootstrap intervals, abstention coverage, and skin-tone subgroup counts under `reports/`.

The in-domain MILK10k test is not evidence of phone readiness. Version 2 adds a patient-separated PAD-UFES-20 phone train/validation/test split under `data/pad_v2/`; the final phone test contains patients absent from phone training and validation.

## Verified results

Version 1.3 adds a Sobel-contour CNN branch, phone-domain fine-tuning,
camera/lighting augmentation, capped skin-tone-aware sampling, and a
validation-selected operating point. ROC-AUC is the higher-concern head’s
threshold-free ranking metric.

| Evaluation | n | Accuracy | Sensitivity | Specificity | Balanced accuracy | ROC-AUC | Melanoma sensitivity |
|---|---:|---:|---:|---:|---:|---:|---:|
| MILK10k in-domain test | 524 | 72.1% | 98.1% | 6.7% | 52.4% | 0.791 | 100.0% (45/45) |
| PAD-UFES-20 patient-separated phone test | 199 | 74.4% | 91.9% | 36.5% | 64.2% | 0.855 | 100.0% (4/4) |

Confusion matrices are TN/FP/FN/TP = 10/139/7/368 in-domain and
23/40/11/125 on the phone test. Compared with Version 1.2 on the same phone
test, Version 1.3 improves accuracy (72.4% → 74.4%), balanced accuracy
(60.2% → 64.2%), specificity (27.0% → 36.5%), and ROC-AUC (0.830 → 0.855),
while sensitivity falls from 93.4% to 91.9%. The phone melanoma subgroup is
only n=4, so 4/4 is not a stable claim.

The reports include 500-resample bootstrap intervals. Only one final training seed was run, so these intervals describe evaluation-sample uncertainty, not training-run variability. Repeated-seed training remains future work and must never reuse a test set for checkpoint or threshold selection.

### Important score interpretation

The model's raw sigmoid outputs are **not cancer probabilities**. The PWA shows a reader-facing 1–10 **pattern-concern score**: 1 means less similar and 10 means more similar to higher-concern training images, relative to validation-selected cutoffs. Raw outputs remain inside an expandable technical section. Neither the 1–10 score nor a no-flag result can diagnose or rule out cancer.

The July 22 Version 1 audit rejected several candidates with leakage or poor
operating points. Version 1.2 then added patient-separated phone-domain
training. Version 1.3 adds the contour and skin-tone robustness experiment.
See `reports/accuracy_upgrade_v2_2026-07-22.md` and
`reports/contour_fairness_upgrade_v13_2026-07-23.md`.

## Export for the iPhone web app

```bash
python scripts/export_onnx.py \
  --checkpoint models/skin_lesion_mobilenet_v3.pt \
  --output web/public/model/skin-lesion-classifier.onnx
```

The export replaces `HardSwish` and `HardSigmoid` with equivalent basic ONNX
operations so iPhone/iPad Safari can use the lower-memory WebGL runtime. The PWA
also downsizes a temporary browser-only working copy of large phone photos before
inference; the original photo is never uploaded.

The exporter compares PyTorch and ONNX outputs and fails if they differ beyond tolerance.

## Run the Netlify app locally

```bash
cd web
npm install
npm run dev
```

Production check:

```bash
npm run build
npm run preview
```

## Deploy to Netlify

Connect the repository in Netlify or run a Netlify CLI deployment from the project root. `netlify.toml` configures the `web` build automatically:

- Base directory: `web`
- Build command: `npm ci && npm run build`
- Publish directory: `web/dist`

After deployment, open the HTTPS site in Safari on an iPhone and choose **Share → Add to Home Screen**.

## Responsible presentation language

Say: “The model identifies image patterns associated with higher-concern categories in held-out MILK10k clinical close-up images.”

Do not say: “The app diagnoses skin cancer from any phone photo.”

## Connections to the AI4ALL camp deck

- The deck’s slide-19 example reports AUC 0.9426 from a pretrained small-data melanoma model trained for 10 epochs at learning rate 0.0005 with repeated runs. It is an educational comparison, not a directly comparable benchmark: its data, label target, split, and evaluation domain differ from this project.
- The deck’s central result—approximately 0.93 familiar-test AUC collapsing to about 0.56 on phone photos—is why this project keeps an external smartphone OOD evaluation instead of treating a random test split as real-world validation.
- A one-head baseline’s 45.5% melanoma sensitivity showed how headline accuracy/AUC can hide a dangerous subtype failure. The final architecture therefore preserves a dedicated melanoma head and reports melanoma sensitivity separately.
- The app checks basic brightness/detail/framing and requires a retake for unusable images. A technically usable image near the model cutoff is conservatively routed to “review recommended.”
- Skin-tone results always include sample sizes and are labeled **fairness not established**. Small or missing groups cannot prove absence of bias.
- Version 1.3 uses capped tone-aware sampling and a contour stream, inspired by
  recent RGB-plus-structure research. This is a robustness experiment, not proof
  that darker-skin bias has been solved.
- No general-purpose vision-language model is used to describe or diagnose lesions.
- Grad-CAM may be added later as a debugging/teaching view, but a heatmap would not be a clinical explanation.
