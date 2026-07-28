# AI4ALL Medical AI — Skin Spot Checker

An educational computer-vision project that analyzes a **clinical close-up image of one skin lesion** and returns a cautious “review recommended” or “no model flag—cancer is not ruled out” result for a technically usable image.

> **Important:** This project is not a medical device and cannot diagnose or rule out cancer. A real diagnosis requires evaluation by a qualified clinician and may require a biopsy. Do not use the result to make medical decisions.

The current PWA is **Mega Version 2.0**. It keeps the verified Version 1.6
input gate and Version 1.5 concern ensemble, and adds a real model-evidence
graph plus an optional contour-CNN occlusion-sensitivity map. The iPhone
presentation path also includes a zero-neural-runtime visual-analysis mode
with a region-of-interest overlay and a five-feature graph.

## Included applications

1. `app.py`: a Python/Streamlit research interface.
2. `web/`: an iPhone-friendly progressive web app for Netlify. All processing
   stays in the browser. Desktop mode runs the exported ONNX models; the
   presentation-safe iPhone mode runs lightweight visual features without
   loading the neural runtime. The app does not intentionally upload the
   selected image.

## Model and data

- Architecture: Version 1.6 three-member on-device concern ensemble preceded
  by a learned MobileNetV3-Small visible-lesion gate
- Members: contour-aware MobileNetV3-Large, a clinical EfficientNet-B0, and a clinical-plus-phone EfficientNet-B0
- Fusion: validation-rank transform plus regularized logistic regression, with a preserved dedicated melanoma safety head
- Concern-model training source: MILK10k clinical close-ups plus
  patient-separated PAD-UFES-20 phone photos
- Input-gate training source: participant-labeled SCIN phone images,
  PAD-UFES-20 phone lesions, and sampled SLICE-3D lesion crops
- Input: 224 × 224 RGB image
- Outputs: a general higher-concern score plus a dedicated melanoma-pattern score
- Decision policy: jointly selected validation thresholds; the 1–10 display is
  separated from the conservative melanoma safety flag
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

Version 1.4 added two complementary EfficientNet-B0 RGB members and a small
rank-logistic fusion layer. One RGB member was retrained on the combined
MILK10k and patient-separated PAD-UFES-20 development data. The dedicated
contour-model melanoma head remains the melanoma safety channel. ROC-AUC is the
higher-concern output’s threshold-free ranking metric.

| Evaluation | n | Accuracy | Sensitivity | Specificity | Balanced accuracy | ROC-AUC | Melanoma sensitivity |
|---|---:|---:|---:|---:|---:|---:|---:|
| MILK10k clinical development test | 524 | 71.9% | 97.9% | 6.7% | 52.3% | 0.809 | 100.0% (45/45) |
| PAD-UFES-20 patient-separated phone development test | 199 | 75.4% | 92.6% | 38.1% | 65.4% | 0.858 | 100.0% (4/4) |

Confusion matrices are TN/FP/FN/TP = 10/139/8/367 on the clinical development
test and 24/39/10/126 on the phone development test. Compared with Version 1.3,
Version 1.4 improves every reported phone headline metric and preserves 4/4
melanoma flags. The phone melanoma subgroup is only n=4, so 4/4 is not a stable
claim. The clinical thresholded operating point is effectively flat while
clinical ROC-AUC improves from 0.791 to 0.809.

The v1.4 development tests were reused while comparing release policies. They
must not be described as one-shot independent validation. Only one new RGB
training seed was completed, so training-run variability remains unknown.

### Important score interpretation

The model's raw sigmoid outputs are **not cancer probabilities**. The PWA shows a reader-facing 1–10 **pattern-concern score**: 1 means less similar and 10 means more similar to higher-concern training images, relative to validation-selected cutoffs. Raw outputs remain inside an expandable technical section. Neither the 1–10 score nor a no-flag result can diagnose or rule out cancer.

The July 22 Version 1 audit rejected several candidates with leakage or poor
operating points. Version 1.2 then added patient-separated phone-domain
training. Version 1.4 adds the ensemble experiment and browser verification.
See `reports/ensemble_upgrade_v14_2026-07-27.md`.

Version 1.5 corrects the misleading score behavior seen when normal skin or an
ordinary mole received 7–9/10. It adds a conservative visible-spot gate,
raises the validation-selected decision thresholds, and computes the 1–10
display from fused ensemble evidence only. Phone development specificity rises
from 38.1% to 46.0%, while sensitivity falls from 92.6% to 86.8%. See
`reports/v15_specificity_and_scoring_2026-07-27.md`.

Version 1.6 replaces the hand-built normal-skin check with a learned
MobileNetV3-Small input-routing model. On a case/patient-separated validation
set (n=475), it reached 95.5% visible-lesion sensitivity, 95.0% specificity,
95.3% balanced accuracy, and ROC-AUC 0.979. “Looks healthy” SCIN labels are
participant annotations rather than pathology-confirmed normal skin, so these
numbers measure the routing task only. A new SLICE-3D-trained concern model was
also tested, but it did not improve the locked phone-photo ensemble AUC and was
not deployed. See `reports/v16_learned_input_gate_2026-07-28.md`.

Mega Version 2.0 was informed by a code audit of a supplied DermaScope
prototype. DermaScope’s graph and attention presentation were useful design
ideas, but its six named “models” are documented feature-based analogues with
no trained CNN/transformer weights, and its 0.949 ROC curve is generated from
published benchmark AUCs. Mega therefore does not reuse its classifier or
claim that AUC. The new evidence graph uses this app’s actual model outputs,
and the optional map runs nine additional occlusion checks through the real
contour CNN. See `reports/mega_app_dermascope_audit_2026-07-28.md`.

### iPhone memory-safe presentation mode

Safari on some iPhones exhausted its WebAssembly heap, and WebGL startup also
stalled on at least one deployed-browser test. Mega therefore defaults to a
presentation-safe visual-analysis path on iPhone/iPad. It loads no neural
runtime and cannot trigger the model-memory error. It shows the detected
region plus shape, border, color, contrast, and texture bars. These
deterministic features are explicitly labeled as not the trained CNN and not a
cancer probability. Macs and desktop browsers retain the trained full path.

On the patient-separated PAD-UFES-20 phone development test, this mode reached
91.9% sensitivity, 36.5% specificity, balanced accuracy 64.2%, ROC-AUC 0.855,
and flagged 4/4 melanoma images. The melanoma subgroup is too small for a
stable claim. The full ensemble’s comparison AUC was 0.864.

For debugging, `?mode=compat`, `?mode=minimal`, `?mode=lite`, and `?mode=full`
explicitly select the zero-runtime visual path or the one-, two-, or
full-model path.

## Export for the iPhone web app

```bash
python scripts/export_v14_ensemble.py \
  --clinical-efficientnet-checkpoint models/experiments/efficientnet_b0_seed2026.pt \
  --efficientnet-checkpoint models/experiments/v14_efficientnet_b0_combined_seed2042.pt \
  --ensemble models/v14_ensemble.json
```

The export replaces unsupported activations with equivalent basic ONNX
operations. The PWA runs the three members sequentially and releases each
session before loading the next, so iPhone/iPad Safari avoids holding all model
graphs in memory at once. It also downsizes a temporary browser-only working
copy of large phone photos; the original photo is never uploaded.

The trained desktop path defaults to single-threaded WASM. WebGL remains
available with `?runtime=webgl` for troubleshooting. The default iPhone path
does not import either neural runtime.

The exporter compares PyTorch and ONNX outputs and fails if they differ beyond tolerance.

The v1.6 gate is exported separately and run first:

```bash
python scripts/export_lesion_presence.py \
  --checkpoint models/experiments/v16_lesion_presence_seed2052.pt \
  --output web/public/model/lesion-presence-mobilenet-v3-small.onnx \
  --metadata-output web/public/model/lesion-presence-metadata.json
```

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
- Version 1.4 retains capped tone-aware sampling and a contour stream, inspired by
  recent RGB-plus-structure research. This is a robustness experiment, not proof
  that darker-skin bias has been solved.
- No general-purpose vision-language model is used to describe or diagnose lesions.
- Grad-CAM may be added later as a debugging/teaching view, but a heatmap would not be a clinical explanation.
