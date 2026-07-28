# Version 1.6 learned input gate

## Release decision

Version 1.6 keeps the Version 1.5 three-member concern ensemble and adds a
small learned input-routing model before it. The gate prevents photos with no
clear centered spot from receiving a misleading 1–10 concern score.

The separate SLICE-3D concern candidate was not deployed because it did not
improve the locked phone-photo ensemble ROC-AUC.

## Visible-lesion gate

- Architecture: ImageNet-pretrained MobileNetV3-Small
- Input: 192 × 192 RGB
- Training rows: 1,884
- Validation rows: 475
- Group overlap between train and validation: 0
- Training sources: SCIN phone images, PAD-UFES-20 phone lesions, and sampled
  SLICE-3D lesion crops
- Threshold selection: maximize specificity subject to at least 95% visible-
  lesion sensitivity, using validation only

Best validation epoch (epoch 5):

| Metric | Result |
|---|---:|
| Sensitivity | 95.5% |
| Specificity | 95.0% |
| Balanced accuracy | 95.3% |
| ROC-AUC | 0.979 |
| Threshold | 0.2381 |

SCIN `LOOKS_HEALTHY` is a participant annotation, not pathology-confirmed
normal skin. These results validate only the input-routing task and are not
cancer-detection performance.

The ONNX export was numerically compared with PyTorch. Maximum absolute output
error was 0.00000067. The browser model is 6.1 MB and is loaded and released
before the concern ensemble to limit peak phone memory.

## Rejected SLICE-3D concern candidate

The candidate was trained from MILK10k, PAD-UFES-20, and sampled permissive
SLICE-3D development data. Patient grouping was used where patient identifiers
were available. Thresholds were selected on development validation only.

Standalone locked evaluation:

| Evaluation | n | Sensitivity | Specificity | Balanced accuracy | ROC-AUC | Melanoma sensitivity |
|---|---:|---:|---:|---:|---:|---:|
| MILK10k clinical development test | 524 | 97.9% | 22.1% | 60.0% | 0.805 | 42/45 |
| PAD-UFES-20 phone development test | 199 | 91.2% | 44.4% | 67.8% | 0.776 | 2/4 |

Replacing the prior phone-trained RGB member produced phone ensemble ROC-AUC
0.857. Adding the candidate as a fourth member produced 0.852. The retained
three-member comparison was 0.864, so neither new fusion was released. The
phone melanoma subgroup is only n=4 and cannot establish melanoma performance.

## Browser verification

The production build was tested with:

- A held-out SCIN `LOOKS_HEALTHY` image: the app returned “No clear spot
  detected” and did not show a concern score.
- A held-out PAD-UFES-20 lesion image: the image passed the gate and completed
  sequential inference through all three concern members.

This is an educational screening prototype, not a diagnosis or a medical
device. Fairness remains not established.
