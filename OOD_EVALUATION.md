# Smartphone out-of-domain evaluation

## Why it exists

A random test split can look strong while a model fails on different cameras, lighting, framing, clinics, or populations. The AI4ALL camp deck illustrates this directly with familiar-test AUC near 0.93 and phone-photo AUC near 0.56. Therefore, MILK10k test performance is reported as **in-domain only**.

## External dataset

Version 2 uses the 994 phone images available in the indexed PAD-UFES-20 mirror and separates patients before training.

- Original source: https://doi.org/10.17632/zr7vgbcyr2.1
- License: CC BY 4.0
- Full dataset: 2,298 smartphone images, 1,641 lesions, and 1,373 patients
- Version 2 split: 636 train, 159 validation, and 199 test images
- Patient counts: 331 train, 82 validation, and 102 test
- Cancer labels: BCC, MEL, SCC
- Other labels: ACK, NEV, SEK
- Phone-test melanoma sample: n=4; far too small for a stable conclusion

No PAD patient appears in more than one split. Phone-test images are not used for gradient training or numerical threshold fitting.

## Frozen-model result

- Accuracy: 72.4%
- Sensitivity: 93.4% (127/136)
- Specificity: 27.0% (17/63)
- Balanced accuracy: 60.2%
- Higher-concern-head ROC-AUC: 0.830
- Dedicated melanoma-head ROC-AUC: 0.849
- Melanoma sensitivity: 100.0% (4/4; too small for a stable claim)
- Confusion matrix TN/FP/FN/TP: 17/46/9/127

Version 2 improves phone discrimination and false-positive behavior relative to Version 1, but it still misses higher-concern images and produces many false alarms. The correct conclusion remains that clinical usefulness and real-world readiness are not established.

## Interpretation rules

- Report sensitivity, specificity, balanced accuracy, ROC-AUC, confusion matrix, melanoma-specific sensitivity, and 95% bootstrap intervals.
- Report abstention coverage and treat abstention as a request for retake/clinician follow-up, not a negative result.
- Do not tune the model or threshold on the OOD result and still call it an untouched external test.
- Do not claim real-world readiness even if the OOD result is encouraging; this is one retrospective dataset.
- Report skin-tone counts and label fairness “not established.”
