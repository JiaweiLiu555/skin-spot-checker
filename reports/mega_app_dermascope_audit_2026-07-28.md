# Mega Version 2.0: DermaScope audit and merge decision

## What was inspected

The supplied DermaScope v1.2 folder was reviewed at the source-code level:
markup, styling, image-processing/segmentation code, six-member scoring code,
fusion, ROC drawing, and the attention overlay.

## What was not adopted

DermaScope states in its own README and `models.js` that its six named members
are **feature-based analogues**, not trained EfficientNet, ConvNeXt,
transformer, contour-CNN, color-CNN, or gradient-boosted models. No learned
weights ship with the app.

Its displayed AUC 0.949 is a published reference result. `roc.js` constructs
curves from those published AUC values using a binormal equation; the curves
are not measurements of the browser app. Its “saliency heat” is a radial
gradient centered on a hand-segmented region, not a CNN attribution map.

For those reasons Mega does not reuse:

- the heuristic six-member cancer score;
- the generated 0.949 ROC claim;
- the scan-specific sensitivity/specificity readout derived from that curve;
- the radial glow presented as saliency.

Those components cannot be fairly benchmarked as trained classifiers on the
locked test sets because they are not trained classifiers.

## What was adopted safely

Two interaction ideas were rebuilt against the real on-device ensemble:

1. **Model evidence graph.** It displays the three actual base-model
   higher-concern outputs after the same empirical validation-rank transform
   used by fusion, plus fused ensemble evidence. Values are labeled model
   signals, not cancer probabilities.
2. **Contour-CNN occlusion sensitivity.** On request, the app hides each cell
   of a 3 × 3 grid and reruns the actual contour-aware ONNX model nine times.
   Warm regions are areas whose removal lowered higher-concern evidence; cool
   regions are areas whose removal raised it.

The sensitivity map is deliberately optional because it adds latency. It is a
coarse debugging and education aid, not a clinical explanation.

## Verification

- Existing automated suite: 15 Python tests passed.
- PWA policy suite: 4 JavaScript tests passed.
- Production build completed.
- Held-out PAD-UFES-20 lesion completed the visible-lesion gate, all three
  concern members, evidence graph, and all nine occlusion passes.
- Visual browser inspection confirmed that the strongest warm cell aligned
  with the visible lesion for the smoke-test image. This single example is a
  UI/implementation check, not an explainability-performance claim.

The classifier metrics remain those documented for Versions 1.5 and 1.6.
Mega Version 2.0 adds transparent interaction; it does not claim a new AUC.
