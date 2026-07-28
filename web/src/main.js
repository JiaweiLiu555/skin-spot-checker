import "./style.css";
import {
  LESION_PRESENCE_THRESHOLD,
  lesionPresenceEvidence,
  patternConcernScore,
  screeningDecision,
} from "./screening-policy.js";

const app = document.querySelector("#app");

app.innerHTML = `
  <header class="topbar">
    <img class="brand-mark" src="/icon.svg" alt="" />
    <div>
      <p class="eyebrow">AI4ALL MEDICAL AI · MEGA</p>
      <h1>Skin Spot Checker</h1>
    </div>
  </header>

  <main>
    <section class="warning" aria-label="Medical safety notice">
      <strong>Educational prototype—not a diagnosis.</strong>
      <span>This tool cannot confirm or rule out cancer. A concerning or changing skin spot should be evaluated by a qualified clinician.</span>
    </section>

    <section class="card intro">
      <span class="step">1</span>
      <div>
        <h2>Add a clear close-up</h2>
        <p>Center one skin spot in bright, even light. Avoid blur, shadows, rulers, and extreme zoom.</p>
      </div>
      <div class="upload-actions">
        <label class="upload-button" for="photo-library-input">
          <span>Choose from Photos</span>
          <input id="photo-library-input" type="file" accept="image/jpeg,image/png,image/webp" />
        </label>
        <label class="capture-button" for="camera-input">
          <span>Take Photo</span>
          <input id="camera-input" type="file" accept="image/jpeg,image/png,image/webp" capture="environment" />
        </label>
      </div>
    </section>

    <section id="preview-card" class="card preview-card hidden" aria-live="polite">
      <div class="section-title">
        <span class="step">2</span>
        <div>
          <h2>Review the image</h2>
          <p id="quality-message">Checking image quality…</p>
        </div>
      </div>
      <div class="image-frame"><img id="preview" alt="Selected skin spot" /></div>
      <button id="analyze-button" class="primary-button" type="button" disabled>Analyze on this device</button>
      <p class="privacy-note">Your image stays in this browser and is not uploaded by the app.</p>
    </section>

    <section id="result-card" class="card result-card hidden" aria-live="assertive">
      <p class="eyebrow">MODEL RESULT</p>
      <div id="result-badge" class="result-badge"></div>
      <h2 id="result-title"></h2>
      <p id="result-copy"></p>
      <div class="concern-score" aria-label="Pattern-concern score from 1 to 10">
        <div class="concern-score-heading">
          <span>Pattern-concern score</span>
          <strong><span id="concern-score-value"></span><small>/10</small></strong>
        </div>
        <div class="score-track" aria-hidden="true"><span id="score-fill"></span></div>
        <div class="score-endpoints">
          <span><strong>1</strong> · Less similar to higher-concern training images</span>
          <span><strong>10</strong> · More similar to higher-concern training images</span>
        </div>
      </div>
      <div class="decision-row">
        <span>Screening flag</span>
        <strong id="decision-value"></strong>
      </div>
      <p class="score-warning"><strong>The 1–10 score is not a cancer probability.</strong> It summarizes the strength of fused evidence from the three-model ensemble.</p>
      <section id="focus-panel" class="focus-panel hidden" aria-label="Image-analysis focus">
        <h3>Where the image analysis is focused</h3>
        <p>The yellow box and warm overlay show the lesion-like region isolated by lightweight computer vision. This helps catch framing mistakes; it is not Grad-CAM, proof of cancer, or a clinical explanation.</p>
        <canvas id="focus-canvas" class="focus-canvas" aria-label="Photo with analyzed region highlighted"></canvas>
      </section>
      <section id="evidence-panel" class="evidence-panel hidden" aria-label="Model evidence graph">
        <h3>Model evidence graph</h3>
        <p>Each bar shows that member’s output rank relative to its validation reference images. These are model signals, not cancer probabilities.</p>
        <div id="evidence-bars" class="evidence-bars"></div>
      </section>
      <details id="sensitivity-details" class="sensitivity-details hidden">
        <summary>Show where the contour CNN is sensitive</summary>
        <p>This optional 3 × 3 occlusion test hides one image region at a time and measures how much the contour model’s output changes. It is a coarse debugging/education aid—not a clinical explanation and not proof that a highlighted region is cancer.</p>
        <button id="sensitivity-button" class="secondary-button compact-button" type="button">Generate sensitivity map</button>
        <p id="sensitivity-status" class="fine-print"></p>
        <canvas id="sensitivity-canvas" class="sensitivity-canvas hidden" aria-label="Contour-model occlusion sensitivity map"></canvas>
        <div id="sensitivity-legend" class="sensitivity-legend hidden">
          <span><i class="legend-warm"></i>Hiding this area lowered concern evidence</span>
          <span><i class="legend-cool"></i>Hiding this area raised concern evidence</span>
        </div>
      </details>
      <details class="technical-details">
        <summary>Show technical model outputs</summary>
        <p id="threshold-note" class="fine-print"></p>
      </details>
      <button id="reset-button" class="secondary-button" type="button">Check another image</button>
    </section>

    <section class="card details">
      <details>
        <summary>What does this result mean?</summary>
        <p>The v1.6 input gate first checks that the photo contains a visible centered skin spot. If it does, the three-member ensemble combines a contour-aware model for border and shape patterns with two independent RGB models for color, texture, and broader morphology. The 1–10 display reflects fused ensemble evidence, not cancer probability.</p>
      </details>
      <details>
        <summary>When should I seek care?</summary>
        <p>Contact a qualified clinician about a new, changing, bleeding, painful, or otherwise concerning skin spot regardless of this result.</p>
      </details>
      <details>
        <summary>Known limitations</summary>
        <p>Performance can change with lighting, camera, skin tone, spot type, and image quality. Tone-aware sampling and lighting augmentation do not prove fairness: darker-skin and phone-photo subgroups remain too small and uneven for that claim.</p>
      </details>
    </section>
  </main>

  <footer>For education and research only · On-device computer vision · Mega Version 2.0</footer>
`;

const inputs = Array.from(document.querySelectorAll('input[type="file"]'));
const previewCard = document.querySelector("#preview-card");
const preview = document.querySelector("#preview");
const qualityMessage = document.querySelector("#quality-message");
const analyzeButton = document.querySelector("#analyze-button");
const resultCard = document.querySelector("#result-card");
const resetButton = document.querySelector("#reset-button");

let selectedImage = null;
let metadataPromise = null;
let presenceMetadataPromise = null;
let compactEnsemblePromise = null;
let runtimePromise = null;
let latestAnalysis = null;

const queryParameters = new URLSearchParams(window.location.search);
const isIOS =
  /iPad|iPhone|iPod/.test(navigator.userAgent) ||
  (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
const requestedMode = queryParameters.get("mode");
const inferenceMode =
  requestedMode === "full" ||
  requestedMode === "lite" ||
  requestedMode === "minimal" ||
  requestedMode === "compat"
    ? requestedMode
    : isIOS
      ? "compat"
      : "full";
const useWebGl =
  queryParameters.get("runtime") === "webgl" ||
  (isIOS && queryParameters.get("runtime") !== "wasm");

function loadRuntime() {
  if (!runtimePromise) {
    runtimePromise = useWebGl
      ? import("onnxruntime-web/webgl")
      : import("onnxruntime-web/wasm").then((runtime) => {
          runtime.env.wasm.numThreads = 1;
          runtime.env.wasm.proxy = false;
          return runtime;
        });
  }
  return runtimePromise;
}

function loadMetadata() {
  metadataPromise ??= fetch("/model/model-metadata.json?v=2.0.1-memory-fix", { cache: "no-store" }).then((response) => {
    if (!response.ok) throw new Error("Model metadata is unavailable.");
    return response.json();
  });
  return metadataPromise;
}

function loadPresenceMetadata() {
  presenceMetadataPromise ??= fetch("/model/lesion-presence-metadata.json?v=1.6.0-release-1", {
    cache: "no-store",
  }).then((response) => {
    if (!response.ok) throw new Error("Visible-spot model metadata is unavailable.");
    return response.json();
  });
  return presenceMetadataPromise;
}

function loadCompactEnsemble() {
  if (inferenceMode === "full" || inferenceMode === "compat") return Promise.resolve(null);
  const filename =
    inferenceMode === "lite" ? "mega-lite-ensemble.json" : "mega-minimal-ensemble.json";
  compactEnsemblePromise ??= fetch(`/model/${filename}?v=2.0.1-memory-fix`, {
    cache: "no-store",
  }).then((response) => {
    if (!response.ok) throw new Error("Memory-safe model policy is unavailable.");
    return response.json();
  });
  return compactEnsemblePromise;
}

function modelLabel(model) {
  const labels = {
    contour_mobilenet_v3: "Contour + shape CNN",
    efficientnet_b0_clinical: "Clinical RGB CNN",
    efficientnet_b0_clinical_phone: "Phone-aware RGB CNN",
  };
  return labels[model.name] ?? model.name;
}

function effectiveMetadata(metadata, compactEnsemble) {
  if (!compactEnsemble) return { ...metadata, runtimeMode: "Full three-model ensemble" };
  const names = new Set(compactEnsemble.base_models);
  const models = metadata.models.filter((model) => names.has(model.name));
  return {
    ...metadata,
    models,
    fusion: { ...metadata.fusion, heads: compactEnsemble.heads },
    thresholds: compactEnsemble.thresholds,
    decisionPolicy: { melanomaModelIndex: 0 },
    runtimeMode:
      inferenceMode === "lite"
        ? "Memory-safe two-model ensemble"
        : "iPhone presentation mode · contour CNN",
  };
}

async function createSession(modelUrl) {
  const runtime = await loadRuntime();
  return runtime.InferenceSession.create(modelUrl, {
    executionProviders: [useWebGl ? "webgl" : "wasm"],
    graphOptimizationLevel: useWebGl ? "all" : "basic",
    enableCpuMemArena: false,
    enableMemPattern: false,
    executionMode: "sequential",
  });
}

function checkImage(image) {
  if (image.naturalWidth < 128 || image.naturalHeight < 128) {
    return { accepted: false, message: "This image is too small. Use at least 128 × 128 pixels." };
  }
  const ratio = Math.max(image.naturalWidth, image.naturalHeight) / Math.min(image.naturalWidth, image.naturalHeight);
  if (ratio > 4) {
    return { accepted: false, message: "Use a less narrow image centered on one lesion." };
  }
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0, 128, 128);
  const pixels = context.getImageData(0, 0, 128, 128).data;
  const lesionEvidence = lesionPresenceEvidence(pixels, 128, 128);
  const gray = new Float32Array(128 * 128);
  let brightnessTotal = 0;
  for (let index = 0; index < gray.length; index += 1) {
    const source = index * 4;
    gray[index] = 0.299 * pixels[source] + 0.587 * pixels[source + 1] + 0.114 * pixels[source + 2];
    brightnessTotal += gray[index];
  }
  const brightness = brightnessTotal / gray.length;
  if (brightness < 35) {
    return { accepted: false, message: "Too dark—retake the photo in bright, even light." };
  }
  if (brightness > 225) {
    return { accepted: false, message: "Overexposed—retake the photo without glare or flash washout." };
  }
  let edgeTotal = 0;
  let edgeSquaredTotal = 0;
  let edgeCount = 0;
  for (let y = 1; y < 127; y += 1) {
    for (let x = 1; x < 127; x += 1) {
      const center = y * 128 + x;
      const laplacian =
        4 * gray[center] - gray[center - 1] - gray[center + 1] - gray[center - 128] - gray[center + 128];
      edgeTotal += laplacian;
      edgeSquaredTotal += laplacian * laplacian;
      edgeCount += 1;
    }
  }
  const edgeMean = edgeTotal / edgeCount;
  const edgeVariance = edgeSquaredTotal / edgeCount - edgeMean * edgeMean;
  if (edgeVariance < 20) {
    return { accepted: false, message: "Too little visible detail—retake the photo in focus and closer to the lesion." };
  }
  if (lesionEvidence < LESION_PRESENCE_THRESHOLD) {
    return {
      accepted: false,
      message: "No clear centered skin spot was detected. Move closer and center one visible spot before analyzing.",
    };
  }
  return { accepted: true, message: `${image.naturalWidth} × ${image.naturalHeight} · Ready to analyze` };
}

function imageToTensor(image, metadata, runtime) {
  const size = metadata.imageSize;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0, size, size);
  const rgba = context.getImageData(0, 0, size, size).data;
  const values = new Float32Array(3 * size * size);
  const plane = size * size;
  for (let index = 0; index < plane; index += 1) {
    const source = index * 4;
    values[index] = (rgba[source] / 255 - metadata.mean[0]) / metadata.std[0];
    values[plane + index] = (rgba[source + 1] / 255 - metadata.mean[1]) / metadata.std[1];
    values[2 * plane + index] = (rgba[source + 2] / 255 - metadata.mean[2]) / metadata.std[2];
  }
  return new runtime.Tensor("float32", values, [1, 3, size, size]);
}

function canvasToBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("The photo could not be prepared."))),
      "image/jpeg",
      0.9,
    );
  });
}

async function createMemorySafePhoto(file) {
  if (typeof createImageBitmap !== "function") return file;
  const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  try {
    const maximumEdge = 1024;
    const scale = Math.min(1, maximumEdge / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(bitmap.width * scale));
    canvas.height = Math.max(1, Math.round(bitmap.height * scale));
    const context = canvas.getContext("2d", { alpha: false });
    context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    const resized = await canvasToBlob(canvas);
    canvas.width = 1;
    canvas.height = 1;
    return resized;
  } finally {
    bitmap.close();
  }
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function empiricalRank(score, sortedReference) {
  let lower = 0;
  let upper = sortedReference.length;
  while (lower < upper) {
    const middle = Math.floor((lower + upper) / 2);
    if (score < sortedReference[middle]) upper = middle;
    else lower = middle + 1;
  }
  return lower / sortedReference.length;
}

function fuseModelScores(baseScores, metadata) {
  const outputNames = ["higher_concern", "melanoma"];
  const fused = {};
  outputNames.forEach((outputName, outputIndex) => {
    const head = metadata.fusion.heads[outputName];
    const logit = baseScores.reduce((total, scores, modelIndex) => {
      const rank = empiricalRank(scores[outputIndex], head.rank_references[modelIndex]);
      return total + head.coefficients[modelIndex] * rank;
    }, head.intercept);
    fused[outputName] = sigmoid(logit);
  });
  return {
    higherConcern: fused.higher_concern,
    melanoma: metadata.decisionPolicy
      ? baseScores[metadata.decisionPolicy.melanomaModelIndex][1]
      : fused.melanoma,
  };
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function analyzeVisualFeatures(image) {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0, size, size);
  const pixels = context.getImageData(0, 0, size, size).data;
  const gray = new Float32Array(size * size);
  let borderR = 0;
  let borderG = 0;
  let borderB = 0;
  let borderCount = 0;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const index = y * size + x;
      const source = index * 4;
      const r = pixels[source];
      const g = pixels[source + 1];
      const b = pixels[source + 2];
      gray[index] = 0.299 * r + 0.587 * g + 0.114 * b;
      if (x < 15 || y < 15 || x >= size - 15 || y >= size - 15) {
        borderR += r;
        borderG += g;
        borderB += b;
        borderCount += 1;
      }
    }
  }
  borderR /= borderCount;
  borderG /= borderCount;
  borderB /= borderCount;

  const saliency = new Float32Array(size * size);
  const sortedSaliency = [];
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const index = y * size + x;
      const source = index * 4;
      const dr = pixels[source] - borderR;
      const dg = pixels[source + 1] - borderG;
      const db = pixels[source + 2] - borderB;
      const colorDistance = Math.sqrt(dr * dr + dg * dg + db * db) / 441.7;
      const nx = (x - size / 2) / (size / 2);
      const ny = (y - size / 2) / (size / 2);
      const centerPrior = 0.3 + 0.7 * Math.exp(-(nx * nx + ny * ny) / 0.42);
      saliency[index] = colorDistance * centerPrior;
      sortedSaliency.push(saliency[index]);
    }
  }
  sortedSaliency.sort((a, b) => a - b);
  const threshold = Math.max(0.045, sortedSaliency[Math.floor(sortedSaliency.length * 0.82)]);
  const candidate = new Uint8Array(size * size);
  for (let index = 0; index < candidate.length; index += 1) {
    candidate[index] = saliency[index] >= threshold ? 1 : 0;
  }

  const visited = new Uint8Array(size * size);
  const stack = new Int32Array(size * size);
  let best = null;
  for (let seed = 0; seed < candidate.length; seed += 1) {
    if (!candidate[seed] || visited[seed]) continue;
    let stackSize = 0;
    stack[stackSize++] = seed;
    visited[seed] = 1;
    const indices = [];
    let sumX = 0;
    let sumY = 0;
    let sumStrength = 0;
    let minX = size;
    let minY = size;
    let maxX = 0;
    let maxY = 0;
    while (stackSize > 0) {
      const index = stack[--stackSize];
      const x = index % size;
      const y = Math.floor(index / size);
      indices.push(index);
      sumX += x;
      sumY += y;
      sumStrength += saliency[index];
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      const neighbors = [index - 1, index + 1, index - size, index + size];
      for (const neighbor of neighbors) {
        if (neighbor < 0 || neighbor >= candidate.length || visited[neighbor] || !candidate[neighbor]) continue;
        const neighborX = neighbor % size;
        if (Math.abs(neighborX - x) > 1) continue;
        visited[neighbor] = 1;
        stack[stackSize++] = neighbor;
      }
    }
    if (indices.length < 10) continue;
    const centroidX = sumX / indices.length;
    const centroidY = sumY / indices.length;
    const dx = (centroidX - size / 2) / (size / 2);
    const dy = (centroidY - size / 2) / (size / 2);
    const centrality = 0.25 + 0.75 * Math.exp(-(dx * dx + dy * dy) / 0.35);
    const componentScore = sumStrength * centrality * Math.min(1, indices.length / 45);
    if (!best || componentScore > best.componentScore) {
      best = { indices, centroidX, centroidY, minX, minY, maxX, maxY, componentScore };
    }
  }

  if (!best) {
    const indices = [];
    for (let y = 42; y < 86; y += 1) {
      for (let x = 42; x < 86; x += 1) indices.push(y * size + x);
    }
    best = {
      indices,
      centroidX: 64,
      centroidY: 64,
      minX: 42,
      minY: 42,
      maxX: 85,
      maxY: 85,
      componentScore: 0,
    };
  }

  const mask = new Uint8Array(size * size);
  for (const index of best.indices) mask[index] = 1;
  const area = best.indices.length;
  let perimeter = 0;
  let mismatch = 0;
  let colorTotal = 0;
  let colorSquaredTotal = 0;
  let contrastTotal = 0;
  let textureTotal = 0;
  let textureCount = 0;
  const centerX = (best.minX + best.maxX) / 2;
  const centerY = (best.minY + best.maxY) / 2;
  for (const index of best.indices) {
    const x = index % size;
    const y = Math.floor(index / size);
    const source = index * 4;
    const channels = [pixels[source], pixels[source + 1], pixels[source + 2]];
    for (const channel of channels) {
      colorTotal += channel;
      colorSquaredTotal += channel * channel;
    }
    contrastTotal += saliency[index] / Math.max(0.3, 0.3 + 0.7 * Math.exp(
      -((((x - 64) / 64) ** 2 + ((y - 64) / 64) ** 2) / 0.42),
    ));
    const neighbors = [index - 1, index + 1, index - size, index + size];
    for (const neighbor of neighbors) {
      if (neighbor < 0 || neighbor >= mask.length || !mask[neighbor]) perimeter += 1;
    }
    if (x + 1 < size && mask[index + 1]) {
      textureTotal += Math.abs(gray[index] - gray[index + 1]);
      textureCount += 1;
    }
    const mirrorX = Math.round(2 * centerX - x);
    const mirrorY = Math.round(2 * centerY - y);
    const horizontalMatch =
      mirrorX >= 0 && mirrorX < size ? mask[y * size + mirrorX] : 0;
    const verticalMatch =
      mirrorY >= 0 && mirrorY < size ? mask[mirrorY * size + x] : 0;
    mismatch += (horizontalMatch ? 0 : 1) + (verticalMatch ? 0 : 1);
  }
  const colorSamples = area * 3;
  const colorMean = colorTotal / Math.max(1, colorSamples);
  const colorStd = Math.sqrt(
    Math.max(0, colorSquaredTotal / Math.max(1, colorSamples) - colorMean * colorMean),
  );
  const compactness = (perimeter * perimeter) / Math.max(1, 4 * Math.PI * area);
  const features = {
    asymmetry: clamp01(mismatch / Math.max(1, 2 * area)),
    border: clamp01((compactness - 1) / 7),
    color: clamp01(colorStd / 65),
    contrast: clamp01((contrastTotal / Math.max(1, area)) * 2.4),
    texture: clamp01((textureTotal / Math.max(1, textureCount)) / 42),
  };
  const irregularity =
    0.28 * features.asymmetry +
    0.28 * features.border +
    0.22 * features.color +
    0.12 * features.texture +
    0.1 * features.contrast;
  const score = Math.max(1, Math.min(10, Math.round(1 + 9 * clamp01(irregularity * 1.18))));
  return { canvas, mask, saliency, threshold, best, features, score };
}

function renderFocusMap(analysis) {
  const canvas = document.querySelector("#focus-canvas");
  const context = canvas.getContext("2d");
  const outputSize = 600;
  const sourceSize = analysis.canvas.width;
  canvas.width = outputSize;
  canvas.height = outputSize;
  context.drawImage(analysis.canvas, 0, 0, outputSize, outputSize);

  const overlay = document.createElement("canvas");
  overlay.width = sourceSize;
  overlay.height = sourceSize;
  const overlayContext = overlay.getContext("2d");
  const overlayData = overlayContext.createImageData(sourceSize, sourceSize);
  let maximum = analysis.threshold + 1e-5;
  for (const value of analysis.saliency) maximum = Math.max(maximum, value);
  for (let index = 0; index < analysis.mask.length; index += 1) {
    if (!analysis.mask[index]) continue;
    const alpha = 70 + Math.round(110 * clamp01(analysis.saliency[index] / maximum));
    overlayData.data[index * 4] = 255;
    overlayData.data[index * 4 + 1] = 82;
    overlayData.data[index * 4 + 2] = 45;
    overlayData.data[index * 4 + 3] = alpha;
  }
  overlayContext.putImageData(overlayData, 0, 0);
  context.drawImage(overlay, 0, 0, outputSize, outputSize);

  const scale = outputSize / sourceSize;
  const box = analysis.best;
  const padding = 5;
  const x = Math.max(0, (box.minX - padding) * scale);
  const y = Math.max(0, (box.minY - padding) * scale);
  const width = Math.min(outputSize - x, (box.maxX - box.minX + padding * 2) * scale);
  const height = Math.min(outputSize - y, (box.maxY - box.minY + padding * 2) * scale);
  context.strokeStyle = "#ffd24d";
  context.lineWidth = 7;
  context.strokeRect(x, y, width, height);
  context.fillStyle = "#ffd24d";
  context.fillRect(x, Math.max(0, y - 30), 168, 30);
  context.fillStyle = "#2b2600";
  context.font = "700 18px system-ui, sans-serif";
  context.fillText("region analyzed", x + 9, Math.max(21, y - 9));
  document.querySelector("#focus-panel").classList.remove("hidden");
}

function renderEvidenceGraph(baseScores, fusedScores, metadata) {
  const panel = document.querySelector("#evidence-panel");
  const bars = document.querySelector("#evidence-bars");
  panel.querySelector("h3").textContent = "Model evidence graph";
  panel.querySelector("p").textContent =
    "Each bar shows that member’s output rank relative to its validation reference images. These are model signals, not cancer probabilities.";
  const referenceSets = metadata.fusion.heads.higher_concern.rank_references;
  const entries = baseScores.map((scores, index) => ({
    label: modelLabel(metadata.models[index]),
    value: empiricalRank(scores[0], referenceSets[index]),
  }));
  if (baseScores.length > 1) {
    entries.push({ label: "Fused ensemble evidence", value: fusedScores.higherConcern });
  }
  bars.innerHTML = entries
    .map(
      ({ label, value }, index) => `
        <div class="evidence-row">
          <div class="evidence-label"><span>${label}</span><strong>${Math.round(value * 100)}</strong></div>
          <div class="evidence-track" aria-label="${label}: ${Math.round(value * 100)} out of 100">
            <span class="${index === entries.length - 1 ? "fused" : ""}" style="width:${Math.max(2, value * 100)}%"></span>
          </div>
        </div>`,
    )
    .join("");
  panel.classList.remove("hidden");
  document.querySelector("#sensitivity-details").classList.remove("hidden");
}

function renderFeatureGraph(features) {
  const panel = document.querySelector("#evidence-panel");
  const bars = document.querySelector("#evidence-bars");
  panel.querySelector("h3").textContent = "Visual feature graph";
  panel.querySelector("p").textContent =
    "Compatibility mode compares visible shape, border, color, contrast, and texture. These are educational image features—not CNN outputs or cancer probabilities.";
  const entries = [
    ["Asymmetry", features.asymmetry],
    ["Border irregularity", features.border],
    ["Color variation", features.color],
    ["Lesion contrast", features.contrast],
    ["Texture variation", features.texture],
  ];
  bars.innerHTML = entries
    .map(
      ([label, value]) => `
        <div class="evidence-row">
          <div class="evidence-label"><span>${label}</span><strong>${Math.round(value * 100)}</strong></div>
          <div class="evidence-track" aria-label="${label}: ${Math.round(value * 100)} out of 100">
            <span style="width:${Math.max(2, value * 100)}%"></span>
          </div>
        </div>`,
    )
    .join("");
  panel.classList.remove("hidden");
  document.querySelector("#sensitivity-details").classList.add("hidden");
}

function normalizedThresholdDistance(score, threshold) {
  const boundedThreshold = Math.min(1 - 1e-6, Math.max(1e-6, threshold));
  return score >= boundedThreshold
    ? (score - boundedThreshold) / (1 - boundedThreshold)
    : (score - boundedThreshold) / boundedThreshold;
}

function showResult(scores, metadata, baseScores) {
  document.querySelector(".concern-score").setAttribute(
    "aria-label",
    "Pattern-concern score from 1 to 10",
  );
  document.querySelector(".concern-score-heading > span").textContent = "Pattern-concern score";
  const scoreEndpoints = document.querySelectorAll(".score-endpoints span");
  scoreEndpoints[0].innerHTML =
    "<strong>1</strong> · Less similar to higher-concern training images";
  scoreEndpoints[1].innerHTML =
    "<strong>10</strong> · More similar to higher-concern training images";
  document.querySelector(".concern-score").classList.remove("hidden");
  document.querySelector(".decision-row").classList.remove("hidden");
  document.querySelector(".score-warning").classList.remove("hidden");
  document.querySelector(".score-warning").innerHTML =
    `<strong>The 1–10 score is not a cancer probability.</strong> ` +
    `It summarizes evidence from ${metadata.runtimeMode}.`;
  document.querySelector(".technical-details").classList.remove("hidden");
  const decision = screeningDecision(scores, metadata.thresholds);
  const decisionMargin = Math.max(
    normalizedThresholdDistance(scores.higherConcern, metadata.thresholds.higher_concern),
    normalizedThresholdDistance(scores.melanoma, metadata.thresholds.melanoma),
  );
  const nearCutoff = Math.abs(decisionMargin) < (metadata.abstentionMargin ?? 0.10);
  const reviewRecommended = decision.reviewRecommended || nearCutoff;
  const concernScore = patternConcernScore(scores.higherConcern);
  const badge = document.querySelector("#result-badge");
  const title = document.querySelector("#result-title");
  const copy = document.querySelector("#result-copy");
  badge.textContent = nearCutoff
    ? "Borderline—follow-up recommended"
    : reviewRecommended
      ? "Follow-up recommended"
      : "No model flag";
  badge.className = `result-badge ${nearCutoff ? "uncertain" : reviewRecommended ? "higher" : "lower"}`;
  title.textContent = reviewRecommended
      ? "Review recommended"
      : "No flag detected—cancer is not ruled out";
  copy.textContent = nearCutoff
    ? "The output is close to the model’s cutoff. The app conservatively recommends review instead of treating a borderline image as reassuring."
    : reviewRecommended
      ? decision.melanomaSafetyFlag && !decision.higherConcernFlag
        ? "The dedicated melanoma-pattern safety head crossed its conservative cutoff even though overall ensemble evidence was lower. This is a follow-up flag, not a cancer probability."
        : "The validated model output crossed the review cutoff. The 1–10 score describes model evidence and is not the chance of cancer."
      : "The model did not cross either review cutoff. False negatives occurred in testing, so this result must not reassure you about a known, changing, or concerning spot.";
  document.querySelector("#decision-value").textContent = nearCutoff
    ? "Near cutoff—flagged"
    : reviewRecommended
      ? "Above cutoff"
      : "Below cutoff";
  document.querySelector("#concern-score-value").textContent = concernScore;
  document.querySelector("#score-fill").style.width = `${((concernScore - 1) / 9) * 100}%`;
  document.querySelector("#threshold-note").textContent =
    `${metadata.runtimeMode}. ` +
    `Higher-concern model output ${(scores.higherConcern * 100).toFixed(1)}% ` +
    `(threshold ${(metadata.thresholds.higher_concern * 100).toFixed(1)}%); ` +
    `melanoma-pattern model output ${(scores.melanoma * 100).toFixed(1)}% ` +
    `(threshold ${(metadata.thresholds.melanoma * 100).toFixed(1)}%). ` +
    "The 1–10 display uses only fused ensemble evidence; these raw technical outputs are not cancer probabilities.";
  renderEvidenceGraph(baseScores, scores, metadata);
  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

function showNoVisibleSpotResult(score, metadata) {
  const badge = document.querySelector("#result-badge");
  badge.textContent = "No clear spot detected";
  badge.className = "result-badge lower";
  document.querySelector("#result-title").textContent = "Center one visible skin spot";
  document.querySelector("#result-copy").textContent =
    "The input model did not find enough evidence of a centered lesion or growth, so the app did not produce a concern score. Move closer, use even light, and try again with one spot centered.";
  document.querySelector(".concern-score").classList.add("hidden");
  document.querySelector(".decision-row").classList.add("hidden");
  document.querySelector(".score-warning").classList.add("hidden");
  document.querySelector("#evidence-panel").classList.add("hidden");
  document.querySelector("#focus-panel").classList.add("hidden");
  document.querySelector("#sensitivity-details").classList.add("hidden");
  document.querySelector(".technical-details").classList.remove("hidden");
  document.querySelector("#threshold-note").textContent =
    `Visible-spot model output ${(score * 100).toFixed(1)}% ` +
    `(validation-selected threshold ${(metadata.threshold * 100).toFixed(1)}%). ` +
    "This output only routes the image; it is not a cancer probability.";
  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

function showCompatibilityResult(analysis, backendUnavailable = true) {
  const reviewRecommended = analysis.score >= 6;
  const badge = document.querySelector("#result-badge");
  badge.textContent = backendUnavailable
    ? "Compatibility analysis"
    : "On-device visual analysis";
  badge.className = `result-badge ${reviewRecommended ? "higher" : "lower"}`;
  document.querySelector("#result-title").textContent = reviewRecommended
    ? "Visual review recommended"
    : "No strong visual-irregularity flag";
  const completionCopy = backendUnavailable
    ? "The trained model backend was unavailable, so the app completed"
    : "This iPhone-safe mode completed";
  document.querySelector("#result-copy").textContent = reviewRecommended
    ? `${completionCopy} with lightweight shape, border, color, contrast, and texture analysis. Several visible features were irregular. This is educational image analysis, not a diagnosis.`
    : `${completionCopy} with lightweight visual-feature analysis. It did not find a strong irregularity flag, but this cannot rule out cancer.`;
  document.querySelector(".concern-score").setAttribute(
    "aria-label",
    "Visual-irregularity score from 1 to 10",
  );
  document.querySelector(".concern-score-heading > span").textContent =
    "Visual-irregularity score";
  const scoreEndpoints = document.querySelectorAll(".score-endpoints span");
  scoreEndpoints[0].innerHTML = "<strong>1</strong> · Fewer visible irregularities";
  scoreEndpoints[1].innerHTML = "<strong>10</strong> · More visible irregularities";
  document.querySelector(".concern-score").classList.remove("hidden");
  document.querySelector("#concern-score-value").textContent = analysis.score;
  document.querySelector("#score-fill").style.width = `${((analysis.score - 1) / 9) * 100}%`;
  document.querySelector(".decision-row").classList.remove("hidden");
  document.querySelector("#decision-value").textContent = reviewRecommended
    ? "Visual feature flag"
    : "No strong visual flag";
  document.querySelector(".score-warning").classList.remove("hidden");
  document.querySelector(".score-warning").innerHTML =
    "<strong>Compatibility mode is not the trained cancer model.</strong> " +
    "Its 1–10 number summarizes visible image irregularity and is not a cancer probability.";
  document.querySelector(".technical-details").classList.remove("hidden");
  document.querySelector("#threshold-note").textContent = backendUnavailable
    ? "The neural-network backend was unavailable. The app used deterministic, on-device image features so the analysis and visualization could still complete. No clinical sensitivity or specificity is claimed for this fallback."
    : "iPhone presentation mode avoids loading the large neural-network runtime. It uses deterministic, on-device image features and always shows the region and graph. No clinical sensitivity or specificity is claimed for this mode.";
  renderFeatureGraph(analysis.features);
  document.querySelector("#sensitivity-details").classList.add("hidden");
  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

function makeOccludedTensor(tensor, metadata, runtime, tileX, tileY, gridSize = 3) {
  const size = metadata.imageSize;
  const plane = size * size;
  const values = new Float32Array(tensor.data);
  const x0 = Math.floor((tileX * size) / gridSize);
  const x1 = Math.ceil(((tileX + 1) * size) / gridSize);
  const y0 = Math.floor((tileY * size) / gridSize);
  const y1 = Math.ceil(((tileY + 1) * size) / gridSize);
  for (let channel = 0; channel < 3; channel += 1) {
    for (let y = y0; y < y1; y += 1) {
      for (let x = x0; x < x1; x += 1) {
        values[channel * plane + y * size + x] = 0;
      }
    }
  }
  return new runtime.Tensor("float32", values, [1, 3, size, size]);
}

function drawSensitivityMap(deltas) {
  const canvas = document.querySelector("#sensitivity-canvas");
  const context = canvas.getContext("2d");
  const size = 600;
  const gridSize = 3;
  canvas.width = size;
  canvas.height = size;
  context.drawImage(selectedImage, 0, 0, size, size);
  const maximum = Math.max(...deltas.map((value) => Math.abs(value)), 1e-5);
  deltas.forEach((delta, index) => {
    const x = index % gridSize;
    const y = Math.floor(index / gridSize);
    const x0 = (x * size) / gridSize;
    const y0 = (y * size) / gridSize;
    const magnitude = Math.abs(delta) / maximum;
    context.fillStyle =
      delta >= 0
        ? `rgba(239, 74, 54, ${0.08 + 0.52 * magnitude})`
        : `rgba(47, 135, 210, ${0.08 + 0.52 * magnitude})`;
    context.fillRect(x0, y0, size / gridSize, size / gridSize);
    context.strokeStyle = "rgba(255,255,255,.65)";
    context.lineWidth = 2;
    context.strokeRect(x0, y0, size / gridSize, size / gridSize);
  });
  canvas.classList.remove("hidden");
  document.querySelector("#sensitivity-legend").classList.remove("hidden");
}

function resetExplainability() {
  latestAnalysis = null;
  document.querySelector("#focus-panel").classList.add("hidden");
  document.querySelector("#evidence-panel").classList.add("hidden");
  document.querySelector("#sensitivity-details").classList.add("hidden");
  document.querySelector("#sensitivity-canvas").classList.add("hidden");
  document.querySelector("#sensitivity-legend").classList.add("hidden");
  document.querySelector("#sensitivity-status").textContent = "";
  const button = document.querySelector("#sensitivity-button");
  button.disabled = false;
  button.textContent = "Generate sensitivity map";
}

async function generateSensitivityMap() {
  if (!latestAnalysis || !selectedImage) return;
  const button = document.querySelector("#sensitivity-button");
  const status = document.querySelector("#sensitivity-status");
  button.disabled = true;
  button.textContent = "Generating map…";
  status.textContent = "Running nine additional contour-model checks on this device…";
  try {
    const runtime = await loadRuntime();
    const { metadata, baseScores } = latestAnalysis;
    const tensor = imageToTensor(selectedImage, metadata, runtime);
    const session = await createSession(metadata.models[0].url);
    const deltas = [];
    try {
      for (let tileY = 0; tileY < 3; tileY += 1) {
        for (let tileX = 0; tileX < 3; tileX += 1) {
          const occluded = makeOccludedTensor(tensor, metadata, runtime, tileX, tileY);
          const outputs = await session.run({ [metadata.inputName]: occluded });
          const result = outputs[metadata.outputName];
          deltas.push(baseScores[0][0] - sigmoid(result.data[0]));
          if (typeof result.dispose === "function") result.dispose();
        }
      }
    } finally {
      await session.release();
    }
    drawSensitivityMap(deltas);
    status.textContent =
      "Map complete. Stronger color means the contour model changed more when that region was hidden.";
  } catch (error) {
    status.textContent = `The optional map could not run: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = "Regenerate sensitivity map";
  }
}

async function handleImageSelection(inputElement) {
  const [file] = inputElement.files;
  if (!file) return;
  resetExplainability();
  if (!file.type.startsWith("image/")) {
    qualityMessage.textContent = "Choose a JPG, PNG, or WebP image.";
    return;
  }
  analyzeButton.disabled = true;
  qualityMessage.textContent = "Preparing a memory-safe copy…";
  let preparedFile;
  try {
    preparedFile = await createMemorySafePhoto(file);
  } catch (error) {
    qualityMessage.textContent = `The photo could not be prepared: ${error.message}`;
    qualityMessage.className = "quality-bad";
    return;
  }
  const url = URL.createObjectURL(preparedFile);
  preview.onload = () => {
    URL.revokeObjectURL(url);
    selectedImage = preview;
    const quality = checkImage(preview);
    qualityMessage.textContent = quality.message;
    qualityMessage.className = quality.accepted ? "quality-good" : "quality-bad";
    analyzeButton.disabled = !quality.accepted;
  };
  preview.src = url;
  previewCard.classList.remove("hidden");
  resultCard.classList.add("hidden");
}

inputs.forEach((inputElement) => {
  inputElement.addEventListener("change", () => handleImageSelection(inputElement));
});

analyzeButton.addEventListener("click", async () => {
  if (!selectedImage) return;
  analyzeButton.disabled = true;
  analyzeButton.textContent = "Loading model…";
  let visualAnalysis = null;
  try {
    visualAnalysis = analyzeVisualFeatures(selectedImage);
    renderFocusMap(visualAnalysis);
  } catch {
    document.querySelector("#focus-panel").classList.add("hidden");
  }
  try {
    if (inferenceMode === "compat") {
      if (!visualAnalysis) {
        throw new Error("The visual analysis could not process this image.");
      }
      showCompatibilityResult(visualAnalysis, false);
      qualityMessage.textContent = "Analysis completed on this device.";
      qualityMessage.className = "quality-good";
      return;
    }
    const [sourceMetadata, compactEnsemble, runtime] = await Promise.all([
      loadMetadata(),
      loadCompactEnsemble(),
      loadRuntime(),
    ]);
    const metadata = effectiveMetadata(sourceMetadata, compactEnsemble);
    if (inferenceMode === "full") {
      const presenceMetadata = await loadPresenceMetadata();
      analyzeButton.textContent = "Checking for a visible spot…";
      const presenceTensor = imageToTensor(selectedImage, presenceMetadata, runtime);
      const presenceSession = await createSession(presenceMetadata.url);
      let presenceScore;
      try {
        const presenceOutputs = await presenceSession.run({
          [presenceMetadata.inputName]: presenceTensor,
        });
        const presenceLogits = presenceOutputs[presenceMetadata.outputName];
        presenceScore = sigmoid(presenceLogits.data[0]);
        if (typeof presenceLogits.dispose === "function") presenceLogits.dispose();
      } finally {
        await presenceSession.release();
      }
      if (presenceScore < presenceMetadata.threshold) {
        showNoVisibleSpotResult(presenceScore, presenceMetadata);
        return;
      }
    }
    const tensor = imageToTensor(selectedImage, metadata, runtime);
    const baseScores = [];
    for (let index = 0; index < metadata.models.length; index += 1) {
      analyzeButton.textContent = `Analyzing model ${index + 1} of ${metadata.models.length}…`;
      const session = await createSession(metadata.models[index].url);
      try {
        const outputs = await session.run({ [metadata.inputName]: tensor });
        const logits = outputs[metadata.outputName].data;
        baseScores.push([sigmoid(logits[0]), sigmoid(logits[1])]);
        if (typeof outputs[metadata.outputName].dispose === "function") {
          outputs[metadata.outputName].dispose();
        }
      } finally {
        await session.release();
      }
    }
    const fusedScores = fuseModelScores(baseScores, metadata);
    latestAnalysis = { metadata, baseScores };
    showResult(fusedScores, metadata, baseScores);
  } catch (error) {
    if (visualAnalysis) {
      showCompatibilityResult(visualAnalysis);
      qualityMessage.textContent =
        "Analysis completed in compatibility mode on this device.";
      qualityMessage.className = "quality-good";
    } else {
      qualityMessage.textContent =
        "This image could not be analyzed. Choose another clear, centered photo.";
      qualityMessage.className = "quality-bad";
    }
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.textContent = "Analyze on this device";
  }
});

resetButton.addEventListener("click", () => {
  inputs.forEach((inputElement) => {
    inputElement.value = "";
  });
  selectedImage = null;
  resetExplainability();
  preview.removeAttribute("src");
  previewCard.classList.add("hidden");
  resultCard.classList.add("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
});

document.querySelector("#sensitivity-button").addEventListener("click", generateSensitivityMap);

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("/service-worker.js"));
}

window.addEventListener("load", () => {
  if (inferenceMode === "compat") return;
  const preload = () => Promise.all([loadMetadata(), loadRuntime()]).catch(() => {});
  if ("requestIdleCallback" in window) window.requestIdleCallback(preload, { timeout: 1500 });
  else window.setTimeout(preload, 500);
});
