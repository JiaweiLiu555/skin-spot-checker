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

  <footer>For education and research only · On-device model ensemble · Mega Version 2.0</footer>
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
let runtimePromise = null;
let latestAnalysis = null;

const useWebGl = new URLSearchParams(window.location.search).get("runtime") === "webgl";

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
  metadataPromise ??= fetch("/model/model-metadata.json?v=1.6.0-release-1", { cache: "no-store" }).then((response) => {
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

function renderEvidenceGraph(baseScores, fusedScores, metadata) {
  const panel = document.querySelector("#evidence-panel");
  const bars = document.querySelector("#evidence-bars");
  const labels = ["Contour + shape CNN", "Clinical RGB CNN", "Phone-aware RGB CNN"];
  const referenceSets = metadata.fusion.heads.higher_concern.rank_references;
  const entries = baseScores.map((scores, index) => ({
    label: labels[index] ?? metadata.models[index].name,
    value: empiricalRank(scores[0], referenceSets[index]),
  }));
  entries.push({ label: "Fused ensemble evidence", value: fusedScores.higherConcern });
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

function normalizedThresholdDistance(score, threshold) {
  const boundedThreshold = Math.min(1 - 1e-6, Math.max(1e-6, threshold));
  return score >= boundedThreshold
    ? (score - boundedThreshold) / (1 - boundedThreshold)
    : (score - boundedThreshold) / boundedThreshold;
}

function showResult(scores, metadata, baseScores) {
  document.querySelector(".concern-score").classList.remove("hidden");
  document.querySelector(".decision-row").classList.remove("hidden");
  document.querySelector(".score-warning").classList.remove("hidden");
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
        : "The fused ensemble output crossed the review cutoff. The 1–10 score describes model evidence and is not the chance of cancer."
      : "The model did not cross either review cutoff. False negatives occurred in testing, so this result must not reassure you about a known, changing, or concerning spot.";
  document.querySelector("#decision-value").textContent = nearCutoff
    ? "Near cutoff—flagged"
    : reviewRecommended
      ? "Above cutoff"
      : "Below cutoff";
  document.querySelector("#concern-score-value").textContent = concernScore;
  document.querySelector("#score-fill").style.width = `${((concernScore - 1) / 9) * 100}%`;
  document.querySelector("#threshold-note").textContent =
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
  document.querySelector("#sensitivity-details").classList.add("hidden");
  document.querySelector(".technical-details").classList.remove("hidden");
  document.querySelector("#threshold-note").textContent =
    `Visible-spot model output ${(score * 100).toFixed(1)}% ` +
    `(validation-selected threshold ${(metadata.threshold * 100).toFixed(1)}%). ` +
    "This output only routes the image; it is not a cancer probability.";
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
  try {
    const [metadata, presenceMetadata, runtime] = await Promise.all([
      loadMetadata(),
      loadPresenceMetadata(),
      loadRuntime(),
    ]);
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
    const ranOutOfMemory = /out of memory/i.test(error.message);
    qualityMessage.textContent = ranOutOfMemory
      ? "This device needs more free memory. Close other Safari tabs, reopen the app, and try again."
      : `The model could not run: ${error.message}`;
    qualityMessage.className = "quality-bad";
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
  const preload = () => Promise.all([loadMetadata(), loadRuntime()]).catch(() => {});
  if ("requestIdleCallback" in window) window.requestIdleCallback(preload, { timeout: 1500 });
  else window.setTimeout(preload, 500);
});
