import "./style.css";

const app = document.querySelector("#app");

app.innerHTML = `
  <header class="topbar">
    <img class="brand-mark" src="/icon.svg" alt="" />
    <div>
      <p class="eyebrow">AI4ALL MEDICAL AI</p>
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
      <p class="score-warning"><strong>The 1–10 score is not a cancer probability.</strong> It shows how far the model output is below or above its validation-selected review cutoff.</p>
      <details class="technical-details">
        <summary>Show technical model outputs</summary>
        <p id="threshold-note" class="fine-print"></p>
      </details>
      <button id="reset-button" class="secondary-button" type="button">Check another image</button>
    </section>

    <section class="card details">
      <details>
        <summary>What does this result mean?</summary>
        <p>The contour-aware CNN compares both color patterns and lesion-border structure with labeled clinical and phone close-up images. “Above cutoff” means the image crossed a validation-selected screening threshold. It does not estimate the chance that you have cancer, and the model has not been clinically validated.</p>
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

  <footer>For education and research only · Contour-aware on-device CNN · Version 1.3</footer>
`;

const inputs = Array.from(document.querySelectorAll('input[type="file"]'));
const previewCard = document.querySelector("#preview-card");
const preview = document.querySelector("#preview");
const qualityMessage = document.querySelector("#quality-message");
const analyzeButton = document.querySelector("#analyze-button");
const resultCard = document.querySelector("#result-card");
const resetButton = document.querySelector("#reset-button");

let selectedImage = null;
let sessionPromise = null;
let metadataPromise = null;
let runtimePromise = null;

const isAppleMobile =
  /iPad|iPhone|iPod/.test(navigator.userAgent) ||
  (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
const useWebGl = isAppleMobile || new URLSearchParams(window.location.search).get("runtime") === "webgl";

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
  metadataPromise ??= fetch("/model/model-metadata.json", { cache: "no-cache" }).then((response) => {
    if (!response.ok) throw new Error("Model metadata is unavailable.");
    return response.json();
  });
  return metadataPromise;
}

function loadSession() {
  sessionPromise ??= loadRuntime().then((runtime) =>
    runtime.InferenceSession.create("/model/skin-lesion-classifier.onnx", {
      executionProviders: [useWebGl ? "webgl" : "wasm"],
      graphOptimizationLevel: useWebGl ? "all" : "basic",
      enableCpuMemArena: false,
      enableMemPattern: false,
      executionMode: "sequential",
    }),
  );
  return sessionPromise;
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

function normalizedThresholdDistance(score, threshold) {
  const boundedThreshold = Math.min(1 - 1e-6, Math.max(1e-6, threshold));
  return score >= boundedThreshold
    ? (score - boundedThreshold) / (1 - boundedThreshold)
    : (score - boundedThreshold) / boundedThreshold;
}

function patternConcernScore(decisionMargin) {
  const boundedMargin = Math.max(-1, Math.min(1, decisionMargin));
  return Math.max(1, Math.min(10, Math.round(1 + 9 * ((boundedMargin + 1) / 2))));
}

function showResult(scores, metadata) {
  const higherConcern =
    scores.higherConcern >= metadata.thresholds.higher_concern ||
    scores.melanoma >= metadata.thresholds.melanoma;
  const decisionMargin = Math.max(
    normalizedThresholdDistance(scores.higherConcern, metadata.thresholds.higher_concern),
    normalizedThresholdDistance(scores.melanoma, metadata.thresholds.melanoma),
  );
  const nearCutoff = Math.abs(decisionMargin) < (metadata.abstentionMargin ?? 0.10);
  const reviewRecommended = higherConcern || nearCutoff;
  const concernScore = patternConcernScore(decisionMargin);
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
      ? "One or more model outputs crossed the review cutoff. Even a raw output such as 18% can be above its cutoff; it is not an 18% chance of cancer."
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
    "These uncalibrated outputs are shown only for project transparency and are not cancer probabilities.";
  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function handleImageSelection(inputElement) {
  const [file] = inputElement.files;
  if (!file) return;
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
    const [metadata, session, runtime] = await Promise.all([loadMetadata(), loadSession(), loadRuntime()]);
    analyzeButton.textContent = "Analyzing…";
    const tensor = imageToTensor(selectedImage, metadata, runtime);
    const outputs = await session.run({ [metadata.inputName]: tensor });
    const logits = outputs[metadata.outputName].data;
    showResult(
      { higherConcern: sigmoid(logits[0]), melanoma: sigmoid(logits[1]) },
      metadata,
    );
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
  preview.removeAttribute("src");
  previewCard.classList.add("hidden");
  resultCard.classList.add("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
});

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("/service-worker.js"));
}

window.addEventListener("load", () => {
  const preload = () => loadSession().catch(() => {
    sessionPromise = null;
  });
  if ("requestIdleCallback" in window) window.requestIdleCallback(preload, { timeout: 1500 });
  else window.setTimeout(preload, 500);
});
