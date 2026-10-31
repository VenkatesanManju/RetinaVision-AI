const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const previewImg = document.getElementById("preview-img");
const placeholder = document.getElementById("dropzone-placeholder");
const runBtn = document.getElementById("run-btn");
const serverStatus = document.getElementById("server-status");
const previewClose = document.getElementById("preview-close");
const checkpointSelect = document.getElementById("checkpoint-select");

const resultsEmpty = document.getElementById("results-empty");
const resultsContent = document.getElementById("results-content");

let selectedFile = null;

const CLASS_COLORS = {
  0: "#34d399", // No DR - green
  1: "#fbbf24", // Mild - yellow
  2: "#fb923c", // Moderate - orange
  3: "#f87171", // Severe - red
  4: "#ef4444", // Proliferative - deep red
};

// --- Server health check ---
fetch("/health")
  .then(r => r.json())
  .then(data => {
    if (data.checkpoint_loaded) {
      serverStatus.textContent = "PyTorch CNN Server Active";
    } else {
      serverStatus.textContent = "Model Server Active";
    }
  })
  .catch(() => { serverStatus.textContent = "Server unreachable"; });

// --- Upload interactions ---
dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.style.borderColor = "#38bdf8";
});
dropzone.addEventListener("dragleave", () => {
  dropzone.style.borderColor = "#334155";
});
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.style.borderColor = "#334155";
  if (e.dataTransfer.files.length) {
    handleFile(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) {
    handleFile(e.target.files[0]);
  }
});

function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.classList.remove("hidden");
    placeholder.classList.add("hidden");
    if (previewClose) previewClose.classList.remove("hidden");
  };
  reader.readAsDataURL(file);
  runBtn.disabled = false;
}

  // preview close button
  if (previewClose) {
    previewClose.addEventListener('click', (e) => {
      e.stopPropagation();
      previewImg.src = '';
      previewImg.classList.add('hidden');
      placeholder.classList.remove('hidden');
      selectedFile = null;
      runBtn.disabled = true;
      previewClose.classList.add('hidden');
    });
  }

// --- Run prediction ---
runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  runBtn.disabled = true;
  runBtn.textContent = "Running inference...";

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("checkpoint", checkpointSelect ? checkpointSelect.value : "kappa");

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    const contentType = res.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
      const body = await res.text();
      throw new Error(`Server returned HTTP ${res.status} instead of JSON: ${body.slice(0, 160)}`);
    }
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || `Server returned HTTP ${res.status}`);
    }

    if (data.error) {
      alert("Error: " + data.error);
      return;
    }

    renderResults(data);
  } catch (err) {
    alert("Prediction failed: " + err);
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "\u26A1 Run Ensemble Classification";
  }
});

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function setImage(id, src) {
  const el = document.getElementById(id);
  if (el) el.src = src;
}

function renderResults(data) {
  const confidenceValue = Number.isFinite(Number(data.confidence)) ? Number(data.confidence) : 0;

  resultsEmpty.classList.add("hidden");
  resultsContent.classList.remove("hidden");

  setText("predicted-stage", data.predicted_label || "-");
  setText("predicted-desc", data.description || "-");
  setText("confidence-value", `${confidenceValue.toFixed(1)}%`);
  const accuracyText = data.model_accuracy == null
    ? "Ensemble test accuracy unavailable."
    : `Ensemble test accuracy: ${Number(data.model_accuracy).toFixed(1)}%`;
  setText("model-accuracy", accuracyText);
  const kappaText = data.model_weighted_kappa == null
    ? "Weighted kappa unavailable."
    : `Weighted kappa: ${Number(data.model_weighted_kappa).toFixed(4)}`;
  setText("model-kappa", kappaText + (data.metrics_source ? ` (${data.metrics_source})` : ""));
  const branchAccuracyText = data.branch_accuracies
    ? `ResNet50: ${(Number(data.branch_accuracies.resnet50 || 0) * 100).toFixed(1)}% | `
      + `EfficientNet-B0: ${(Number(data.branch_accuracies.efficientnet_b0 || 0) * 100).toFixed(1)}% | `
      + `Ensemble: ${(Number(data.branch_accuracies.ensemble || 0) * 100).toFixed(1)}%`
    : "Individual model accuracy unavailable.";
  setText("branch-accuracies", branchAccuracyText);

  setImage("input-image", "data:image/png;base64," + (data.input_image_b64 || ""));
  setImage("gradcam-image", "data:image/png;base64," + (data.gradcam_image_b64 || ""));

  const probsList = document.getElementById("probs-list");
  if (probsList) {
    probsList.innerHTML = "";
    (data.class_probabilities || []).forEach((item) => {
      const color = CLASS_COLORS[item.class_index] || "#38bdf8";
      const row = document.createElement("div");
      row.className = "prob-item";
      row.innerHTML = `
        <div class="prob-item-top">
          <span>${item.label}</span>
          <span>${item.probability}%</span>
        </div>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill" style="width:${item.probability}%; background:${color};"></div>
        </div>
      `;
      probsList.appendChild(row);
    });
  }
}
