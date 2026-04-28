const API = window.location.origin || "";
const THREAD_KEY = "paw-connect-threads";
const LOCATION_CACHE_TTL_MS = 5 * 60 * 1000;

const state = {
  location: { lat: null, lng: null, name: null, accuracy: null, timestamp: null, source: null },
  currentFile: null,
  lastReport: null,
  currentAnimalReportIndex: 0,
  modalReport: null,
  modalAnimalReportIndex: 0,
  stream: null,
  reports: [],
  threads: [],
  activeThread: null,
  editingMessageIndex: null,
  reportLoadSeq: 0,
  selectedReportIds: new Set(),
  dashboardView: window.localStorage?.getItem("paw-connect-dashboard-view") || "cards",
  dashboardSelecting: false,
  rescueContext: null,
  publicConfig: {
    upiVpa: null,
    upiPayeeName: "Paw Connect",
    upiNote: "Support street animal care",
  },
  rescueSuggestionsTimer: null,
  rescueSuggestionAbort: null,
  rescueAutocompleteItems: [],
};

document.addEventListener("DOMContentLoaded", () => {
  loadPublicConfig();
  bindNavigation();
  bindScan();
  bindChat();
  bindDashboard();
  bindRescue();
  bindModal();
  bindDonation();
  loadCareBotHistory();
  loadReports();
  loadDbHealth();
  loadRescueContacts();
  window.addEventListener("resize", handleResponsiveLayoutChange);
  showPage("scan");
  syncSidebarChevron();
  requestPreciseLocation(false);
});

function bindNavigation() {
  document.querySelectorAll("[data-page]").forEach((button) => {
    button.addEventListener("click", () => showPage(button.dataset.page));
  });
  document.getElementById("mobileMenuBtn")?.addEventListener("click", toggleMobileSidebar);
  document.getElementById("sidebarBackdrop")?.addEventListener("click", closeMobileSidebar);
  document.getElementById("locBtn")?.addEventListener("click", enableLocation);
}

function showPage(name) {
  document.querySelectorAll(".page").forEach((page) => page.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach((nav) => nav.classList.remove("active"));
  const page = document.getElementById(`page-${name}`);
  const nav = document.getElementById(`nav-${name}`);
  if (page) page.classList.add("active");
  if (nav) nav.classList.add("active");
  closeMobileSidebar();
  if (name === "dashboard") renderReportCards(getFilteredReports());
  if (name === "rescue") loadRescueContacts();
  if (name === "carebot") renderMessages();
}

function toggleMobileSidebar() {
  const sidebar = document.getElementById("sidebar");
  if (!sidebar) return;
  const isOpen = sidebar.classList.toggle("open");
  document.body.classList.toggle("sidebar-open", isOpen);
  document.getElementById("sidebarBackdrop")?.classList.toggle("open", isOpen);
  updateMobileMenuState(isOpen);
}

function closeMobileSidebar() {
  document.getElementById("sidebar")?.classList.remove("open");
  document.getElementById("sidebarBackdrop")?.classList.remove("open");
  document.body.classList.remove("sidebar-open");
  updateMobileMenuState(false);
}

function updateMobileMenuState(isOpen) {
  const btn = document.getElementById("mobileMenuBtn");
  if (!btn) return;
  btn.setAttribute("aria-expanded", String(isOpen));
  btn.classList.toggle("open", isOpen);
}

function handleResponsiveLayoutChange() {
  syncDashboardViewButtons();
  if (document.getElementById("page-dashboard")?.classList.contains("active")) {
    renderReportCards(getFilteredReports());
  }
}

function bindScan() {
  const uploadZone = document.getElementById("uploadZone");
  const fileInput = document.getElementById("fileInput");
  const openCameraBtn = document.getElementById("openCameraBtn");
  const captureBtn = document.getElementById("captureBtn");
  const closeCameraBtn = document.getElementById("closeCameraBtn");
  const analyseBtn = document.getElementById("analyseBtn");
  const askCareBotBtn = document.getElementById("askCareBotBtn");
  const closeScanReportBtn = document.getElementById("closeScanReportBtn");

  uploadZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    uploadZone.classList.add("drag-over");
  });
  uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
  uploadZone.addEventListener("drop", handleDrop);
  fileInput.addEventListener("change", handleFileSelect);
  openCameraBtn.addEventListener("click", openCamera);
  captureBtn.addEventListener("click", capturePhoto);
  closeCameraBtn.addEventListener("click", closeCamera);
  analyseBtn.addEventListener("click", analyseImage);
  askCareBotBtn?.addEventListener("click", askCareBotAboutReport);
  closeScanReportBtn?.addEventListener("click", closeScanReport);
  document.addEventListener("click", handleReportActionClick);
}

function bindDonation() {
  const form = document.getElementById("donationForm");
  if (!form) return;
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    await openUpiDonationFlow();
  });
  const copyBtn = document.getElementById("copyUpiBtn");
  if (copyBtn) copyBtn.addEventListener("click", copyUpiId);
  const openBtn = document.getElementById("openUpiBtn");
  if (openBtn) openBtn.addEventListener("click", openUpiDonationFlow);
}

async function loadPublicConfig() {
  try {
    const response = await fetch(`${API}/public-config`);
    if (!response.ok) return;
    const data = await response.json();
    state.publicConfig.upiVpa = data?.upi_vpa || null;
    state.publicConfig.upiPayeeName = data?.upi_payee_name || "Paw Connect";
    state.publicConfig.upiNote = data?.upi_note || "Support street animal care";
    updateUpiUi();
  } catch (error) {
    console.warn("Public config could not be loaded.", error);
  }
}

function enableLocation() {
  requestPreciseLocation(true);
}

function requestPreciseLocation(showAlertOnFailure = false) {
  requestBestEffortLocation({ forceFresh: true, strictHighAccuracy: true, showAlertOnFailure }).catch(() => {
    updateLocationUI(false, "Location permission needed");
    if (showAlertOnFailure) {
      window.alert("Please allow location access so the report can capture the exact upload location.");
    }
  });
}

function clearStoredLocation() {
  state.location.lat = null;
  state.location.lng = null;
  state.location.accuracy = null;
  state.location.timestamp = null;
  state.location.source = null;
  state.location.name = null;
}

function commitLocation(pos, sourceLabel) {
  state.location.lat = pos.coords.latitude;
  state.location.lng = pos.coords.longitude;
  state.location.accuracy = pos.coords.accuracy ?? null;
  state.location.timestamp = Date.now();
  state.location.source = sourceLabel;
  state.location.name = "Location acquired";
  return pos.coords.accuracy ?? null;
}

function getFreshPreciseLocation() {
  const hasRecentLocation =
    state.location.lat != null
    && state.location.lng != null
    && state.location.timestamp != null
    && Date.now() - state.location.timestamp < LOCATION_CACHE_TTL_MS
    && state.location.source !== "fallback";
  if (hasRecentLocation) {
    return Promise.resolve({
      coords: {
        latitude: state.location.lat,
        longitude: state.location.lng,
        accuracy: state.location.accuracy ?? null,
      },
    });
  }
  return requestBestEffortLocation({ forceFresh: false, strictHighAccuracy: true, showAlertOnFailure: false });
}

function requestBestEffortLocation({ forceFresh = false, strictHighAccuracy = false, showAlertOnFailure = false } = {}) {
  if (!navigator.geolocation) {
    return Promise.reject(new Error("Geolocation is not supported by your browser."));
  }

  if (strictHighAccuracy) {
    return requestStrictHighAccuracyLocation({ forceFresh, showAlertOnFailure });
  }

  const attempts = [
    { enableHighAccuracy: true, timeout: 15000, maximumAge: forceFresh ? 0 : LOCATION_CACHE_TTL_MS, label: "gps" },
    ...(strictHighAccuracy ? [] : [
      { enableHighAccuracy: false, timeout: 8000, maximumAge: LOCATION_CACHE_TTL_MS, label: "network" },
    ]),
  ];
  const maxAcceptedAccuracy = strictHighAccuracy ? 250 : Number.POSITIVE_INFINITY;

  const tryAttempt = (index) => new Promise((resolve, reject) => {
    const attempt = attempts[index];
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const accuracy = pos.coords.accuracy ?? null;
        if (strictHighAccuracy && (typeof accuracy !== "number" || !Number.isFinite(accuracy) || accuracy > maxAcceptedAccuracy)) {
          if (index + 1 < attempts.length) {
            tryAttempt(index + 1).then(resolve).catch(reject);
            return;
          }
          reject(new Error("Unable to obtain a precise GPS fix."));
          return;
        }
        state.location.lat = pos.coords.latitude;
        state.location.lng = pos.coords.longitude;
        state.location.accuracy = accuracy;
        state.location.timestamp = Date.now();
        state.location.source = attempt.label;
        state.location.name = "Location acquired";
        updateLocationUI(true, `${formatAccuracyText(accuracy)}${attempt.label === "network" ? " (network)" : ""}`);
        resolve(pos);
      },
      () => {
        if (index + 1 < attempts.length) {
          tryAttempt(index + 1).then(resolve).catch(reject);
          return;
        }
        reject(new Error("Location permission denied or unavailable."));
      },
      attempts[index]
    );
  });

  return tryAttempt(0).catch((error) => {
    if (showAlertOnFailure) {
      window.alert("Please allow location access so the report can capture the exact upload location.");
    }
    throw error;
  });
}

function requestStrictHighAccuracyLocation({ forceFresh = false, showAlertOnFailure = false } = {}) {
  if (!navigator.geolocation) {
    return Promise.reject(new Error("Geolocation is not supported by your browser."));
  }

  const waitMs = forceFresh ? 22000 : 18000;
  const maxAcceptedAccuracy = 120;

  return new Promise((resolve, reject) => {
    let settled = false;
    let watchId = null;
    let timeoutId = null;
    let best = null;

    const finish = (pos) => {
      if (settled) return;
      settled = true;
      if (watchId != null) navigator.geolocation.clearWatch(watchId);
      if (timeoutId != null) window.clearTimeout(timeoutId);
      commitLocation(pos, "gps");
      updateLocationUI(true, `${formatAccuracyText(pos.coords.accuracy)}${(pos.coords.accuracy ?? Number.POSITIVE_INFINITY) > maxAcceptedAccuracy ? " (coarse)" : ""}`);
      resolve(pos);
    };

    watchId = navigator.geolocation.watchPosition(
      (pos) => {
        const accuracy = typeof pos.coords.accuracy === "number" && Number.isFinite(pos.coords.accuracy)
          ? pos.coords.accuracy
          : Number.POSITIVE_INFINITY;
        if (!best || accuracy < best.accuracy) {
          best = { pos, accuracy };
        }
        if (accuracy <= maxAcceptedAccuracy) {
          finish(pos);
        }
      },
      (error) => {
        if (settled) return;
        if (watchId != null) navigator.geolocation.clearWatch(watchId);
        if (timeoutId != null) window.clearTimeout(timeoutId);
        reject(error instanceof Error ? error : new Error("Unable to obtain a precise GPS fix."));
      },
      { enableHighAccuracy: true, maximumAge: forceFresh ? 0 : 0, timeout: 15000 }
    );

    timeoutId = window.setTimeout(() => {
      if (settled) return;
      if (watchId != null) navigator.geolocation.clearWatch(watchId);
      if (best) {
        finish(best.pos);
        return;
      }
      if (showAlertOnFailure) {
        window.alert("Please allow location access so the report can capture the exact upload location.");
      }
      reject(new Error("Unable to obtain a precise GPS fix."));
    }, waitMs);
  });
}

function formatAccuracyText(accuracy) {
  if (typeof accuracy === "number" && Number.isFinite(accuracy)) {
    return `GPS active (${Math.round(accuracy)}m)`;
  }
  return "GPS active";
}

function updateLocationUI(active, text) {
  const dot = document.getElementById("locDot");
  const span = document.getElementById("locationText");
  const btn = document.getElementById("locBtn");
  if (active) {
    dot.classList.add("active");
    btn.textContent = "GPS Active";
  } else {
    dot.classList.remove("active");
    btn.textContent = "Enable GPS";
  }
  span.textContent = text;
}

function getLocationSourceText() {
  if (state.location.source === "gps") return "GPS";
  if (state.location.source === "network") return "Network";
  if (state.location.source === "photo") return "Photo EXIF";
  if (state.location.source === "fallback") return "Fallback";
  return "Unknown";
}

function isNotRecognizedReport(report, data = {}) {
  const status = String(report?.analysis_status || data?.analysis_status || "").toLowerCase();
  const animal = String(report?.animal_type || report?.animal_detected || data?.animal_type || data?.animal_detected || "").toLowerCase();
  return status !== "animal_detected" || animal === "unknown" || animal === "not recognized" || animal === "";
}

function getReportHealthScore(report, data = {}) {
  if (isNotRecognizedReport(report, data)) return null;
  const score = report?.health_score ?? data?.health_score ?? null;
  if (score == null || score === "") return null;
  const numeric = Number(score);
  return Number.isFinite(numeric) ? numeric : null;
}

function handleFileSelect(event) {
  const file = event.target.files?.[0];
  if (file) setPreview(file);
}

function handleDrop(event) {
  event.preventDefault();
  document.getElementById("uploadZone").classList.remove("drag-over");
  const file = event.dataTransfer.files?.[0];
  if (file && file.type.startsWith("image/")) setPreview(file);
}

function setPreview(file) {
  state.currentFile = file;
  const reader = new FileReader();
  reader.onload = (event) => {
    const img = document.getElementById("previewImg");
    img.src = event.target.result;
    img.classList.remove("hidden");
    document.getElementById("uploadPlaceholder").classList.add("hidden");
  };
  reader.readAsDataURL(file);
  document.getElementById("analyseBtn").disabled = false;
}

function resetScanInput() {
  state.currentFile = null;
  document.getElementById("fileInput").value = "";
  const scanAnimalName = document.getElementById("scanAnimalName");
  if (scanAnimalName) scanAnimalName.value = "";
  document.getElementById("previewImg").src = "";
  document.getElementById("previewImg").classList.add("hidden");
  document.getElementById("cameraFeed").classList.add("hidden");
  document.getElementById("uploadPlaceholder").classList.remove("hidden");
  document.getElementById("cameraControls").classList.add("hidden");
  document.getElementById("analyseBtn").disabled = true;
}

async function openCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    const video = document.getElementById("cameraFeed");
    video.srcObject = state.stream;
    video.classList.remove("hidden");
    document.getElementById("uploadPlaceholder").classList.add("hidden");
    document.getElementById("cameraControls").classList.remove("hidden");
  } catch {
    window.alert("Camera access denied or not available.");
  }
}

function capturePhoto() {
  const video = document.getElementById("cameraFeed");
  const canvas = document.getElementById("hiddenCanvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  canvas.toBlob((blob) => {
    if (!blob) return;
    const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
    closeCamera();
    setPreview(file);
  }, "image/jpeg", 0.92);
}

function closeCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  document.getElementById("cameraFeed").classList.add("hidden");
  document.getElementById("cameraControls").classList.add("hidden");
  if (!state.currentFile) {
    document.getElementById("uploadPlaceholder").classList.remove("hidden");
  }
}

async function analyseImage() {
  if (!state.currentFile) return;
  const btn = document.getElementById("analyseBtn");
  btn.disabled = true;
  showResultState("loading");

  try {
    clearStoredLocation();
    updateLocationUI(false, "Detecting current location...");
    try {
      await getFreshPreciseLocation();
    } catch (locationError) {
      clearStoredLocation();
      updateLocationUI(false, "Using photo only");
    }
    const form = new FormData();
    form.append("image", state.currentFile);
    const scanAnimalName = document.getElementById("scanAnimalName")?.value.trim();
    if (scanAnimalName) form.append("animal_name", scanAnimalName);
    if (state.location.lat != null) form.append("location_lat", String(state.location.lat));
    if (state.location.lng != null) form.append("location_long", String(state.location.lng));
    form.append("prefer_current_location", "true");
    form.append("contact_rescue", document.getElementById("rescueCheckbox").checked ? "true" : "false");

    const res = await fetch(`${API}/api/predict`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();
    if (scanAnimalName && !data.animal_name) {
      data.animal_name = scanAnimalName;
    }
    state.lastReport = data;
    state.currentAnimalReportIndex = 0;
    if (data.location_lat != null && data.location_long != null) {
      state.location.lat = Number(data.location_lat);
      state.location.lng = Number(data.location_long);
      state.location.source = "gps";
      state.location.timestamp = Date.now();
    }
    if (data.location_name || data.location_address) {
      state.location.name = data.location_address || data.location_name;
      updateLocationUI(true, `${data.location_name || "Location acquired"} · ${getLocationSourceText()}`);
    }
    renderReport(data);
    upsertReportIntoDashboard(data);
    resetScanInput();
    const rescueLocation = {
      location: [data.location_address, data.location_name].filter(Boolean).join(" - ") || "",
    };
    if (data.location_lat != null && data.location_long != null) {
      rescueLocation.lat = Number(data.location_lat);
      rescueLocation.lng = Number(data.location_long);
    }
    loadRescueContacts(rescueLocation);
  } catch (err) {
    showResultState("idle");
    window.alert(`Analysis failed: ${err.message}`);
  } finally {
    btn.disabled = false;
  }
}

function showResultState(kind) {
  document.getElementById("resultIdle").classList.add("hidden");
  document.getElementById("resultLoading").classList.add("hidden");
  document.getElementById("reportContent").classList.add("hidden");
  if (kind === "idle") document.getElementById("resultIdle").classList.remove("hidden");
  if (kind === "loading") document.getElementById("resultLoading").classList.remove("hidden");
  if (kind === "report") document.getElementById("reportContent").classList.remove("hidden");
}

function closeScanReport() {
  showResultState("idle");
  resetScanInput();
}

function renderReport(data) {
  showResultState("report");
  const preview = document.getElementById("reportPreviewImage");
  const confidenceNote = document.getElementById("reportConfidenceNote");
  if (data.image_url) {
    preview.src = data.image_url;
    preview.classList.remove("hidden");
  } else {
    preview.classList.add("hidden");
  }
  if (confidenceNote) confidenceNote.classList.toggle("hidden", !data.image_url);
  const reports = getAnimalReports(data);
  const activeReport = reports[0] || {};

  const animalName = activeReport.animal_name || data.animal_name || activeReport.animal_type || data.animal_type || "Unknown animal";
  const animalType = activeReport.animal_type || data.animal_type || "Animal";
  const breedGuess = activeReport.breed_guess || data.breed_guess || "";
  const animalDescription = activeReport.animal_description || data.animal_description || "";
  const injuryDescription = activeReport.injury_description || data.injury_description || "";
  const showBreedGuess = Boolean(breedGuess && breedGuess.toLowerCase() !== "unknown");
  setText("reportAnimal", titleCase(animalName));
  setText("reportAnimalType", titleCase(animalType));
  setText("reportBreedGuess", breedGuess ? `Breed guess: ${titleCase(breedGuess)}` : "-");
  setSectionVisibility("reportBreedGuess", showBreedGuess);
  setText("reportDate", formatDateTime(data.created_at || new Date()));

  const score = getReportHealthScore(activeReport, data);
  const healthWord = getHealthWord(score, activeReport.health_status || data.health_status);
  const badge = document.getElementById("statusBadgeLg");
  if (badge) {
    badge.textContent = healthWord;
    badge.className = `status-badge-lg ${scoreClass(score)}`;
  }

  document.getElementById("reportLocation").textContent =
    [data.location_address, data.location_name].filter(Boolean).join(" - ") || "Location unavailable";
  document.getElementById("vitalScore").textContent = healthWord;
  applyScoreStyling(document.getElementById("vitalScore"), score);
  applyScoreCardStyling(document.getElementById("vitalScore")?.closest(".vital-card"), score);
  if (confidenceNote) confidenceNote.textContent = formatAiConfidence(activeReport, data);
  setText("vitalHelpType", score != null ? formatHelpType(activeReport.help_type || data.help_type || "none") : "Not recognized");
  setText("vitalUrgency", score != null ? (activeReport.urgency_label || data.urgency_label || (activeReport.needs_rescue ? "Urgent" : "Non-urgent")) : "Not recognized");
  renderAlertStrip(activeReport, data);
  setText("whatIsWrong", score != null ? (activeReport.what_is_wrong || data.what_is_wrong || "No plain-language problem summary is available.") : "The animal could not be recognized from this image.");
  if (score != null) {
    renderEmergencyBlock("emergency", activeReport.emergency_plan || data.emergency_plan || {});
  } else {
    renderEmergencyBlock("emergency", { summary: "Not recognized as a supported animal.", immediate_steps: [], avoid_steps: [], contact_priority: "Retake a clearer image" });
  }
  setText("conditionSummary", score != null ? (activeReport.condition_summary || data.condition_summary || "No condition summary available.") : "Not recognized as a supported animal.");
  setText("animalDescription", score != null ? (animalDescription || data.animal_description || "No animal description available.") : "No supported animal was recognized in this image.");
  setText("injuryDescription", score != null ? (injuryDescription || data.injury_description || "No injury description available.") : "No supported animal was recognized in this image.");
  setText("bodyCondition", score != null ? (activeReport.body_condition || data.body_condition || "No body condition note available.") : "No supported animal was recognized in this image.");
  setText("healthSummary", score != null ? (activeReport.health_summary || data.health_summary || "No clinical notes available.") : "No medical advice is generated for unrecognized animals.");

  const conditions = Array.isArray(activeReport.detected_conditions)
    ? activeReport.detected_conditions
    : (data.detected_conditions || []);
  if (score == null) {
    renderListSection("findingsSection", "findingsList", []);
    renderListSection("issuesSection", "primaryIssuesList", []);
    renderListSection("symptomsSection", "visibleSymptomsList", []);
    setSectionVisibility("conditionSummary", false);
    setSectionVisibility("animalDescriptionSection", false);
    setSectionVisibility("injuryDescriptionSection", false);
    setSectionVisibility("bodyCondition", false);
    setSectionVisibility("healthSummary", false);
    setSectionVisibility("findingsSection", false);
    setSectionVisibility("issuesSection", false);
    setSectionVisibility("symptomsSection", false);
  } else {
    renderListSection("findingsSection", "findingsList", conditions);
    renderListSection("issuesSection", "primaryIssuesList", activeReport.primary_issues || data.primary_issues || []);
    renderListSection("symptomsSection", "visibleSymptomsList", activeReport.visible_symptoms || data.visible_symptoms || []);
    setSectionVisibility("conditionSummary", true);
    setSectionVisibility("animalDescriptionSection", true);
    setSectionVisibility("injuryDescriptionSection", true);
    setSectionVisibility("bodyCondition", true);
    setSectionVisibility("healthSummary", true);
    setSectionVisibility("findingsSection", true);
    setSectionVisibility("issuesSection", true);
    setSectionVisibility("symptomsSection", true);
  }

  const actionsList = document.getElementById("actionsList");
  actionsList.innerHTML = "";
  const actions = Array.isArray(activeReport.recommended_actions) ? activeReport.recommended_actions : (data.recommended_actions || []);
  actions.forEach((action) => {
    const li = document.createElement("li");
    li.textContent = action;
    actionsList.appendChild(li);
  });
  if (score == null) {
    actionsList.innerHTML = "";
  } else if (!actions.length) {
    const li = document.createElement("li");
    li.textContent = "Observe the animal carefully and contact rescue or a vet if it seems distressed.";
    actionsList.appendChild(li);
  }

}

function renderAlertStrip(activeReport, data) {
  const strip = document.getElementById("reportAlertStrip");
  const title = document.getElementById("reportUrgencyLabel");
  const text = document.getElementById("reportTriageReasoning");
  if (!strip || !title || !text) return;
  const level = String(activeReport.urgency_level || data.urgency_level || "none").replaceAll("_", "-");
  strip.className = `report-alert-strip ${level}`;
  title.textContent = activeReport.urgency_label || data.urgency_label || "No action needed";
  text.textContent = activeReport.triage_reasoning || data.triage_reasoning || "No triage reasoning available.";
}

function renderEmergencyBlock(prefix, emergencyPlan) {
  const section = document.getElementById(`${prefix}Section`);
  const summary = document.getElementById(`${prefix}Summary`);
  const stepsList = document.getElementById(`${prefix}Steps`);
  const avoidList = document.getElementById(`${prefix}Avoid`);
  const priority = document.getElementById(`${prefix}Priority`);
  if (!section || !summary || !stepsList || !avoidList || !priority) return;

  const steps = Array.isArray(emergencyPlan?.immediate_steps) ? emergencyPlan.immediate_steps.filter(Boolean) : [];
  const avoid = Array.isArray(emergencyPlan?.avoid_steps) ? emergencyPlan.avoid_steps.filter(Boolean) : [];
  const hasContent = Boolean(emergencyPlan?.summary) || Boolean(emergencyPlan?.contact_priority) || steps.length || avoid.length;
  section.style.display = hasContent ? "" : "none";
  summary.textContent = emergencyPlan?.summary || "No emergency action is needed right now.";
  priority.textContent = emergencyPlan?.contact_priority ? `Priority: ${emergencyPlan.contact_priority}` : "Priority: Monitor safely";

  stepsList.innerHTML = "";
  steps.forEach((step) => {
    const li = document.createElement("li");
    li.textContent = formatCondition(step);
    stepsList.appendChild(li);
  });
  if (!steps.length) {
    const li = document.createElement("li");
    li.textContent = "Observe the animal from a safe distance and re-scan if conditions change.";
    stepsList.appendChild(li);
  }

  avoidList.innerHTML = "";
  avoid.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = formatCondition(item);
    avoidList.appendChild(li);
  });
  if (!avoid.length) {
    const li = document.createElement("li");
    li.textContent = "Avoid sudden handling or crowding.";
    avoidList.appendChild(li);
  }
}

function renderEmergencyMarkup(emergencyPlan, prefix) {
  const steps = Array.isArray(emergencyPlan?.immediate_steps) ? emergencyPlan.immediate_steps.filter(Boolean) : [];
  const avoid = Array.isArray(emergencyPlan?.avoid_steps) ? emergencyPlan.avoid_steps.filter(Boolean) : [];
  const summary = emergencyPlan?.summary || "No emergency action is needed right now.";
  const priority = emergencyPlan?.contact_priority || "Monitor safely";
  const stepItems = (steps.length ? steps : ["Observe the animal from a safe distance and re-scan if conditions change."])
    .map((step) => `<li>${esc(step)}</li>`)
    .join("");
  const avoidItems = (avoid.length ? avoid : ["Avoid sudden handling or crowding."])
    .map((item) => `<li>${esc(item)}</li>`)
    .join("");
  return `
    <div class="report-section emergency-section" id="${esc(prefix)}EmergencySection">
      <h4 class="section-heading">Emergency SOS</h4>
      <p class="report-text" id="${esc(prefix)}EmergencySummary">${esc(summary)}</p>
      <div class="emergency-grid">
        <div class="emergency-box">
          <h5>Do Now</h5>
          <ol class="actions-list" id="${esc(prefix)}EmergencySteps">${stepItems}</ol>
        </div>
        <div class="emergency-box">
          <h5>Avoid</h5>
          <ul class="findings-list" id="${esc(prefix)}EmergencyAvoid">${avoidItems}</ul>
        </div>
      </div>
      <p class="modal-meta-note" id="${esc(prefix)}EmergencyPriority">Priority: ${esc(priority)}</p>
    </div>
  `;
}

function renderListSection(sectionId, listId, values) {
  const section = document.getElementById(sectionId);
  const list = document.getElementById(listId);
  if (!section || !list) return;
  const items = Array.isArray(values) ? values.filter(Boolean) : [];
  list.innerHTML = "";
  if (!items.length) {
    section.style.display = "none";
    return;
  }
  items.forEach((value) => {
    const li = document.createElement("li");
    li.textContent = formatCondition(value);
    list.appendChild(li);
  });
  section.style.display = "";
}

function getAnimalReports(data) {
  const source = (Array.isArray(data.animal_reports) && data.animal_reports.length ? data.animal_reports[0] : data) || {};
  return [{
    ...source,
    animal_type: source.animal_type || data.animal_type,
    animal_detected: source.animal_detected || data.animal_detected || source.animal_type || data.animal_type,
    health_status: normalizeHealthStatus(source.health_status || data.health_status),
    health_confidence: source.health_confidence ?? data.health_confidence,
    health_score: source.health_score ?? data.health_score,
    detection_confidence: source.detection_confidence ?? data.detection_confidence,
    bounding_box: source.bounding_box || data.bounding_box || {
      x1: source.bbox_x1 ?? data.bbox_x1,
      y1: source.bbox_y1 ?? data.bbox_y1,
      x2: source.bbox_x2 ?? data.bbox_x2,
      y2: source.bbox_y2 ?? data.bbox_y2
    },
    detected_conditions: Array.isArray(source.detected_conditions) ? source.detected_conditions : (Array.isArray(data.detected_conditions) ? data.detected_conditions : []),
    guidance: source.guidance || data.guidance,
    condition_summary: source.condition_summary || data.condition_summary,
    recommended_actions: Array.isArray(source.recommended_actions) ? source.recommended_actions : (Array.isArray(data.recommended_actions) ? data.recommended_actions : []),
    needs_rescue: source.needs_rescue ?? data.needs_rescue,
    health_summary: source.health_summary || data.health_summary,
    breed_guess: source.breed_guess || data.breed_guess,
    animal_description: source.animal_description || data.animal_description,
    injury_description: source.injury_description || data.injury_description,
    urgency_level: source.urgency_level || data.urgency_level,
    urgency_label: source.urgency_label || data.urgency_label,
    primary_issues: Array.isArray(source.primary_issues) ? source.primary_issues : (Array.isArray(data.primary_issues) ? data.primary_issues : []),
    visible_symptoms: Array.isArray(source.visible_symptoms) ? source.visible_symptoms : (Array.isArray(data.visible_symptoms) ? data.visible_symptoms : []),
    body_condition: source.body_condition || data.body_condition,
    what_is_wrong: source.what_is_wrong || data.what_is_wrong,
    help_type: source.help_type || data.help_type,
    triage_reasoning: source.triage_reasoning || data.triage_reasoning,
    emergency_plan: source.emergency_plan || data.emergency_plan || {},
    avoid_steps: Array.isArray(source.avoid_steps) ? source.avoid_steps : (Array.isArray(data.avoid_steps) ? data.avoid_steps : []),
    contact_priority: source.contact_priority || data.contact_priority || ""
  }];
}

function bindChat() {
  document.getElementById("newChatBtn").addEventListener("click", () => newChatThread(true));
  document.getElementById("sendBtn").addEventListener("click", sendMessage);
  document.getElementById("toggleSidebarBtn").addEventListener("click", toggleDesktopCarebotSidebar);
  document.getElementById("renameChatBtn").addEventListener("click", renameCurrentThread);
  document.getElementById("cancelEditBtn").addEventListener("click", cancelEditMessage);
  document.getElementById("chatDrawerBtn").addEventListener("click", toggleCarebotDrawer);
  document.getElementById("chatDrawerBackdrop").addEventListener("click", closeCarebotDrawer);
  const input = document.getElementById("chatInput");
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });
  input.addEventListener("input", () => autoResize(input));
  document.querySelectorAll("[data-chip]").forEach((chip) => {
    chip.addEventListener("click", () => sendChip(chip.dataset.chip));
  });
}

function loadCareBotHistory() {
  try {
    state.threads = JSON.parse(localStorage.getItem(THREAD_KEY) || "[]");
  } catch {
    state.threads = [];
  }
  if (state.threads.length === 0) newChatThread(false);
  else activateThread(state.threads[0].id);
  renderThreadList();
}

function saveCareBotHistory() {
  localStorage.setItem(THREAD_KEY, JSON.stringify(state.threads));
}

function newChatThread(activate = true) {
  const thread = { id: String(Date.now()), title: "New conversation", messages: [] };
  state.threads.unshift(thread);
  saveCareBotHistory();
  renderThreadList();
  if (activate) activateThread(thread.id);
}

function activateThread(id) {
  state.activeThread = id;
  state.editingMessageIndex = null;
  syncEditBanner();
  renderThreadList();
  renderMessages();
  closeCarebotDrawer();
}

function deleteThread(id) {
  state.threads = state.threads.filter((thread) => thread.id !== id);
  if (state.threads.length === 0) newChatThread(false);
  if (state.activeThread === id) state.activeThread = state.threads[0].id;
  saveCareBotHistory();
  renderThreadList();
  renderMessages();
}

function renderThreadList() {
  const list = document.getElementById("chatThreadList");
  list.innerHTML = "";
  state.threads.forEach((thread) => {
    const div = document.createElement("div");
    div.className = `chat-thread${thread.id === state.activeThread ? " active" : ""}`;
    div.innerHTML = `
      <span class="chat-thread-title">${esc(thread.title)}</span>
      <button class="chat-thread-rename" type="button">Rename</button>
      <button class="chat-thread-del" type="button">X</button>
    `;
    div.addEventListener("click", () => activateThread(thread.id));
    div.querySelector(".chat-thread-rename").addEventListener("click", (event) => {
      event.stopPropagation();
      renameThread(thread.id);
    });
    div.querySelector(".chat-thread-del").addEventListener("click", (event) => {
      event.stopPropagation();
      deleteThread(thread.id);
    });
    list.appendChild(div);
  });
}

function getActiveThread() {
  return state.threads.find((thread) => thread.id === state.activeThread);
}

function renderMessages() {
  const thread = getActiveThread();
  const container = document.getElementById("chatMessages");
  container.innerHTML = `
    <div class="message bot">
      <div class="msg-avatar">CB</div>
      <div class="message-body">
        <div class="msg-bubble">
          Hello. I am Care Bot. I can help with domestic animal first aid, care questions, warning signs, and what to do next before reaching a vet.
        </div>
      </div>
    </div>
  `;
  if (!thread) return;
  thread.messages.forEach((message, index) => appendMessage(message.role, message.content, false, index));
  scrollChatToBottom();
}

function appendMessage(role, content, scroll = true, index = -1) {
  const container = document.getElementById("chatMessages");
  const div = document.createElement("div");
  div.className = `message ${role === "assistant" ? "bot" : "user"}`;
  const actionHtml = role === "user"
    ? `
      <div class="message-actions">
        <button class="message-action" type="button" data-edit-index="${index}" title="Edit" aria-label="Edit">
          <svg viewBox="0 0 24 24" class="message-action-icon" aria-hidden="true">
            <path d="M4 16.5V20h3.5L18.8 8.7a1.5 1.5 0 0 0 0-2.1l-1.4-1.4a1.5 1.5 0 0 0-2.1 0L4 16.5z"></path>
          </svg>
        </button>
        <button class="message-action" type="button" data-regenerate-from-index="${index}" title="Regenerate" aria-label="Regenerate">
          <svg viewBox="0 0 24 24" class="message-action-icon" aria-hidden="true">
            <path d="M6 12a6 6 0 0 1 10-4.5L18 6v5h-5l1.9-1.9A4 4 0 1 0 16 15h2.2A6 6 0 0 1 6 12z"></path>
          </svg>
        </button>
        <button class="message-action" type="button" data-copy-content="${index}" data-copy-role="user" title="Copy" aria-label="Copy">
          <svg viewBox="0 0 24 24" class="message-action-icon" aria-hidden="true">
            <path d="M8 8h10v10H8z"></path>
            <path d="M6 16V6h10" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"></path>
          </svg>
        </button>
      </div>
    `
    : `
      <div class="message-actions">
        <button class="message-action" type="button" data-copy-content="${index}" data-copy-role="assistant" title="Copy" aria-label="Copy">
          <svg viewBox="0 0 24 24" class="message-action-icon" aria-hidden="true">
            <path d="M8 8h10v10H8z"></path>
            <path d="M6 16V6h10" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"></path>
          </svg>
        </button>
      </div>
    `;
  div.innerHTML = `
    <div class="msg-avatar">${role === "assistant" ? "CB" : "U"}</div>
    <div class="message-body">
      <div class="msg-bubble">${formatBotText(content)}</div>
      ${actionHtml}
    </div>
  `;
  container.appendChild(div);
  const editButton = div.querySelector("[data-edit-index]");
  if (editButton) {
    editButton.addEventListener("click", () => startEditMessage(index));
  }
  const regenerateButton = div.querySelector("[data-regenerate-from-index]");
  if (regenerateButton) {
    regenerateButton.addEventListener("click", () => regenerateFromUserMessage(index));
  }
  div.querySelectorAll("[data-copy-content]").forEach((button) => {
    button.addEventListener("click", async () => {
      const roleName = button.getAttribute("data-copy-role");
      await copyMessageContent(index, roleName);
    });
  });
  if (scroll) scrollChatToBottom();
  return div;
}

function showTyping() {
  const container = document.getElementById("chatMessages");
  const div = document.createElement("div");
  div.className = "message bot";
  div.id = "typingIndicator";
  div.innerHTML = `
    <div class="msg-avatar">CB</div>
    <div class="message-body">
      <div class="msg-bubble"><span class="typing-dots"><span></span><span></span><span></span></span></div>
    </div>
  `;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function removeTyping() {
  document.getElementById("typingIndicator")?.remove();
}

async function sendMessage() {
  const input = document.getElementById("chatInput");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  input.style.height = "auto";

  const thread = getActiveThread();
  if (!thread) return;

  if (state.editingMessageIndex != null) {
    thread.messages[state.editingMessageIndex].content = text;
    thread.messages = thread.messages.slice(0, state.editingMessageIndex + 1);
    state.editingMessageIndex = null;
    syncEditBanner();
    saveCareBotHistory();
    renderMessages();
  } else {
    if (thread.messages.length === 0) {
      thread.title = text.slice(0, 38) + (text.length > 38 ? "..." : "");
    }
    thread.messages.push({ role: "user", content: text });
    saveCareBotHistory();
    appendMessage("user", text, true, thread.messages.length - 1);
    renderThreadList();
    document.getElementById("quickChips").style.display = "none";
  }

  showTyping();
  try {
    const reply = await callCareBotAPI(thread.messages);
    removeTyping();
    thread.messages.push({ role: "assistant", content: reply });
    saveCareBotHistory();
    appendMessage("assistant", reply, true, thread.messages.length - 1);
    renderThreadList();
  } catch {
    removeTyping();
    thread.messages.push({ role: "assistant", content: "Care Bot is unavailable right now. Please check the backend and try again." });
    saveCareBotHistory();
    appendMessage("assistant", "Care Bot is unavailable right now. Please check the backend and try again.", true, thread.messages.length - 1);
  }
}

function sendChip(text) {
  document.getElementById("chatInput").value = text;
  sendMessage();
}

function autoResize(el) {
  el.style.height = "auto";
  el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
}

async function callCareBotAPI(messages) {
  const payload = {
    messages,
    animal_type: state.lastReport?.animal_type || null,
    health_status: state.lastReport?.health_status || null,
    detected_conditions: state.lastReport?.detected_conditions || [],
    location_name: state.lastReport?.location_name || state.location.name || null
  };

  const res = await fetch(`${API}/api/medical-chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error(`Backend error ${res.status}`);
  const data = await res.json();
  return data.reply || "No response received.";
}

function askCareBotAboutReport() {
  if (state.lastReport) {
    const report = state.lastReport;
      const intro = `I just scanned a ${report.animal_detected || report.animal_type || "domestic animal"}. Health score is ${report.health_score ?? "unknown"}%. ${report.condition_summary || ""} What should I do next?`;
      showPage("carebot");
      document.getElementById("chatInput").value = intro;
      autoResize(document.getElementById("chatInput"));
    } else {
      showPage("carebot");
    }
}

function startEditMessage(index) {
  const thread = getActiveThread();
  if (!thread || index < 0 || !thread.messages[index] || thread.messages[index].role !== "user") return;
  state.editingMessageIndex = index;
  const input = document.getElementById("chatInput");
  input.value = thread.messages[index].content;
  syncEditBanner();
  autoResize(input);
  input.focus();
}

function cancelEditMessage() {
  state.editingMessageIndex = null;
  document.getElementById("chatInput").value = "";
  syncEditBanner();
}

function syncEditBanner() {
  const banner = document.getElementById("editBanner");
  const text = document.getElementById("editBannerText");
  const sendBtn = document.getElementById("sendBtn");
  if (state.editingMessageIndex == null) {
    banner.classList.add("hidden");
    sendBtn.textContent = "Send";
    return;
  }
  banner.classList.remove("hidden");
  text.textContent = "Editing previous message";
  sendBtn.textContent = "Save";
}

function renameCurrentThread() {
  if (!state.activeThread) return;
  renameThread(state.activeThread);
}

function renameThread(id) {
  const thread = state.threads.find((item) => item.id === id);
  if (!thread) return;
  const nextName = window.prompt("Rename this chat", thread.title);
  if (!nextName) return;
  thread.title = nextName.trim() || thread.title;
  saveCareBotHistory();
  renderThreadList();
}

async function regenerateLastResponse() {
  const thread = getActiveThread();
  if (!thread) return;
  if (!thread.messages.length) return;
  if (thread.messages[thread.messages.length - 1]?.role === "assistant") {
    thread.messages.pop();
  }
  const hasUser = thread.messages.some((message) => message.role === "user");
  if (!hasUser) return;
  saveCareBotHistory();
  renderMessages();
  showTyping();
  try {
    const reply = await callCareBotAPI(thread.messages);
    removeTyping();
    thread.messages.push({ role: "assistant", content: reply });
    saveCareBotHistory();
    appendMessage("assistant", reply);
  } catch {
    removeTyping();
    appendMessage("assistant", "Care Bot could not regenerate the reply right now.");
  }
}

async function regenerateFromUserMessage(index) {
  const thread = getActiveThread();
  if (!thread) return;
  if (index < 0 || index >= thread.messages.length) return;
  if (thread.messages[index].role !== "user") return;
  thread.messages = thread.messages.slice(0, index + 1);
  saveCareBotHistory();
  renderMessages();
  showTyping();
  try {
    const reply = await callCareBotAPI(thread.messages);
    removeTyping();
    thread.messages.push({ role: "assistant", content: reply });
    saveCareBotHistory();
    appendMessage("assistant", reply, true, thread.messages.length - 1);
    renderThreadList();
  } catch {
    removeTyping();
    thread.messages.push({ role: "assistant", content: "Care Bot could not regenerate the reply right now." });
    saveCareBotHistory();
    appendMessage("assistant", "Care Bot could not regenerate the reply right now.", true, thread.messages.length - 1);
  }
}

async function copyMessageContent(index, roleName) {
  const thread = getActiveThread();
  if (!thread) return;
  const direct = thread.messages[index];
  const content = direct?.role === roleName ? direct.content : null;
  if (!content) return;
  try {
    await navigator.clipboard.writeText(content);
  } catch {
    window.prompt("Copy this message", content);
  }
}

function scrollChatToBottom() {
  const container = document.getElementById("chatMessages");
  container.scrollTop = container.scrollHeight;
}

function toggleCarebotDrawer() {
  const sidebar = document.getElementById("carebotSidebar");
  const backdrop = document.getElementById("chatDrawerBackdrop");
  sidebar.classList.toggle("open");
  backdrop.classList.toggle("open");
}

function closeCarebotDrawer() {
  document.getElementById("carebotSidebar").classList.remove("open");
  document.getElementById("chatDrawerBackdrop").classList.remove("open");
}

function toggleDesktopCarebotSidebar() {
  const layout = document.querySelector(".carebot-layout");
  if (!layout) return;
  if (window.innerWidth <= 980) {
    toggleCarebotDrawer();
    return;
  }
  layout.classList.toggle("sidebar-collapsed");
  syncSidebarChevron();
}

function syncSidebarChevron() {
  const layout = document.querySelector(".carebot-layout");
  const btn = document.getElementById("toggleSidebarBtn");
  if (!layout || !btn || window.innerWidth <= 980) return;
  btn.textContent = layout.classList.contains("sidebar-collapsed") ? ">" : "<";
}

function bindDashboard() {
  document.getElementById("filterAnimal").addEventListener("change", () => renderReportCards(getFilteredReports()));
  document.getElementById("filterStatus").addEventListener("change", () => renderReportCards(getFilteredReports()));
  document.getElementById("filterDate").addEventListener("change", () => renderReportCards(getFilteredReports()));
  document.getElementById("clearFiltersBtn").addEventListener("click", clearDashboardFilters);
  document.getElementById("deleteAllBtn").addEventListener("click", deleteAllReports);
  document.getElementById("selectAllReportsBtn")?.addEventListener("click", toggleSelectAllVisibleReports);
  document.getElementById("deleteSelectedBtn")?.addEventListener("click", deleteSelectedReports);
  document.getElementById("dashboardCardsBtn")?.addEventListener("click", () => setDashboardView("cards"));
  document.getElementById("dashboardTableBtn")?.addEventListener("click", () => setDashboardView("table"));
  syncDashboardViewButtons();
  syncBulkActionButtons();
}

async function loadReports() {
  const requestSeq = ++state.reportLoadSeq;
  try {
    const res = await fetch(`${API}/api/reports`);
    const reports = await res.json();
    if (requestSeq !== state.reportLoadSeq) return;
    state.reports = Array.isArray(reports) ? reports : [];
    syncSelectedReportIds();
    renderStats(expandReportEntries(state.reports));
    renderReportCards(getFilteredReports());
  } catch {
    if (requestSeq !== state.reportLoadSeq) return;
    document.getElementById("reportCards").innerHTML = `<p class="loading-msg">Could not load reports. Is the backend running?</p>`;
  }
}

async function loadDbHealth() {
  const pill = document.getElementById("dbStatusPill");
  if (!pill) return;
  try {
    const res = await fetch(`${API}/api/db-health`);
    if (!res.ok) throw new Error(`Status ${res.status}`);
    const data = await res.json();
    pill.className = "db-status-pill connected";
    pill.textContent = `MySQL connected · ${data.report_count ?? 0} reports`;
  } catch (error) {
    pill.className = "db-status-pill disconnected";
    pill.textContent = "MySQL not connected";
  }
}

function upsertReportIntoDashboard(report) {
  if (!report || report.report_id == null) return;
  const reportId = Number(report.report_id);
  const nextReport = { ...report, report_id: reportId };
  const exists = state.reports.some((item) => Number(item.report_id) === reportId);
  state.reports = exists
    ? state.reports.map((item) => (Number(item.report_id) === reportId ? { ...item, ...nextReport } : item))
    : [nextReport, ...state.reports];
  syncSelectedReportIds();
  renderStats(expandReportEntries(state.reports));
  renderReportCards(getFilteredReports());
}

function syncSelectedReportIds() {
  const validIds = new Set(state.reports.map((report) => Number(report.report_id)));
  state.selectedReportIds = new Set([...state.selectedReportIds].filter((reportId) => validIds.has(Number(reportId))));
  syncBulkActionButtons();
}

function isReportSelected(reportId) {
  return state.selectedReportIds.has(Number(reportId));
}

function setReportSelected(reportId, selected) {
  const id = Number(reportId);
  if (selected) state.selectedReportIds.add(id);
  else state.selectedReportIds.delete(id);
  syncBulkActionButtons();
  renderReportCards(getFilteredReports());
}

function toggleSelectAllVisibleReports() {
  const visibleReports = getFilteredReports();
  if (!visibleReports.length) return;
  const allSelected = visibleReports.every((report) => isReportSelected(report.parent_report_id || report.report_id));
  if (allSelected) {
    visibleReports.forEach((report) => state.selectedReportIds.delete(Number(report.parent_report_id || report.report_id)));
  } else {
    visibleReports.forEach((report) => state.selectedReportIds.add(Number(report.parent_report_id || report.report_id)));
  }
  syncBulkActionButtons();
  renderReportCards(visibleReports);
}

function syncBulkActionButtons() {
  const deleteSelectedBtn = document.getElementById("deleteSelectedBtn");
  const selectAllBtn = document.getElementById("selectAllReportsBtn");
  const selectedCount = state.selectedReportIds.size;
  if (deleteSelectedBtn) {
    deleteSelectedBtn.disabled = selectedCount === 0;
    deleteSelectedBtn.textContent = selectedCount > 0 ? `Delete Selected (${selectedCount})` : "Delete Selected";
  }
  if (selectAllBtn) {
    const visibleReports = getFilteredReports();
    const allVisibleSelected = visibleReports.length > 0 && visibleReports.every((report) => isReportSelected(report.parent_report_id || report.report_id));
    selectAllBtn.textContent = allVisibleSelected ? "Clear Selection" : "Select All";
  }
}

async function deleteSelectedReports() {
  const selectedIds = [...state.selectedReportIds];
  if (!selectedIds.length) return;
  if (!window.confirm(`Delete ${selectedIds.length} selected report${selectedIds.length === 1 ? "" : "s"}?`)) return;
  state.reports = state.reports.filter((report) => !selectedIds.includes(Number(report.report_id)));
  selectedIds.forEach((reportId) => state.selectedReportIds.delete(Number(reportId)));
  renderStats(expandReportEntries(state.reports));
  renderReportCards(getFilteredReports());
  const results = await Promise.allSettled(selectedIds.map((id) => fetch(`${API}/api/reports/${Number(id)}`, { method: "DELETE" })));
  const failed = results.some((result) => result.status === "rejected" || !result.value?.ok);
  if (failed) {
    await loadReports();
  }
}

function isMobileDashboard() {
  return window.matchMedia("(max-width: 860px)").matches;
}

function getActiveDashboardView() {
  return state.dashboardView === "table" ? "table" : "cards";
}

function setDashboardView(view) {
  state.dashboardView = view === "table" ? "table" : "cards";
  window.localStorage?.setItem("paw-connect-dashboard-view", state.dashboardView);
  syncDashboardViewButtons();
  renderReportCards(getFilteredReports());
}

function syncDashboardViewButtons() {
  const activeView = getActiveDashboardView();
  const cardsBtn = document.getElementById("dashboardCardsBtn");
  const tableBtn = document.getElementById("dashboardTableBtn");
  if (cardsBtn) {
    cardsBtn.classList.toggle("active", activeView === "cards");
    cardsBtn.setAttribute("aria-pressed", String(activeView === "cards"));
  }
  if (tableBtn) {
    tableBtn.classList.toggle("active", activeView === "table");
    tableBtn.setAttribute("aria-pressed", String(activeView === "table"));
  }
}

function getFilteredReports() {
  const animal = document.getElementById("filterAnimal")?.value || "";
  const status = document.getElementById("filterStatus")?.value || "";
  const date = document.getElementById("filterDate")?.value || "";
  return expandReportEntries(state.reports).filter((entry) => {
    const animalOk = !animal || (entry.animal_type || "").toLowerCase() === animal.toLowerCase();
    const reportStatus = normalizeHealthStatus(entry.health_status);
    const statusOk = !status || reportStatus === status;
    const dateOk = !date || String(entry.created_at || "").startsWith(date);
    return animalOk && statusOk && dateOk;
  });
}

function renderStats(reports) {
  const total = reports.length;
  const serious = reports.filter((r) => normalizeHealthStatus(r.health_status) === "Seriously Injured").length;
  const mild = reports.filter((r) => normalizeHealthStatus(r.health_status) === "Mildly Injured").length;
  const healthy = reports.filter((r) => normalizeHealthStatus(r.health_status) === "Healthy").length;
  const unknown = reports.filter((r) => getReportHealthScore(r) == null).length;
  setNum("statTotal", total);
  setNum("statSerious", serious);
  setNum("statMild", mild);
  setNum("statHealthy", healthy);
  setNum("statUnknown", unknown);
}

function renderReportCards(reports) {
  const container = document.getElementById("reportCards");
  const view = getActiveDashboardView();
  syncDashboardViewButtons();
  if (!reports.length) {
    container.innerHTML = `<div class="empty-state"><span>PAW</span><p>No reports found.</p></div>`;
    syncBulkActionButtons();
    return;
  }
  if (view === "table") {
    container.className = "report-cards report-table-shell";
    container.innerHTML = `
      <table class="report-table">
        <thead>
          <tr>
            <th class="report-select-head">Select</th>
            <th>Re.No</th>
            <th>Animal Name</th>
            <th>Location</th>
            <th>Date Time</th>
            <th>Health Type</th>
            <th>Delete</th>
          </tr>
        </thead>
        <tbody>
          ${reports.map((report, index) => renderReportTableRow(report, index + 1)).join("")}
        </tbody>
      </table>
    `;
    bindDashboardTableInteractions(container);
    return;
  }

  container.className = "report-cards report-cards-grid";
  container.innerHTML = reports.map((report) => renderReportCard(report)).join("");
  bindDashboardCardInteractions(container);
  syncBulkActionButtons();
}

function renderReportCard(report) {
  const score = getReportHealthScore(report);
  const displayName = titleCase(report.animal_name || report.animal_detected || report.animal_type || "Unknown animal");
  const animalType = titleCase(report.animal_type || report.animal_detected || "Animal");
  const statusLabel = getHealthWord(score, report.health_status);
  const locationLabel = report.location_name || report.location_address || "No location";
  const reportId = report.parent_report_id || report.report_id;
  return `
    <article class="report-card${isReportSelected(reportId) ? " selected" : ""}" data-report-id="${esc(reportId)}">
      <label class="report-select-chip" title="Select report">
        <input class="report-select-checkbox" type="checkbox" data-report-select="${esc(reportId)}" ${isReportSelected(reportId) ? "checked" : ""}>
        <span>Select</span>
      </label>
      ${report.image_url ? `<img class="report-preview-image report-card-image" src="${esc(report.image_url)}" alt="${esc(displayName)}">` : ""}
      <div class="report-card-body">
        <input
          class="report-name-input report-name-input-card"
          data-report-name-input="${esc(reportId)}"
          value="${esc(report.animal_name || "")}"
          placeholder="${esc(displayName)}"
          aria-label="Edit animal name"
        >
        <p class="report-card-type">${esc(animalType)}</p>
        <div class="report-meta">
          <span class="pill ${esc(scoreClass(score))}">${esc(statusLabel)}</span>
          <span class="pill">${esc(locationLabel)}</span>
          <span class="pill">${esc(formatDateTime(report.created_at))}</span>
        </div>
      </div>
      <div class="report-actions">
        <button class="mini-btn danger icon-only report-delete-btn" type="button" title="Delete report" aria-label="Delete report">
          <svg viewBox="0 0 24 24" class="bin-icon" aria-hidden="true">
            <path d="M9 4.5h6l1 1.5H20v2H4v-2h4z"></path>
            <path d="M7 9h10l-.8 10.2a1.5 1.5 0 0 1-1.5 1.3H9.3a1.5 1.5 0 0 1-1.5-1.3z"></path>
            <path d="M10 11.2v6.2M14 11.2v6.2" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"></path>
          </svg>
        </button>
      </div>
    </article>
  `;
}

function renderReportTableRow(report, index) {
  const score = getReportHealthScore(report);
  const displayName = titleCase(report.animal_name || report.animal_detected || report.animal_type || "Unknown animal");
  const animalType = titleCase(report.animal_type || report.animal_detected || "Animal");
  const statusLabel = getHealthWord(score, report.health_status);
  const locationLabel = report.location_name || report.location_address || "No location";
  const reportId = report.parent_report_id || report.report_id;
  return `
    <tr data-report-id="${esc(reportId)}" class="${isReportSelected(reportId) ? "selected-row" : ""}">
      <td class="report-select-cell">
        <input class="report-select-checkbox" type="checkbox" data-report-select="${esc(reportId)}" ${isReportSelected(reportId) ? "checked" : ""}>
      </td>
      <td class="report-index">${esc(index)}</td>
      <td class="report-name-cell">
        <input class="report-name-input" data-report-name-input="${esc(reportId)}" value="${esc(report.animal_name || "")}" placeholder="${esc(displayName)}" aria-label="Edit animal name">
        <div class="report-card-type">${esc(animalType)}</div>
      </td>
      <td>${esc(locationLabel)}</td>
      <td>${esc(formatDateTime(report.created_at))}</td>
      <td><span class="pill ${esc(scoreClass(score))}">${esc(statusLabel)}</span></td>
      <td>
        <button class="mini-btn danger icon-only report-delete-btn" type="button" title="Delete report" aria-label="Delete report">
          <svg viewBox="0 0 24 24" class="bin-icon" aria-hidden="true">
            <path d="M9 4.5h6l1 1.5H20v2H4v-2h4z"></path>
            <path d="M7 9h10l-.8 10.2a1.5 1.5 0 0 1-1.5 1.3H9.3a1.5 1.5 0 0 1-1.5-1.3z"></path>
            <path d="M10 11.2v6.2M14 11.2v6.2" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"></path>
          </svg>
        </button>
      </td>
    </tr>
  `;
}

function bindDashboardCardInteractions(container) {
  container.querySelectorAll(".report-card").forEach((card) => {
    const reportId = Number(card.dataset.reportId);
    card.addEventListener("click", (event) => {
      if (event.target.closest("input, button, label")) return;
      openStoredReport(findExpandedReport(reportId));
    });
    card.querySelector(".report-delete-btn")?.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteReport(reportId);
    });
    const selectBox = card.querySelector("[data-report-select]");
    if (selectBox) {
      selectBox.addEventListener("click", (event) => event.stopPropagation());
      selectBox.addEventListener("change", () => setReportSelected(reportId, selectBox.checked));
    }
    const nameInput = card.querySelector("[data-report-name-input]");
    if (nameInput) {
      nameInput.addEventListener("click", (event) => event.stopPropagation());
      nameInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          nameInput.blur();
        }
      });
      nameInput.addEventListener("blur", () => updateReportAnimalName(reportId, nameInput.value));
    }
  });
}

function bindDashboardTableInteractions(container) {
  container.querySelectorAll("tr[data-report-id]").forEach((row) => {
    const reportId = Number(row.dataset.reportId);
    row.addEventListener("click", (event) => {
      if (event.target.closest("input, button")) return;
      openStoredReport(findExpandedReport(reportId));
    });
    const selectBox = row.querySelector("[data-report-select]");
    if (selectBox) {
      selectBox.addEventListener("click", (event) => event.stopPropagation());
      selectBox.addEventListener("change", () => setReportSelected(reportId, selectBox.checked));
    }
    row.querySelector(".report-delete-btn")?.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteReport(reportId);
    });
    const nameInput = row.querySelector("[data-report-name-input]");
    if (nameInput) {
      nameInput.addEventListener("click", (event) => event.stopPropagation());
      nameInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          nameInput.blur();
        }
      });
      nameInput.addEventListener("blur", () => updateReportAnimalName(reportId, nameInput.value));
    }
  });
}

function findExpandedReport(reportId) {
  return getFilteredReports().find((report) => Number(report.parent_report_id || report.report_id) === Number(reportId))
    || expandReportEntries(state.reports).find((report) => Number(report.parent_report_id || report.report_id) === Number(reportId))
    || null;
}

async function updateReportAnimalName(reportId, nextName) {
  const trimmed = String(nextName || "").trim();
  const report = state.reports.find((item) => Number(item.report_id) === Number(reportId));
  if (!report) return;
  const currentName = String(report.animal_name || "").trim();
  if (trimmed === currentName) return;
  const previousReports = state.reports.map((item) => ({ ...item }));
  state.reports = state.reports.map((item) => Number(item.report_id) === Number(reportId)
    ? { ...item, animal_name: trimmed || null }
    : item);
  renderStats(expandReportEntries(state.reports));
  renderReportCards(getFilteredReports());
  try {
    const res = await fetch(`${API}/api/reports/${reportId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ animal_name: trimmed || null }),
    });
    if (!res.ok) throw new Error(`Status ${res.status}`);
    const updated = await res.json();
    state.reports = state.reports.map((item) => Number(item.report_id) === Number(reportId)
      ? { ...item, ...updated, animal_name: updated.animal_name ?? null }
      : item);
    renderStats(expandReportEntries(state.reports));
    renderReportCards(getFilteredReports());
  } catch (error) {
    state.reports = previousReports;
    renderStats(expandReportEntries(state.reports));
    renderReportCards(getFilteredReports());
    console.warn("Could not rename report.", error);
  }
}

function openStoredReport(report) {
  if (!report) return;
  state.modalReport = buildStoredReportPayload(report.source_report || report);
  state.modalAnimalReportIndex = 0;
  renderStoredReportModal();
  document.getElementById("reportModal").classList.remove("hidden");
}

async function deleteReport(id) {
  if (!window.confirm("Delete this report?")) return;
  const reportId = Number(id);
  state.reports = state.reports.filter((report) => Number(report.report_id) !== reportId);
  state.selectedReportIds.delete(reportId);
  renderStats(expandReportEntries(state.reports));
  renderReportCards(getFilteredReports());
  try {
    const res = await fetch(`${API}/api/reports/${reportId}`, { method: "DELETE" });
    if (!res.ok) {
      await loadReports();
    }
  } catch (error) {
    await loadReports();
    console.warn("Could not delete report.", error);
  }
}

async function deleteAllReports() {
  if (!window.confirm("Delete all reports?")) return;
  state.reports = [];
  state.selectedReportIds.clear();
  renderStats([]);
  renderReportCards([]);
  try {
    const res = await fetch(`${API}/api/reports`, { method: "DELETE" });
    if (!res.ok) {
      await loadReports();
    }
  } catch (error) {
    await loadReports();
    console.warn("Could not delete all reports.", error);
  }
}

function clearDashboardFilters() {
  document.getElementById("filterAnimal").value = "";
  document.getElementById("filterStatus").value = "";
  document.getElementById("filterDate").value = "";
  renderReportCards(getFilteredReports());
}

function expandReportEntries(reports) {
  return reports.map((report) => {
    const payload = buildStoredReportPayload(report);
    return {
      ...payload,
      parent_report_id: report.report_id,
      image_path: report.image_path,
      image_url: report.image_path ? `/${buildReportImageUrl(report.image_path)}` : "",
      source_report: report
    };
  });
}

function buildStoredReportPayload(report) {
  const normalizedStatus = normalizeHealthStatus(report.health_status);
  const imageUrl = report.image_path ? `/${buildReportImageUrl(report.image_path)}` : "";
  const animalReports = Array.isArray(report.animal_reports) ? report.animal_reports : [];
  const primaryAnimalReport = animalReports[0] || {};
  const primaryDetectedConditions = Array.isArray(report.detected_conditions) ? report.detected_conditions : (Array.isArray(primaryAnimalReport.detected_conditions) ? primaryAnimalReport.detected_conditions : []);
  return {
    report_id: report.report_id,
    image_path: report.image_path,
    image_url: imageUrl,
    created_at: report.created_at,
    analysis_status: report.analysis_status,
    is_animal: report.analysis_status === "animal_detected",
    animal_type: report.animal_type,
    animal_detected: report.animal_detected || report.animal_type,
    location_name: report.location_name,
    location_address: report.location_address,
    detection_confidence: report.detection_confidence,
    health_status: normalizedStatus,
    health_status_code: report.health_status_code || "",
    health_score: report.health_score,
    health_confidence: report.confidence_score,
    animal_name: report.animal_name || primaryAnimalReport.animal_name || "",
    bounding_box: {
      x1: report.bbox_x1,
      y1: report.bbox_y1,
      x2: report.bbox_x2,
      y2: report.bbox_y2
    },
    guidance: report.guidance,
    health_summary: report.guidance || getDefaultGuidance(normalizedStatus),
    condition_summary: buildModalConditionSummary(report, normalizedStatus, primaryDetectedConditions),
    breed_guess: report.breed_guess || primaryAnimalReport.breed_guess || "",
    animal_description: report.animal_description || primaryAnimalReport.animal_description || "",
    injury_description: report.injury_description || primaryAnimalReport.injury_description || "",
    recommended_actions: getStatusActions(report),
    needs_rescue: normalizedStatus === "Seriously Injured",
    needs_help: report.needs_help ?? normalizedStatus === "Seriously Injured",
    detected_conditions: primaryDetectedConditions,
    rescue_prompt: "Do you want to contact a rescue team?",
    rescue_contacts: [],
    vet_contacts: [],
    emergency_plan: report.emergency_plan || primaryAnimalReport.emergency_plan || {},
    avoid_steps: Array.isArray(report.avoid_steps) ? report.avoid_steps : (Array.isArray(primaryAnimalReport.avoid_steps) ? primaryAnimalReport.avoid_steps : []),
    contact_priority: report.contact_priority || primaryAnimalReport.contact_priority || "",
    urgency_level: report.urgency_level || primaryAnimalReport.urgency_level || "",
    urgency_label: report.urgency_label || primaryAnimalReport.urgency_label || "",
    primary_issues: Array.isArray(report.primary_issues) ? report.primary_issues : (Array.isArray(primaryAnimalReport.primary_issues) ? primaryAnimalReport.primary_issues : []),
    visible_symptoms: Array.isArray(report.visible_symptoms) ? report.visible_symptoms : (Array.isArray(primaryAnimalReport.visible_symptoms) ? primaryAnimalReport.visible_symptoms : []),
    body_condition: report.body_condition || primaryAnimalReport.body_condition || "",
    what_is_wrong: report.what_is_wrong || primaryAnimalReport.what_is_wrong || "",
    help_type: report.help_type || primaryAnimalReport.help_type || "none",
    triage_reasoning: report.triage_reasoning || primaryAnimalReport.triage_reasoning || "",
    animal_reports: animalReports,
    other_detections: []
  };
}

function bindRescue() {
  const input = document.getElementById("rescueLocationInput");
  const searchBtn = document.getElementById("searchRescueBtn");
  const currentBtn = document.getElementById("rescueCurrentLocationBtn");
  const suggestions = document.getElementById("rescueSuggestions");
  if (input) {
    input.addEventListener("input", () => handleRescueLocationTyping(input.value));
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        submitRescueLocationSearch(input.value.trim());
      }
    });
  }
  if (searchBtn) {
    searchBtn.addEventListener("click", () => submitRescueLocationSearch(input?.value.trim() || ""));
  }
  if (currentBtn) currentBtn.addEventListener("click", loadRescueContactsFromCurrentLocation);
  if (suggestions) {
    suggestions.addEventListener("click", (event) => {
      const button = event.target.closest("[data-rescue-suggestion]");
      if (!button) return;
      const index = Number(button.getAttribute("data-rescue-index"));
      const selectedItem = Number.isInteger(index) && index >= 0 ? state.rescueAutocompleteItems[index] : null;
      const label = selectedItem?.label || button.getAttribute("data-rescue-suggestion") || "";
      const lat = button.getAttribute("data-rescue-lat");
      const lng = button.getAttribute("data-rescue-lng");
      const address = selectedItem?.address || button.getAttribute("data-rescue-address") || "";
      const inputEl = document.getElementById("rescueLocationInput");
      if (inputEl) inputEl.value = label;
      hideRescueSuggestions();
      loadRescueContacts(lat != null && lng != null
        ? { location: [address, label].filter(Boolean).join(" - ") || label, lat: Number(lat), lng: Number(lng) }
        : { location: [address, label].filter(Boolean).join(" - ") || label });
    });
  }
  document.addEventListener("click", (event) => {
    handleContactNavigationClick(event);
    if (!event.target.closest(".rescue-search-wrap")) {
      hideRescueSuggestions();
    }
  });
}

function handleRescueLocationTyping(value) {
  const query = String(value || "").trim();
  if (state.rescueSuggestionsTimer) window.clearTimeout(state.rescueSuggestionsTimer);
  if (query.length < 2) {
    hideRescueSuggestions();
    return;
  }
  state.rescueSuggestionsTimer = window.setTimeout(() => fetchRescueSuggestions(query), 220);
}

async function fetchRescueSuggestions(query) {
  const suggestions = document.getElementById("rescueSuggestions");
  if (!suggestions) return;
  setRescueLoading(true);
  if (state.rescueSuggestionAbort) {
    state.rescueSuggestionAbort.abort();
  }
  const controller = new AbortController();
  state.rescueSuggestionAbort = controller;
  try {
    const res = await fetch(`${API}/api/locations/autocomplete?q=${encodeURIComponent(query)}`, { signal: controller.signal, cache: "no-store" });
    if (!res.ok) throw new Error(`Status ${res.status}`);
    const data = await res.json();
    const items = Array.isArray(data?.suggestions) ? data.suggestions : [];
    state.rescueAutocompleteItems = items;
    if (!items.length) {
      suggestions.innerHTML = "";
      suggestions.classList.add("hidden");
      return;
    }
    suggestions.innerHTML = items.map((item, index) => {
      const mainText = item.main_text || item.label || "";
      const secondaryText = item.secondary_text || item.address || "";
      return `
      <button type="button" class="rescue-suggestion-btn"
        data-rescue-index="${index}"
        data-rescue-suggestion="${esc(item.label || "")}"
        data-rescue-lat="${item.lat ?? ""}"
        data-rescue-lng="${item.lon ?? ""}"
        data-rescue-address="${esc(item.address || "")}">
        <span class="rescue-suggestion-pin" aria-hidden="true">
          <svg viewBox="0 0 24 24" class="rescue-suggestion-pin-svg">
            <path d="M12 21s6-5.2 6-11a6 6 0 0 0-12 0c0 5.8 6 11 6 11z"></path>
            <circle cx="12" cy="10" r="2.2"></circle>
          </svg>
        </span>
        <span class="rescue-suggestion-copy">
          <span class="rescue-suggestion-title">${esc(mainText)}</span>
          ${secondaryText ? `<span class="rescue-suggestion-address">${esc(secondaryText)}</span>` : ""}
        </span>
      </button>
    `}).join("");
    suggestions.classList.remove("hidden");
  } catch (error) {
    if (error?.name === "AbortError") return;
    state.rescueAutocompleteItems = [];
    suggestions.innerHTML = "";
    suggestions.classList.add("hidden");
  } finally {
    setRescueLoading(false);
  }
}

function hideRescueSuggestions() {
  const suggestions = document.getElementById("rescueSuggestions");
  if (!suggestions) return;
  suggestions.innerHTML = "";
  suggestions.classList.add("hidden");
}

function normalizeSearchText(value) {
  return String(value || "").trim().toLocaleLowerCase();
}

function buildRescueSearchQuery(rawValue) {
  const location = String(rawValue || "").trim();
  if (!location) return { location: "" };
  const normalizedLocation = normalizeSearchText(location);
  const exactMatch = state.rescueAutocompleteItems.find((item) => String(item.label || "").toLowerCase() === normalizedLocation)
    || state.rescueAutocompleteItems.find((item) => String(item.address || "").toLowerCase() === normalizedLocation)
    || state.rescueAutocompleteItems.find((item) => String(item.label || "").toLowerCase().includes(normalizedLocation))
    || state.rescueAutocompleteItems.find((item) => String(item.address || "").toLowerCase().includes(normalizedLocation));
  if (exactMatch && exactMatch.lat != null && exactMatch.lon != null) {
    return {
      location: [exactMatch.address, exactMatch.label].filter(Boolean).join(" - ") || exactMatch.label || location,
      lat: Number(exactMatch.lat),
      lng: Number(exactMatch.lon)
    };
  }
  return { location };
}

async function resolveRescueLocationQuery(rawValue) {
  const location = String(rawValue || "").trim();
  if (!location) return { location: "" };
  const cached = buildRescueSearchQuery(location);
  if (cached.lat != null && cached.lng != null) {
    return cached;
  }

  try {
    const res = await fetch(`${API}/api/locations/autocomplete?q=${encodeURIComponent(location)}`, { cache: "no-store" });
    if (res.ok) {
      const data = await res.json();
      const items = Array.isArray(data?.suggestions) ? data.suggestions : [];
      state.rescueAutocompleteItems = items;
      const normalizedLocation = normalizeSearchText(location);
      const exactMatch = items.find((item) => String(item.label || "").toLowerCase() === normalizedLocation)
        || items.find((item) => String(item.address || "").toLowerCase() === normalizedLocation)
        || items.find((item) => String(item.label || "").toLowerCase().includes(normalizedLocation))
        || items.find((item) => String(item.address || "").toLowerCase().includes(normalizedLocation));
      if (exactMatch && exactMatch.lat != null && exactMatch.lon != null) {
        return {
          location: [exactMatch.address, exactMatch.label].filter(Boolean).join(" - ") || exactMatch.label || location,
          lat: Number(exactMatch.lat),
          lng: Number(exactMatch.lon),
        };
      }
    }
  } catch (error) {
    console.warn("Could not resolve typed rescue location.", error);
  }
  return cached;
}

async function submitRescueLocationSearch(rawValue) {
  const query = String(rawValue || "").trim();
  if (!query) {
    await loadRescueContacts({ location: "" });
    return;
  }
  const resolved = await resolveRescueLocationQuery(query);
  await loadRescueContacts(resolved);
}

async function loadRescueContacts(options = null) {
  const rescueList = document.getElementById("rescueList");
  const vetList = document.getElementById("vetList");
  const query = options || state.rescueContext;
  const hasSearch = Boolean(query && (query.location || query.lat != null || query.lng != null));
  hideRescueSuggestions();
  setRescueLoading(hasSearch);

  if (options) {
    state.rescueContext = options.lat != null && options.lng != null ? options : null;
  }

  if (hasSearch) {
    try {
      const params = new URLSearchParams();
      if (query.location) params.set("location", query.location);
      if (query.lat != null && query.lng != null) {
        params.set("lat", String(query.lat));
        params.set("lng", String(query.lng));
      }
      const res = await fetch(`${API}/api/contacts/nearby?${params.toString()}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`Status ${res.status}`);
      const data = await res.json();
      renderRescueStatus(data.location_status, data.location_message);
      if (data.searched_location && (data.searched_location.name || data.searched_location.address || data.searched_location.lat != null || data.searched_location.lng != null)) {
        state.rescueContext = {
          location: [data.searched_location.address, data.searched_location.name].filter(Boolean).join(" - ") || query.location || "",
          lat: data.searched_location.lat != null ? Number(data.searched_location.lat) : undefined,
          lng: data.searched_location.lng != null ? Number(data.searched_location.lng) : undefined,
        };
      }
      const rescueContacts = sortContactsByDistance(data.rescue_contacts || []);
      const vetContacts = sortContactsByDistance(data.vet_contacts || []);
      if ((rescueContacts.length || vetContacts.length)) {
        renderRescueContactLists(rescueContacts, vetContacts);
      } else {
        renderRescueLookupFallback(
          [data.searched_location?.address, data.searched_location?.name, query.location].filter(Boolean).join(" - ") || "this location",
          data.location_status === "no_live_contacts"
        );
      }
      setRescueLoading(false);
      return;
    } catch (error) {
      console.warn("Could not load rescue contacts from search.", error);
      if (query?.location) {
        renderRescueLookupFallback(query.location, true);
        setRescueLoading(false);
        return;
      }
    }
  }

  const rescue = [];
  const vets = [];
  renderRescueStatus("idle", "");
  renderRescueContactLists(rescue, vets);
  setRescueLoading(false);
}

function renderRescueLookupFallback(locationText, useMapsFallback = true) {
  const rescueList = document.getElementById("rescueList");
  const vetList = document.getElementById("vetList");
  const mapQuery = encodeURIComponent(locationText || "");
  const mapLink = useMapsFallback
    ? `https://www.google.com/maps/search/?api=1&query=${mapQuery}`
    : `https://www.google.com/maps/search/${mapQuery}`;
  const fallbackCard = `
    <div class="contact-card rescue-fallback-card">
      <h4>Not detected</h4>
      <p class="contact-address"><a class="fallback-map-link" href="${esc(mapLink)}" target="_blank" rel="noopener noreferrer">Get contacts here</a></p>
    </div>
  `;
  renderRescueStatus("contact_fetch_failed", "Not detected.");
  rescueList.innerHTML = fallbackCard;
  vetList.innerHTML = "";
}

function renderRescueStatus(status, message) {
  const line = document.getElementById("rescueStatusLine");
  if (!line) return;
  const text = String(message || "").trim();
  if (!text) {
    line.textContent = "";
    line.className = "rescue-status-line hidden";
    return;
  }
  line.textContent = text;
  line.className = `rescue-status-line ${String(status || "unknown").replaceAll("_", "-")}`;
  line.classList.remove("hidden");
}

function renderRescueContactLists(rescue, vets, emptyMessage = null) {
  const rescueDisplay = pickNearbyContacts(rescue);
  const vetDisplay = pickNearbyContacts(vets);
  if (!rescueDisplay.contacts.length && !vetDisplay.contacts.length) {
    const emptyText = String(emptyMessage || "").trim() || "Not detected.";
    document.getElementById("rescueList").innerHTML = `<p class="loading-msg">${esc(emptyText)}</p>`;
    document.getElementById("vetList").innerHTML = "";
    return;
  }
  document.getElementById("rescueList").innerHTML = rescueDisplay.contacts.length
    ? `${rescueDisplay.isFarFallback ? `<p class="distance-warning">No contacts within 6 km, showing farther rescue contacts.</p>` : ""}${rescueDisplay.contacts.map(renderRescueCard).join("")}`
    : `<p class="loading-msg">${esc(emptyMessage || "Scan an animal and enable rescue help to load rescue services.")}</p>`;
  document.getElementById("vetList").innerHTML = vetDisplay.contacts.length
    ? `${vetDisplay.isFarFallback ? `<p class="distance-warning">No contacts within 6 km, showing farther veterinary clinics.</p>` : ""}${vetDisplay.contacts.map(renderVetCard).join("")}`
    : `<p class="loading-msg">${esc(emptyMessage || "Nearby veterinary clinics appear after a location-based scan.")}</p>`;
}

async function loadRescueContactsFromCurrentLocation() {
  const input = document.getElementById("rescueLocationInput");
  if (input) input.value = "";
  try {
    setRescueLoading(true);
    await requestBestEffortLocation({ forceFresh: true, showAlertOnFailure: false });
    if (state.location.lat == null || state.location.lng == null) {
      throw new Error("Could not read current location.");
    }
    state.rescueContext = {
      lat: state.location.lat,
      lng: state.location.lng,
    };
    loadRescueContacts(state.rescueContext);
  } catch (error) {
    window.alert("Please allow location access to load nearby contacts.");
  } finally {
    setRescueLoading(false);
  }
}

function setRescueLoading(isLoading) {
  const indicator = document.getElementById("rescueLoadingIndicator");
  if (!indicator) return;
  if (isLoading) {
    indicator.classList.remove("hidden");
  } else {
    indicator.classList.add("hidden");
  }
}

function renderRescueCard(contact) {
  const navLink = buildContactNavigationLink(contact);
  return `
    <div class="contact-card">
      <h4>${esc(contact.name)}</h4>
      <p class="contact-address">${esc(contact.address || contact.area || "Address not listed")}</p>
      ${contact.phone ? `<p class="contact-phone"><a href="tel:${esc(contact.phone)}">${esc(contact.phone)}</a></p>` : ""}
      <div class="contact-meta-row">
        ${navLink ? `<button class="contact-map-link" type="button" data-contact-nav="${esc(navLink)}">Map</button>` : `<span></span>`}
        ${renderDistanceBadge(contact.distance_km)}
      </div>
    </div>
  `;
}

function renderVetCard(contact) {
  const navLink = buildContactNavigationLink(contact);
  return `
    <div class="contact-card">
      <h4>${esc(contact.name)}</h4>
      <p class="contact-address">${esc(contact.address || contact.area || "Address not listed")}</p>
      ${contact.phone ? `<p class="contact-phone"><a href="tel:${esc(contact.phone)}">${esc(contact.phone)}</a></p>` : ""}
      <div class="contact-meta-row">
        ${navLink ? `<button class="contact-map-link" type="button" data-contact-nav="${esc(navLink)}">Map</button>` : `<span></span>`}
        ${renderDistanceBadge(contact.distance_km)}
      </div>
    </div>
  `;
}

function pickNearbyContacts(contacts) {
  const nearby = contacts.filter((contact) => typeof contact.distance_km === "number" && contact.distance_km <= 6);
  if (nearby.length) {
    return { contacts: nearby, isFarFallback: false };
  }
  return { contacts, isFarFallback: contacts.length > 0 };
}

function sortContactsByDistance(contacts) {
  return [...contacts].sort((left, right) => {
    const leftDistance = typeof left.distance_km === "number" ? left.distance_km : Number.POSITIVE_INFINITY;
    const rightDistance = typeof right.distance_km === "number" ? right.distance_km : Number.POSITIVE_INFINITY;
    return leftDistance - rightDistance;
  });
}

function renderDistanceBadge(distanceKm) {
  if (typeof distanceKm !== "number" || Number.isNaN(distanceKm)) return "";
  const rounded = distanceKm < 1 ? `${Math.round(distanceKm * 1000)} m away` : `${distanceKm.toFixed(1)} km away`;
  return `<span class="contact-distance">${esc(rounded)}</span>`;
}

function buildContactMapLink(contact) {
  const destination = [contact.address, contact.area, contact.name].filter(Boolean).join(" ");
  if (!destination) return contact.maps_link || "";
  const origin = state.location.lat != null && state.location.lng != null
    ? `&origin=${encodeURIComponent(`${state.location.lat},${state.location.lng}`)}`
    : "";
  return `https://www.google.com/maps/dir/?api=1&destination=${encodeURIComponent(destination)}${origin}&travelmode=driving`;
}

function buildContactNavigationLink(contact) {
  return buildContactMapLink(contact);
}

function handleContactNavigationClick(event) {
  const button = event.target.closest("[data-contact-nav]");
  if (!button) return;
  const url = button.getAttribute("data-contact-nav");
  if (!url) return;
  const ok = window.confirm("Start navigation on the map?");
  if (!ok) return;
  window.open(url, "_blank", "noopener,noreferrer");
}

function bindModal() {
  const modal = document.getElementById("reportModal");
  document.getElementById("closeReportModalBtn").addEventListener("click", closeReportModal);
  modal.addEventListener("click", (event) => {
    if (event.target === modal) closeReportModal();
  });
}

function openReportModal(report) {
  const detectedConditions = Array.isArray(report.detected_conditions) ? report.detected_conditions : [];
  const emergencyPlan = report.emergency_plan || {};
  const actions = Array.isArray(emergencyPlan.immediate_steps) && emergencyPlan.immediate_steps.length
    ? emergencyPlan.immediate_steps
    : getStatusActions(report);
  const locationText = [report.location_address, report.location_name].filter(Boolean).join(" - ") || "Location unavailable";
  const score = getReportHealthScore(report);
  const urgency = getUrgencyFromScore(score);
  const conditionSummary = report.condition_summary || buildModalConditionSummary(report, normalizeHealthStatus(report.health_status), detectedConditions);
  const clinicalNotes = report.health_summary || report.guidance || getDefaultGuidance(normalizeHealthStatus(report.health_status));
  const healthWord = getHealthWord(score, report.health_status);
  const urgencyLevel = String(report.urgency_level || "none").replaceAll("_", "-");
  const animalName = report.animal_name || report.animal_type || "Unknown animal";
  const animalType = report.animal_type || "Animal";
  const breedGuess = report.breed_guess || "";
  const animalDescription = report.animal_description || "";
  const injuryDescription = report.injury_description || "";
  const showBreedGuess = Boolean(breedGuess && breedGuess.toLowerCase() !== "unknown");

  document.getElementById("modalContent").innerHTML = `
    <div class="modal-report">
      ${report.image_path ? `<img class="report-preview-image" src="/${esc(buildReportImageUrl(report.image_path))}" alt="${esc(report.animal_type || "animal")}">` : ""}
      ${report.image_path ? `<div class="image-confidence-note">${esc(formatAiConfidence(report))}</div>` : ""}
      <div class="report-header">
        <div class="report-header-left">
          <h2>${esc(titleCase(animalName))}</h2>
          <span class="report-animal-type">${esc(titleCase(animalType))}</span>
          ${showBreedGuess ? `<span class="report-animal-breed">${esc(`Breed guess: ${titleCase(breedGuess)}`)}</span>` : ""}
        </div>
        <div class="report-header-actions">
          <button class="report-icon-btn" type="button" data-report-action="rescue" title="Nearby contacts" aria-label="Nearby contacts">
            <svg viewBox="0 0 24 24" class="report-icon-svg" aria-hidden="true">
              <path d="M8.4 4.8c.6-.6 1.5-.8 2.2-.3l2 1.4c.8.5 1 1.5.6 2.3l-1.1 1.8a1 1 0 0 0 .1 1.2l.5.6c1.2 1.3 2.5 2.6 3.9 3.8l.6.5a1 1 0 0 0 1.2.1l1.8-1.1c.8-.5 1.8-.2 2.3.6l1.4 2c.5.7.3 1.6-.3 2.2l-1.2 1.2c-1.1 1.1-2.7 1.6-4.2 1.1-2.8-.9-5.6-3-8.2-5.6-2.6-2.6-4.7-5.4-5.6-8.2-.5-1.5 0-3.1 1.1-4.2z"></path>
            </svg>
          </button>
          <button class="report-icon-btn" type="button" data-page="carebot" title="Care Bot" aria-label="Care Bot">
            <svg viewBox="0 0 24 24" class="report-icon-svg" aria-hidden="true">
              <path d="M6 6.5h12A2.5 2.5 0 0 1 20.5 9v6A2.5 2.5 0 0 1 18 17.5H11l-4.5 3V17.5H6A2.5 2.5 0 0 1 3.5 15V9A2.5 2.5 0 0 1 6 6.5z"></path>
              <path d="M12 9.2l.8 1.7 1.9.2-1.4 1.2.4 1.8-1.7-.9-1.7.9.4-1.8-1.4-1.2 1.9-.2z" class="report-icon-cut"></path>
            </svg>
          </button>
          <div class="status-badge-lg ${esc(scoreClass(score))}" id="statusBadgeLg">${esc(healthWord)}</div>
        </div>
      </div>

      <div class="report-alert-strip ${esc(urgencyLevel)}">
        <div class="report-alert-title">${esc(report.urgency_label || "No action needed")}</div>
        <div class="report-alert-text">${esc(report.triage_reasoning || "No triage reasoning available.")}</div>
      </div>

      <div class="vitals-row">
        <div class="vital-card ${esc(scoreClass(score))}">
          <span class="vital-label">Health</span>
          <span class="vital-value ${esc(scoreClass(score))}">${esc(healthWord)}</span>
        </div>
        <div class="vital-card">
          <span class="vital-label">Help Type</span>
          <span class="vital-value">${esc(score != null ? formatHelpType(report.help_type || "none") : "Not recognized")}</span>
        </div>
        <div class="vital-card">
          <span class="vital-label">Urgency</span>
          <span class="vital-value">${esc(score != null ? urgency : "Not recognized")}</span>
        </div>
      </div>
      <div class="report-section">
        <h4 class="section-heading">${score != null ? "Condition Summary" : "Not Recognized"}</h4>
        <p class="report-text">${esc(score != null ? conditionSummary : "The animal could not be recognized from this image.")}</p>
      </div>

      ${score != null ? `
        <div class="report-section">
          <h4 class="section-heading">Animal Description</h4>
          <p class="report-text">${esc(animalDescription || report.animal_description || "No animal description available.")}</p>
        </div>
        <div class="report-section">
          <h4 class="section-heading">Injury Description</h4>
          <p class="report-text">${esc(injuryDescription || report.injury_description || "No injury description available.")}</p>
        </div>
      ` : ""}

      ${score != null && detectedConditions.length ? `
        <div class="report-section">
          <h4 class="section-heading">Visual Findings</h4>
          <ul class="findings-list">
            ${detectedConditions.map((condition) => `<li>${esc(formatCondition(condition))}</li>`).join("")}
          </ul>
        </div>
      ` : ""}

      ${score != null ? `
        <div class="report-section">
          <h4 class="section-heading">Clinical Notes</h4>
          <p class="report-text">${esc(clinicalNotes)}</p>
        </div>` : ""}

      <div class="report-section">
        <h4 class="section-heading">What’s Wrong</h4>
        <p class="report-text">${esc(score != null ? (report.what_is_wrong || "No plain-language problem summary is available.") : "The animal could not be recognized from this image.")}</p>
      </div>

      ${score != null ? `
      <div class="report-section">
        <h4 class="section-heading">Recommended Actions</h4>
        <ol class="actions-list">
          ${actions.map((action) => `<li>${esc(action)}</li>`).join("")}
        </ol>
      </div>` : ""}

      ${score != null ? renderEmergencyMarkup(emergencyPlan, "modal") : renderEmergencyMarkup({ summary: "Not recognized as a supported animal.", immediate_steps: [], avoid_steps: [], contact_priority: "Retake a clearer image" }, "modal")}

      <div class="location-strip report-location-strip">
        <span class="loc-icon">LOC</span>
        <span>${esc(locationText)}</span>
      </div>

      <div class="report-footer">
        <div class="report-footer-actions">
          <button class="report-footer-btn" type="button" data-report-action="download" title="Download report" aria-label="Download report">
            <svg viewBox="0 0 24 24" class="report-footer-svg" aria-hidden="true">
              <path d="M12 3v9m0 0 3.5-3.5M12 12l-3.5-3.5"></path>
              <path d="M5 14.5V19h14v-4.5"></path>
            </svg>
          </button>
          <button class="report-footer-btn" type="button" data-report-action="send" title="Send report" aria-label="Send report">
            <svg viewBox="0 0 24 24" class="report-footer-svg" aria-hidden="true">
              <path d="M4.5 12 19.5 4.5 15 19.5l-3.2-6.3L4.5 12z"></path>
              <path d="M11.8 13.2 19.5 4.5"></path>
            </svg>
          </button>
          <button class="report-footer-btn" type="button" data-report-action="print" title="Print report" aria-label="Print report">
            <svg viewBox="0 0 24 24" class="report-footer-svg" aria-hidden="true">
              <path d="M7 8V4h10v4"></path>
              <path d="M6 10h12a2 2 0 0 1 2 2v5h-3v3H7v-3H4v-5a2 2 0 0 1 2-2z"></path>
            </svg>
          </button>
        </div>
        <span class="report-footer-date">${esc(formatDateTime(report.created_at))}</span>
      </div>
    </div>
  `;
  document.getElementById("reportModal").classList.remove("hidden");
}

function closeReportModal() {
  document.getElementById("reportModal").classList.add("hidden");
  state.modalReport = null;
  state.modalAnimalReportIndex = 0;
}

function renderStoredReportModal() {
  const data = state.modalReport;
  if (!data) return;
  state.modalAnimalReportIndex = 0;
  const activeReport = getAnimalReports(data)[0] || {};
  const score = getReportHealthScore(activeReport, data);
  const isUnknown = isNotRecognizedReport(activeReport, data) || score == null;
  const conditions = Array.isArray(activeReport.detected_conditions) ? activeReport.detected_conditions : (data.detected_conditions || []);
  const visibleSymptoms = Array.isArray(activeReport.visible_symptoms) ? activeReport.visible_symptoms : (data.visible_symptoms || []);
  const emergencyPlan = activeReport.emergency_plan || data.emergency_plan || {};
  const actions = Array.isArray(activeReport.recommended_actions) && activeReport.recommended_actions.length
    ? activeReport.recommended_actions
    : (Array.isArray(emergencyPlan.immediate_steps) && emergencyPlan.immediate_steps.length ? emergencyPlan.immediate_steps : (data.recommended_actions || []));
  const locationText = [data.location_address, data.location_name].filter(Boolean).join(' - ') || 'Location unavailable';
  const healthWord = getHealthWord(score, activeReport.health_status || data.health_status);
  const conditionSummary = activeReport.condition_summary || buildModalConditionSummary(activeReport, normalizeHealthStatus(activeReport.health_status || data.health_status), conditions);
  const animalName = activeReport.animal_name || data.animal_name || activeReport.animal_type || data.animal_type || 'Unknown animal';
  const animalType = activeReport.animal_type || data.animal_type || 'Animal';
  const urgencyLevel = String(activeReport.urgency_level || data.urgency_level || 'none').replaceAll('_', '-');
  const breedGuess = activeReport.breed_guess || data.breed_guess || '';
  const animalDescription = activeReport.animal_description || data.animal_description || '';
  const injuryDescription = activeReport.injury_description || data.injury_description || '';
  const showBreedGuess = Boolean(breedGuess && breedGuess.toLowerCase() !== 'unknown');

  document.getElementById('modalContent').innerHTML = `
    <div class="modal-report full-report-modal">
      ${data.image_url ? `<img class="report-preview-image" src="${esc(data.image_url)}" alt="${esc(activeReport.animal_type || data.animal_type || 'animal')}">` : ''}
      ${data.image_url ? `<div class="image-confidence-note">${esc(formatAiConfidence(activeReport, data))}</div>` : ''}
      <div class="report-header">
        <div class="report-header-left">
          <h2>${esc(titleCase(animalName))}</h2>
          <span class="report-animal-type">${esc(titleCase(animalType))}</span>
          ${showBreedGuess ? `<span class="report-animal-breed">${esc(`Breed guess: ${titleCase(breedGuess)}`)}</span>` : ''}
        </div>
        <div class="report-header-actions">
          <button class="report-icon-btn" type="button" data-report-action="rescue" title="Nearby contacts" aria-label="Nearby contacts">
            <svg viewBox="0 0 24 24" class="report-icon-svg" aria-hidden="true">
              <path d="M8.4 4.8c.6-.6 1.5-.8 2.2-.3l2 1.4c.8.5 1 1.5.6 2.3l-1.1 1.8a1 1 0 0 0 .1 1.2l.5.6c1.2 1.3 2.5 2.6 3.9 3.8l.6.5a1 1 0 0 0 1.2.1l1.8-1.1c.8-.5 1.8-.2 2.3.6l1.4 2c.5.7.3 1.6-.3 2.2l-1.2 1.2c-1.1 1.1-2.7 1.6-4.2 1.1-2.8-.9-5.6-3-8.2-5.6-2.6-2.6-4.7-5.4-5.6-8.2-.5-1.5 0-3.1 1.1-4.2z"></path>
            </svg>
          </button>
          <button class="report-icon-btn" type="button" data-page="carebot" title="Care Bot" aria-label="Care Bot">
            <svg viewBox="0 0 24 24" class="report-icon-svg" aria-hidden="true">
              <path d="M6 6.5h12A2.5 2.5 0 0 1 20.5 9v6A2.5 2.5 0 0 1 18 17.5H11l-4.5 3V17.5H6A2.5 2.5 0 0 1 3.5 15V9A2.5 2.5 0 0 1 6 6.5z"></path>
              <path d="M12 9.2l.8 1.7 1.9.2-1.4 1.2.4 1.8-1.7-.9-1.7.9.4-1.8-1.4-1.2 1.9-.2z" class="report-icon-cut"></path>
            </svg>
          </button>
          <div class="status-badge-lg ${esc(scoreClass(score))}">${esc(isUnknown ? 'Not recognized' : healthWord)}</div>
        </div>
      </div>
      ${isUnknown ? `
        <div class="report-alert-strip none">
          <div class="report-alert-title">Not recognized</div>
          <div class="report-alert-text">The image could not be confirmed as a supported animal case, so no medical advice is generated.</div>
        </div>
      ` : `
        <div class="report-alert-strip ${esc(urgencyLevel)}">
          <div class="report-alert-title">${esc(activeReport.urgency_label || data.urgency_label || 'No action needed')}</div>
          <div class="report-alert-text">${esc(activeReport.triage_reasoning || data.triage_reasoning || 'No triage reasoning available.')}</div>
        </div>
      `}
      <div class="vitals-row">
        <div class="vital-card ${esc(scoreClass(score))}">
          <span class="vital-label">Health</span>
          <span class="vital-value ${esc(scoreClass(score))}">${esc(isUnknown ? 'Not recognized' : healthWord)}</span>
        </div>
        <div class="vital-card">
          <span class="vital-label">Help Type</span>
          <span class="vital-value">${esc(isUnknown ? 'Not recognized' : formatHelpType(activeReport.help_type || data.help_type || 'none'))}</span>
        </div>
        <div class="vital-card">
          <span class="vital-label">Urgency</span>
          <span class="vital-value">${esc(isUnknown ? 'Not recognized' : (activeReport.urgency_label || data.urgency_label || (activeReport.needs_rescue ? 'Urgent' : 'Non-urgent')))}</span>
        </div>
      </div>
      <div class="report-section">
        <h4 class="section-heading">${esc(isUnknown ? 'Not Recognized' : 'What’s Wrong')}</h4>
        <p class="report-text">${esc(isUnknown ? 'The animal could not be recognized from this image.' : (activeReport.what_is_wrong || data.what_is_wrong || 'No plain-language problem summary is available.'))}</p>
      </div>
      ${!isUnknown ? `
        <div class="report-section">
          <h4 class="section-heading">Animal Description</h4>
          <p class="report-text">${esc(animalDescription || data.animal_description || 'No animal description available.')}</p>
        </div>
        <div class="report-section">
          <h4 class="section-heading">Injury Description</h4>
          <p class="report-text">${esc(injuryDescription || data.injury_description || 'No injury description available.')}</p>
        </div>
      ` : ''}
      ${!isUnknown ? `
        <div class="report-section">
          <h4 class="section-heading">Condition Summary</h4>
          <p class="report-text">${esc(conditionSummary || 'No condition summary available.')}</p>
        </div>
        ${conditions.length ? `
          <div class="report-section">
            <h4 class="section-heading">Visible Findings</h4>
            <ul class="findings-list">
              ${conditions.map((condition) => `<li>${esc(formatCondition(condition))}</li>`).join('')}
            </ul>
          </div>
        ` : ''}
        ${visibleSymptoms.length ? `
          <div class="report-section">
            <h4 class="section-heading">Visible Symptoms</h4>
            <ul class="findings-list">
              ${visibleSymptoms.map((symptom) => `<li>${esc(formatCondition(symptom))}</li>`).join('')}
            </ul>
          </div>
        ` : ''}
        <div class="report-section">
          <h4 class="section-heading">What’s Wrong</h4>
          <p class="report-text">${esc(activeReport.what_is_wrong || data.what_is_wrong || 'No plain-language problem summary is available.')}</p>
        </div>
        <div class="report-section">
          <h4 class="section-heading">Recommended Actions</h4>
          <ol class="actions-list">
            ${(actions.length ? actions : ['Observe the animal carefully and contact rescue or a vet if it seems distressed.']).map((action) => `<li>${esc(action)}</li>`).join('')}
          </ol>
        </div>
      ` : ''}
      ${renderEmergencyMarkup(isUnknown ? { summary: 'Not recognized as a supported animal.', immediate_steps: [], avoid_steps: [], contact_priority: 'Retake a clearer image' } : emergencyPlan, 'modal')}

      <div class="location-strip report-location-strip">
        <span class="loc-icon">LOC</span>
        <span>${esc(locationText)}</span>
      </div>

      <div class="report-footer">
        <div class="report-footer-actions">
          <button class="report-footer-btn" type="button" data-report-action="download" title="Download report" aria-label="Download report">
            <svg viewBox="0 0 24 24" class="report-footer-svg" aria-hidden="true">
              <path d="M12 3v9m0 0 3.5-3.5M12 12l-3.5-3.5"></path>
              <path d="M5 14.5V19h14v-4.5"></path>
            </svg>
          </button>
          <button class="report-footer-btn" type="button" data-report-action="send" title="Send report" aria-label="Send report">
            <svg viewBox="0 0 24 24" class="report-footer-svg" aria-hidden="true">
              <path d="M4.5 12 19.5 4.5 15 19.5l-3.2-6.3L4.5 12z"></path>
              <path d="M11.8 13.2 19.5 4.5"></path>
            </svg>
          </button>
          <button class="report-footer-btn" type="button" data-report-action="print" title="Print report" aria-label="Print report">
            <svg viewBox="0 0 24 24" class="report-footer-svg" aria-hidden="true">
              <path d="M7 8V4h10v4"></path>
              <path d="M6 10h12a2 2 0 0 1 2 2v5h-3v3H7v-3H4v-5a2 2 0 0 1 2-2z"></path>
            </svg>
          </button>
        </div>
        <span class="report-footer-date">${esc(formatDateTime(data.created_at))}</span>
      </div>
    </div>
  `;
}

function buildDonationPaymentPageUrl(baseUrl, details) {
  const normalizedBaseUrl = baseUrl.includes("://") ? baseUrl : `https://${baseUrl}`;
  const url = new URL(normalizedBaseUrl);
  url.searchParams.set("amount", String(details.amount));
  url.searchParams.set("email", details.email);
  url.searchParams.set("full_name", details.name);
  if (details.phone) {
    url.searchParams.set("phone", details.phone);
  }
  return url.toString();
}

async function openDonationPaymentPage() {
  const name = document.getElementById("donorName").value.trim();
  const email = document.getElementById("donorEmail").value.trim();
  const amount = document.getElementById("donationAmount").value.trim();
  const animal = document.getElementById("donationAnimal").value.trim();
  const duration = document.getElementById("donationDuration").value.trim();
  const phoneInput = document.getElementById("donorPhone");
  const phone = phoneInput ? phoneInput.value.trim() : "";
  const feedback = document.getElementById("donationFeedback");
  const pageUrl = state.publicConfig.razorpayPaymentPageUrl;

  if (!pageUrl) {
    feedback.textContent = "Set RAZORPAY_PAYMENT_PAGE_URL in .env to enable Razorpay donations.";
    return;
  }

  const numericAmount = Math.max(1, Math.round(Number(amount || 0)));
  const targetUrl = buildDonationPaymentPageUrl(pageUrl, { name, email, amount: numericAmount, phone });
  const popup = window.open(targetUrl, "_blank", "noopener,noreferrer");
  if (!popup) {
    window.location.href = targetUrl;
  }
  feedback.textContent = "Razorpay Payment Page opened. After payment, the receipt is emailed automatically if receipts are enabled on the page.";
  if (animal || duration) {
    feedback.textContent += ` Donation note: ${[animal, duration].filter(Boolean).join(" Â· ")}.`;
  }
}

function buildUpiIntentUrl(details) {
  const vpa = state.publicConfig.upiVpa;
  if (!vpa) return null;
  const params = new URLSearchParams();
  params.set("pa", vpa);
  params.set("pn", details.payeeName || state.publicConfig.upiPayeeName || "Paw Connect");
  params.set("am", String(details.amount));
  params.set("cu", "INR");
  params.set("tn", details.note || state.publicConfig.upiNote || "Support street animal care");
  if (details.email) {
    params.set("tr", details.email);
  }
  return `upi://pay?${params.toString()}`;
}

function updateUpiUi() {
  const upiIdValue = document.getElementById("upiIdValue");
  const upiNoteText = document.getElementById("upiNoteText");
  const openBtn = document.getElementById("openUpiBtn");
  const copyBtn = document.getElementById("copyUpiBtn");
  const vpa = state.publicConfig.upiVpa;
  if (upiIdValue) {
    upiIdValue.textContent = vpa || "Set UPI_VPA in .env";
  }
  if (upiNoteText) {
    upiNoteText.textContent = vpa
      ? `${state.publicConfig.upiNote}. Free UPI flow, no gateway fee.`
      : "Set UPI_VPA in .env to enable the free UPI donation flow.";
  }
  if (openBtn) openBtn.disabled = !vpa;
  if (copyBtn) copyBtn.disabled = !vpa;
}

function copyUpiId() {
  const feedback = document.getElementById("donationFeedback");
  const vpa = state.publicConfig.upiVpa;
  if (!vpa) {
    if (feedback) feedback.textContent = "Set UPI_VPA in .env first.";
    return;
  }
  navigator.clipboard.writeText(vpa).then(() => {
    if (feedback) feedback.textContent = "UPI ID copied. Paste it into your UPI app.";
  }).catch(() => {
    if (feedback) feedback.textContent = `UPI ID: ${vpa}`;
  });
}

async function openUpiDonationFlow() {
  const email = document.getElementById("donorEmail").value.trim();
  const amount = document.getElementById("donationAmount").value.trim();
  const animal = document.getElementById("donationAnimal").value.trim();
  const duration = document.getElementById("donationDuration").value.trim();
  const feedback = document.getElementById("donationFeedback");
  const upiUrl = buildUpiIntentUrl({
    amount: Math.max(1, Math.round(Number(amount || 0))),
    email,
    payeeName: state.publicConfig.upiPayeeName,
    note: [state.publicConfig.upiNote, animal, duration].filter(Boolean).join(" Â· "),
  });

  if (!upiUrl) {
    feedback.textContent = "Set UPI_VPA in .env to enable the free UPI donation flow.";
    return;
  }

  const popup = window.open(upiUrl, "_blank", "noopener,noreferrer");
  if (!popup) {
    window.location.href = upiUrl;
  }
  feedback.textContent = "UPI app opened. After payment, bank or UPI confirmation is used instead of a paid gateway receipt.";
  if (animal || duration) {
    feedback.textContent += ` Donation note: ${[animal, duration].filter(Boolean).join(" Â· ")}.`;
  }
}

function statusClass(status) {
  if (status === "Seriously Injured" || status === "Serious") return "serious";
  if (status === "Mildly Injured" || status === "Mild") return "mild";
  if (status === "Healthy") return "healthy";
  return "";
}

function scoreClass(score) {
  if (typeof score !== "number" || Number.isNaN(score)) return "";
  if (score > 80) return "healthy";
  if (score > 40) return "mild";
  return "serious";
}

function getUrgencyFromScore(score) {
  if (typeof score !== "number" || Number.isNaN(score)) return "Undetermined";
  if (score > 80) return "Stable";
  if (score > 40) return "Monitor closely";
  return "Immediate attention";
}

function applyScoreStyling(element, score) {
  if (!element) return;
  element.classList.remove("healthy", "mild", "serious");
  const cls = scoreClass(score);
  if (cls) element.classList.add(cls);
}

function applyScoreCardStyling(card, score) {
  if (!card) return;
  card.classList.remove("healthy", "mild", "serious");
  const cls = scoreClass(score);
  if (cls) card.classList.add(cls);
}

function getHealthWord(score, fallbackStatus = null) {
  if (typeof score === "number" && !Number.isNaN(score)) {
    if (score > 80) return "Healthy";
    if (score > 40) return "Mild";
    return "Serious";
  }
  const normalized = normalizeHealthStatus(fallbackStatus);
  if (normalized === "Healthy") return "Healthy";
  if (normalized === "Mildly Injured") return "Mild";
  if (normalized === "Seriously Injured") return "Serious";
  if (normalized === "NotApplicable") return "Not Applicable";
  return "Not recognized";
}

function formatAiConfidence(report, data = {}) {
  const score = report?.confidence_score ?? report?.detection_confidence ?? data?.confidence_score ?? data?.detection_confidence ?? null;
  if (typeof score !== "number" || Number.isNaN(score)) return "AI --";
  const pct = Math.min(99, Math.max(1, Math.round(score * 100)));
  return `AI-${pct}%`;
}

function getPrimaryIssueText(report = {}, data = {}) {
  const candidates = [
    ...(Array.isArray(report.primary_issues) ? report.primary_issues : []),
    ...(Array.isArray(data.primary_issues) ? data.primary_issues : []),
    ...(Array.isArray(report.detected_conditions) ? report.detected_conditions : []),
    ...(Array.isArray(data.detected_conditions) ? data.detected_conditions : []),
  ].filter(Boolean);
  if (candidates.length) return formatCondition(candidates[0]);
  const fallback = report.primary_issue || data.primary_issue || report.condition_summary || data.condition_summary || "";
  if (fallback) return formatCondition(fallback);
  return "Not recognized";
}

function getReportPayloadForActions() {
  return state.modalReport || state.lastReport || null;
}

function handleReportActionClick(event) {
  const button = event.target.closest("[data-report-action]");
  if (!button) return;
  const action = button.getAttribute("data-report-action");
  if (!action) return;
  if (action === "rescue") {
    openRescueContactsFromReport(getReportPayloadForActions());
    return;
  }
  if (action === "print") {
    openPrintableReportPayload(getReportPayloadForActions(), true);
    return;
  }
  if (action === "download") {
    downloadReportPdfPayload(getReportPayloadForActions());
    return;
  }
  if (action === "send") {
    sendReportPayload(getReportPayloadForActions());
  }
}

function openRescueContactsFromReport(payload) {
  if (!payload) return;
  const locationText = [payload.location_address, payload.location_name].filter(Boolean).join(" - ") || "";
  const input = document.getElementById("rescueLocationInput");
  if (input) input.value = locationText;
  const context = {};
  if (locationText) context.location = locationText;
  if (payload.location_lat != null && payload.location_long != null) {
    context.lat = Number(payload.location_lat);
    context.lng = Number(payload.location_long);
  }
  state.rescueContext = Object.keys(context).length ? context : null;
  showPage("rescue");
  if (input) input.focus();
  if (locationText) {
    loadRescueContacts(state.rescueContext || { location: locationText });
  }
}

function buildReportSummaryText(payload) {
  if (!payload) return "";
  const reports = getAnimalReports(payload);
  const activeReport = reports[0] || {};
  const score = getReportHealthScore(activeReport, payload);
  const healthWord = getHealthWord(score, activeReport.health_status || payload.health_status);
  const animalName = titleCase(activeReport.animal_name || payload.animal_name || activeReport.animal_type || payload.animal_type || "Unknown animal");
  const animalType = titleCase(activeReport.animal_type || payload.animal_type || "Animal");
  const locationText = [payload.location_address, payload.location_name].filter(Boolean).join(" - ") || "Location unavailable";
  const urgency = score != null ? (activeReport.urgency_label || payload.urgency_label || (activeReport.needs_rescue ? "Urgent" : "Non-urgent")) : "Not recognized";
  const primaryIssue = getPrimaryIssueText(activeReport, payload);
  return [
    `Animal: ${animalName}`,
    `Type: ${animalType}`,
    `Health: ${healthWord}`,
    `Help Type: ${score != null ? formatHelpType(activeReport.help_type || payload.help_type || "none") : "Not recognized"}`,
    `Urgency: ${urgency}`,
    `Primary Issue: ${primaryIssue}`,
    `Location: ${locationText}`,
  ].join("\n");
}

function buildReportPdfLines(payload) {
  if (!payload) return [];
  const reports = getAnimalReports(payload);
  const activeReport = reports[0] || {};
  const score = getReportHealthScore(activeReport, payload);
  const healthWord = getHealthWord(score, activeReport.health_status || payload.health_status);
  const animalName = titleCase(activeReport.animal_name || payload.animal_name || activeReport.animal_type || payload.animal_type || "Unknown animal");
  const animalType = titleCase(activeReport.animal_type || payload.animal_type || "Animal");
  const breedGuess = activeReport.breed_guess || payload.breed_guess || "";
  const animalDescription = activeReport.animal_description || payload.animal_description || "";
  const injuryDescription = activeReport.injury_description || payload.injury_description || "";
  const conditionSummary = activeReport.condition_summary || payload.condition_summary || "";
  const bodyCondition = activeReport.body_condition || payload.body_condition || "";
  const whatIsWrong = activeReport.what_is_wrong || payload.what_is_wrong || "";
  const clinicalNotes = activeReport.health_summary || payload.health_summary || "";
  const emergencyPlan = activeReport.emergency_plan || payload.emergency_plan || {};
  const emergencySummary = emergencyPlan.summary || payload.triage_reasoning || "";
  const immediateSteps = Array.isArray(emergencyPlan.immediate_steps) ? emergencyPlan.immediate_steps : [];
  const avoidSteps = Array.isArray(emergencyPlan.avoid_steps) ? emergencyPlan.avoid_steps : [];
  const locationText = [payload.location_address, payload.location_name].filter(Boolean).join(" - ") || "Location unavailable";
  const reportDate = formatDateTime(payload.created_at || new Date());
  const confidence = formatAiConfidence(activeReport, payload);
  const urgency = score != null ? (activeReport.urgency_label || payload.urgency_label || (activeReport.needs_rescue ? "Urgent" : "Non-urgent")) : "Not recognized";

  const lines = [
    "Paw Connect Report",
    `Animal: ${animalName}`,
    `Type: ${animalType}`,
  ];
  if (breedGuess && breedGuess.toLowerCase() !== "unknown") {
    lines.push(`Breed guess: ${titleCase(breedGuess)}`);
  }
  lines.push(
    `Health: ${healthWord}`,
    `Help Type: ${score != null ? formatHelpType(activeReport.help_type || payload.help_type || "none") : "Not recognized"}`,
    `Urgency: ${urgency}`,
    `AI Confidence: ${confidence}`,
    "",
    "Condition Summary:",
    conditionSummary || "No condition summary available.",
    "",
    "Animal Description:",
    animalDescription || "No animal description available.",
    "",
    "Injury Description:",
    injuryDescription || "No injury description available.",
    "",
    "Body Condition:",
    bodyCondition || "No body condition note available.",
    "",
    "What's Wrong:",
    whatIsWrong || "No plain-language problem summary is available.",
    "",
    "Clinical Notes:",
    clinicalNotes || "No clinical notes available.",
    "",
    "Emergency SOS:",
    emergencySummary || "No emergency summary available.",
  );
  if (immediateSteps.length) {
    lines.push("");
    lines.push("Immediate Steps:");
    immediateSteps.forEach((step, index) => lines.push(`${index + 1}. ${step}`));
  }
  if (avoidSteps.length) {
    lines.push("");
    lines.push("Avoid:");
    avoidSteps.forEach((step, index) => lines.push(`${index + 1}. ${step}`));
  }
  lines.push(
    "",
    `Location: ${locationText}`,
    `Date: ${reportDate}`,
  );
  return lines;
}

function sanitizePdfText(text) {
  return String(text || "")
    .normalize("NFKD")
    .replace(/[^\x09\x0A\x0D\x20-\x7E]/g, "?")
    .replaceAll("\\", "\\\\")
    .replaceAll("(", "\\(")
    .replaceAll(")", "\\)")
    .replaceAll("\r", "");
}

function wrapPdfLines(lines, maxChars = 90) {
  const wrapped = [];
  for (const line of lines) {
    const raw = sanitizePdfText(line);
    if (!raw) {
      wrapped.push("");
      continue;
    }
    let remaining = raw;
    while (remaining.length > maxChars) {
      let breakIndex = remaining.lastIndexOf(" ", maxChars);
      if (breakIndex < Math.floor(maxChars * 0.5)) breakIndex = maxChars;
      wrapped.push(remaining.slice(0, breakIndex).trimEnd());
      remaining = remaining.slice(breakIndex).trimStart();
    }
    wrapped.push(remaining);
  }
  return wrapped;
}

async function buildReportPdfBlob(payload) {
  const pageWidth = 612;
  const pageHeight = 792;
  const margin = 40;
  const fontSize = 11;
  const lineHeight = 14;
  const titleHeight = 24;
  const imageBoxHeight = 180;
  const imageBoxWidth = pageWidth - margin * 2;
  const lines = wrapPdfLines(buildReportPdfLines(payload));
  const previewUrl = payload.image_url || (payload.image_path ? `/${buildReportImageUrl(payload.image_path)}` : "");
  const imageData = previewUrl ? await loadImageAsJpegData(previewUrl) : null;

  const textTop = margin + titleHeight + (imageData ? imageBoxHeight + 18 : 0);
  const textBottom = margin;
  const linesPerPage = Math.max(1, Math.floor((pageHeight - textTop - textBottom) / lineHeight));
  const pages = [];
  for (let i = 0; i < lines.length; i += linesPerPage) {
    pages.push(lines.slice(i, i + linesPerPage));
  }
  if (!pages.length) pages.push(["No report data available."]);

  const objects = [];
  const encoder = new TextEncoder();
  const addObject = (parts) => {
    objects.push(parts);
    return objects.length;
  };

  const fontObj = addObject(["<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]);
  let imageObj = null;
  if (imageData) {
    imageObj = addObject(
      [
        `<< /Type /XObject /Subtype /Image /Width ${imageData.width} /Height ${imageData.height} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${imageData.bytes.length} >>\nstream\n`,
        imageData.bytes,
        "\nendstream",
      ]
    );
  }

  const contentObjects = [];
  const pageObjects = [];

  pages.forEach((pageLines, pageIndex) => {
    const contentLines = [];
    if (pageIndex === 0) {
      contentLines.push("q");
      contentLines.push("BT");
      contentLines.push(`/F1 18 Tf`);
      contentLines.push(`1 0 0 1 ${margin} ${pageHeight - margin - 8} Tm`);
      contentLines.push(`(${sanitizePdfText("Paw Connect Report")}) Tj`);
      contentLines.push("ET");
      if (imageObj && imageData) {
        const render = fitWithinBox(imageData.width, imageData.height, imageBoxWidth, imageBoxHeight);
        const imageX = margin + (imageBoxWidth - render.width) / 2;
        const imageY = pageHeight - margin - titleHeight - render.height;
        contentLines.push(`q`);
        contentLines.push(`${render.width.toFixed(2)} 0 0 ${render.height.toFixed(2)} ${imageX.toFixed(2)} ${imageY.toFixed(2)} cm`);
        contentLines.push(`/Im1 Do`);
        contentLines.push("Q");
      }
      contentLines.push("Q");
    }
    contentLines.push("BT");
    contentLines.push(`/F1 ${fontSize} Tf`);
    contentLines.push(`1 0 0 1 ${margin} ${pageHeight - textTop} Tm`);
    pageLines.forEach((line, index) => {
      if (index > 0) contentLines.push(`0 -${lineHeight} Td`);
      contentLines.push(`(${sanitizePdfText(line)}) Tj`);
    });
    contentLines.push("ET");
    const stream = contentLines.join("\n");
    contentObjects.push(addObject([`<< /Length ${encoder.encode(stream).length} >>\nstream\n${stream}\nendstream`]));
  });

  const pagesObj = addObject(["<< /Type /Pages /Kids [] /Count 0 >>"]);
  pages.forEach((_, index) => {
    const resources = imageObj && index === 0
      ? `<< /Font << /F1 ${fontObj} 0 R >> /XObject << /Im1 ${imageObj} 0 R >> >>`
      : `<< /Font << /F1 ${fontObj} 0 R >> >>`;
    pageObjects.push(addObject([`<< /Type /Page /Parent ${pagesObj} 0 R /MediaBox [0 0 ${pageWidth} ${pageHeight}] /Resources ${resources} /Contents ${contentObjects[index]} 0 R >>`]));
  });
  objects[pagesObj - 1] = [`<< /Type /Pages /Kids [ ${pageObjects.map((id) => `${id} 0 R`).join(" ")} ] /Count ${pageObjects.length} >>`];
  const catalogObj = addObject([`<< /Type /Catalog /Pages ${pagesObj} 0 R >>`]);

  const pdfParts = [];
  const offsets = [0];
  let pdfLength = 0;
  const pushPart = (part) => {
    pdfParts.push(part);
    pdfLength += typeof part === "string" ? encoder.encode(part).length : part.byteLength;
  };
  pushPart("%PDF-1.4\n");
  objects.forEach((obj, index) => {
    offsets[index + 1] = pdfLength;
    pushPart(`${index + 1} 0 obj\n`);
    obj.forEach((part) => pushPart(part));
    pushPart("\nendobj\n");
  });
  const xrefStart = pdfLength;
  pushPart(`xref\n0 ${objects.length + 1}\n`);
  pushPart(`0000000000 65535 f \n`);
  for (let i = 1; i <= objects.length; i += 1) {
    pushPart(`${String(offsets[i]).padStart(10, "0")} 00000 n \n`);
  }
  pushPart(`trailer\n<< /Size ${objects.length + 1} /Root ${catalogObj} 0 R >>\nstartxref\n${xrefStart}\n%%EOF`);
  return new Blob(pdfParts, { type: "application/pdf" });
}

async function downloadReportPdfPayload(payload) {
  if (!payload) return;
  const blob = await buildReportPdfBlob(payload);
  const reportId = payload.report_id ? `report-${payload.report_id}` : "report";
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${reportId}.pdf`;
  anchor.rel = "noopener";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1500);
}

async function loadImageAsJpegData(src) {
  const response = await fetch(src);
  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  try {
    const image = await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = objectUrl;
    });
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth || image.width;
    canvas.height = image.naturalHeight || image.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(image, 0, 0);
    const jpegUrl = canvas.toDataURL("image/jpeg", 0.9);
    const base64 = jpegUrl.split(",")[1] || "";
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return { width: canvas.width, height: canvas.height, bytes };
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

function fitWithinBox(width, height, maxWidth, maxHeight) {
  const ratio = Math.min(maxWidth / width, maxHeight / height, 1);
  return {
    width: width * ratio,
    height: height * ratio,
  };
}

function buildPrintableReportHtml(payload) {
  if (!payload) return "";
  const reports = getAnimalReports(payload);
  const activeReport = reports[0] || {};
  const score = getReportHealthScore(activeReport, payload);
  const isUnknown = isNotRecognizedReport(activeReport, payload) || score == null;
  const healthWord = getHealthWord(score, activeReport.health_status || payload.health_status);
  const urgency = score != null ? (activeReport.urgency_label || payload.urgency_label || (activeReport.needs_rescue ? "Urgent" : "Non-urgent")) : "Not recognized";
  const animalName = titleCase(activeReport.animal_name || payload.animal_name || activeReport.animal_type || payload.animal_type || "Unknown animal");
  const animalType = titleCase(activeReport.animal_type || payload.animal_type || "Animal");
  const conditionSummary = activeReport.condition_summary || payload.condition_summary || "";
  const whatIsWrong = activeReport.what_is_wrong || payload.what_is_wrong || "No plain-language problem summary is available.";
  const emergencyPlan = activeReport.emergency_plan || payload.emergency_plan || {};
  const emergencySummary = emergencyPlan.summary || payload.triage_reasoning || "No emergency summary available.";
  const immediateSteps = Array.isArray(emergencyPlan.immediate_steps) && emergencyPlan.immediate_steps.length
    ? emergencyPlan.immediate_steps
    : (Array.isArray(payload.recommended_actions) ? payload.recommended_actions : []);
  const avoidSteps = Array.isArray(emergencyPlan.avoid_steps) ? emergencyPlan.avoid_steps : [];
  const locationText = [payload.location_address, payload.location_name].filter(Boolean).join(" - ") || "Location unavailable";
  const reportDate = formatDateTime(payload.created_at || new Date());
  const previewUrl = payload.image_url || (payload.image_path ? `/${buildReportImageUrl(payload.image_path)}` : "");
  const confidence = formatAiConfidence(activeReport, payload);
  const html = `
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <base href="${esc(window.location.origin + "/")}">
      <title>Paw Connect Report</title>
      <style>
        :root { color-scheme: light; }
        body {
          font-family: Arial, sans-serif;
          margin: 0;
          padding: 28px;
          color: #1f2937;
          background: #fff;
        }
        .report-shell {
          max-width: 900px;
          margin: 0 auto;
        }
        .report-top {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 16px;
          margin-bottom: 18px;
        }
        .report-title {
          font-size: 28px;
          margin: 0 0 4px;
        }
        .report-subtitle {
          margin: 0;
          color: #6b7280;
        }
        .badge {
          display: inline-block;
          padding: 6px 10px;
          border-radius: 999px;
          font-weight: 700;
          font-size: 12px;
          background: #eef7ea;
          color: #3b6d11;
        }
        .badge.mild { background: #faeeda; color: #854f0b; }
        .badge.serious { background: #fcebeb; color: #a32d2d; }
        .preview {
          width: 100%;
          max-height: 260px;
          object-fit: contain;
          border-radius: 14px;
          background: #f8fafc;
          margin: 0 0 10px;
        }
        .ai-note {
          text-align: right;
          color: #6b7280;
          font-size: 11px;
          margin: 0 0 14px;
        }
        .vitals {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 12px;
          margin: 18px 0;
        }
        .vital {
          border: 1px solid #e5e7eb;
          border-radius: 14px;
          padding: 12px 14px;
          background: #fff;
        }
        .vital-label {
          display: block;
          font-size: 11px;
          color: #6b7280;
          text-transform: uppercase;
          letter-spacing: .05em;
          margin-bottom: 6px;
        }
        .vital-value {
          font-size: 14px;
          font-weight: 700;
        }
        .section {
          border: 1px solid #e5e7eb;
          border-radius: 14px;
          padding: 14px 16px;
          margin: 0 0 14px;
        }
        .section h3 {
          margin: 0 0 8px;
          font-size: 16px;
        }
        .section p, .section li {
          color: #374151;
          line-height: 1.55;
        }
        .footer {
          margin-top: 18px;
          display: flex;
          justify-content: space-between;
          gap: 12px;
          color: #6b7280;
          font-size: 12px;
        }
        @page { margin: 14mm; }
      </style>
    </head>
    <body>
      <div class="report-shell">
        ${previewUrl ? `<img class="preview" src="${esc(previewUrl)}" alt="${esc(animalName)}">` : ""}
        ${previewUrl ? `<div class="ai-note">${esc(confidence)}</div>` : ""}
        <div class="report-top">
          <div>
            <h1 class="report-title">${esc(animalName)}</h1>
            <p class="report-subtitle">${esc(animalType)}</p>
          </div>
          <div class="badge ${esc(scoreClass(score))}">${esc(healthWord)}</div>
        </div>
        <div class="vitals">
          <div class="vital"><span class="vital-label">Health</span><span class="vital-value">${esc(healthWord)}</span></div>
          <div class="vital"><span class="vital-label">Help Type</span><span class="vital-value">${esc(score != null ? formatHelpType(activeReport.help_type || payload.help_type || "none") : "Not recognized")}</span></div>
          <div class="vital"><span class="vital-label">Urgency</span><span class="vital-value">${esc(urgency)}</span></div>
        </div>
        <div class="section">
          <h3>Condition Summary</h3>
          <p>${esc(isUnknown ? "The animal could not be recognized from this image." : (conditionSummary || "No condition summary available."))}</p>
        </div>
        <div class="section">
          <h3>What’s Wrong</h3>
          <p>${esc(isUnknown ? "The animal could not be recognized from this image." : whatIsWrong)}</p>
        </div>
        <div class="section">
          <h3>Emergency SOS</h3>
          <p>${esc(emergencySummary)}</p>
          ${immediateSteps.length ? `<ol>${immediateSteps.map((step) => `<li>${esc(step)}</li>`).join("")}</ol>` : ""}
          ${avoidSteps.length ? `<ul>${avoidSteps.map((step) => `<li>${esc(step)}</li>`).join("")}</ul>` : ""}
        </div>
        <div class="section">
          <h3>Location</h3>
          <p>${esc(locationText)}</p>
        </div>
        <div class="footer">
          <span>${esc(reportDate)}</span>
          <span>${esc(payload.report_id ? `Report #${payload.report_id}` : "Paw Connect")}</span>
        </div>
      </div>
    </body>
    </html>
  `;
  return html;
}

function openPrintableReportPayload(payload, autoPrint = true) {
  if (!payload) return;
  const html = buildPrintableReportHtml(payload);
  if (!html) return;
  const existingFrame = document.getElementById("reportPrintFrame");
  if (existingFrame) existingFrame.remove();
  const iframe = document.createElement("iframe");
  iframe.id = "reportPrintFrame";
  iframe.title = "Printable report";
  iframe.style.position = "fixed";
  iframe.style.right = "0";
  iframe.style.bottom = "0";
  iframe.style.width = "1px";
  iframe.style.height = "1px";
  iframe.style.border = "0";
  iframe.style.opacity = "0";
  iframe.style.pointerEvents = "none";
  iframe.setAttribute("aria-hidden", "true");
  iframe.srcdoc = html;
  const cleanup = () => {
    window.setTimeout(() => iframe.remove(), 1000);
  };
  iframe.onload = () => {
    if (!autoPrint) return;
    const frameWindow = iframe.contentWindow;
    if (!frameWindow) {
      cleanup();
      return;
    }
    const triggerPrint = () => {
      frameWindow.focus();
      frameWindow.print();
      cleanup();
    };
    window.setTimeout(triggerPrint, 300);
  };
  document.body.appendChild(iframe);
}

async function sendReportPayload(payload) {
  if (!payload) return;
  const text = buildReportSummaryText(payload);
  const shareData = {
    title: "Paw Connect Report",
    text,
  };
  try {
    if (navigator.share) {
      await navigator.share(shareData);
      return;
    }
  } catch {
    // Fallback below.
  }
  try {
    await navigator.clipboard.writeText(text);
    window.alert("Report text copied. You can paste it into WhatsApp, email, or SMS.");
  } catch {
    window.prompt("Copy report text:", text);
  }
}

function buildReportImageUrl(imagePath) {
  if (!imagePath) return "";
  return imagePath.startsWith("/") ? imagePath.slice(1) : imagePath;
}

function buildModalConditionSummary(report, status, detectedConditions) {
  if (status === "Healthy") {
    return `The ${titleCase(report.animal_type || "animal")} appears stable with no strong visible injury markers in the stored report.`;
  }
  if (detectedConditions.length) {
    return `Visible concerns were noted for this ${titleCase(report.animal_type || "animal")}: ${detectedConditions.map(formatCondition).join(", ")}.`;
  }
  if (status === "Mildly Injured") {
    return `The report suggests a mild visible issue that should be monitored and checked by a vet if the animal looks distressed.`;
  }
  if (status === "Seriously Injured") {
    return `The report suggests a serious condition and urgent rescue or veterinary attention may be required.`;
  }
  return `The stored report could not provide a more specific condition summary for this animal.`;
}

function getDefaultGuidance(status) {
  if (status === "Healthy") {
    return "The animal appears fine in this report. Keep a safe distance, observe calmly, and only intervene if the condition changes.";
  }
  if (status === "Mildly Injured") {
    return "The animal may have a mild issue. Keep the area safe, avoid stressing it, and consider contacting a nearby vet or rescue.";
  }
  if (status === "Seriously Injured") {
    return "The animal may need urgent help. Contact rescue or a nearby vet immediately and avoid moving it unless necessary for safety.";
  }
  return "No additional clinical note was stored for this report.";
}

function getStatusActions(report) {
  const status = normalizeHealthStatus(report.health_status);
  if (status === "Healthy") {
    return [
      "Observe the animal from a safe distance for a few minutes.",
      "Do not crowd or force contact if the animal is calm and mobile.",
      "If behaviour changes or distress appears later, rescan or contact a nearby vet."
    ];
  }
  if (status === "Mildly Injured") {
    return [
      "Keep the animal away from traffic, people, and other animals if possible.",
      "Contact a nearby vet or rescue team for guidance if limping, swelling, or visible wounds continue.",
      "Offer help only if it can be done safely without causing more stress."
    ];
  }
  if (status === "Seriously Injured") {
    return [
      "Call a rescue team or nearby vet immediately.",
      "Keep the animal warm, quiet, and away from further danger until help arrives.",
      "Do not attempt aggressive handling unless there is an immediate life-threatening risk."
    ];
  }
  return [
    "Review the report carefully and monitor the animal.",
    "Use Care Bot or contact a vet if you are unsure about the next step."
  ];
}

function formatRescueStatus(status) {
  if (!status) return "Not requested";
  return titleCase(String(status).replaceAll("_", " "));
}

function normalizeHealthStatus(status) {
  if (status === "Serious") return "Seriously Injured";
  if (status === "Mild") return "Mildly Injured";
  if (status === "healthy") return "Healthy";
  if (status === "mild_injury" || status === "moderate_injury") return "Mildly Injured";
  if (status === "severe_injury" || status === "critical") return "Seriously Injured";
  return status || "Unknown";
}

function formatHealthStatusLabel(status) {
  if (status === "Seriously Injured") return "Serious";
  if (status === "Mildly Injured") return "Mild";
  if (status === "Healthy") return "Healthy";
  if (status === "NotApplicable") return "Not Applicable";
  return titleCase(String(status || "Unknown").replaceAll("_", " "));
}

function formatHelpType(value) {
  return titleCase(String(value || "none").replaceAll("_", " "));
}

function formatCondition(value) {
  return titleCase(String(value || "").replaceAll("_", " "));
}

function formatBotText(text) {
  return esc(text).replaceAll("\n", "<br>");
}

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function setSectionVisibility(id, visible) {
  const node = document.getElementById(id);
  if (!node) return;
  const section = node.closest(".report-section") || node;
  if (visible) section.classList.remove("hidden");
  else section.classList.add("hidden");
}

function setNum(id, value) {
  document.getElementById(id).textContent = String(value);
}

function titleCase(value) {
  return String(value)
    .split(" ")
    .map((part) => part ? part[0].toUpperCase() + part.slice(1) : part)
    .join(" ");
}

function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("en-IN");
}

function formatDateTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("en-GB", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).replace(",", "");
}

function esc(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

