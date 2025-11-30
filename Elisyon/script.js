// ================= GLOBAL CONFIG =================
const API_BASE = window.location.origin;         // FastAPI root (works on localhost + ngrok)
const PREDICT_URL = `${API_BASE}/predict`;        // POST image + species/mode
const SIGNUP_URL = `${API_BASE}/auth/signup`;     // POST JSON { name, email, password }
const LOGIN_URL = `${API_BASE}/auth/login`;       // POST JSON { email, password }
const HISTORY_URL = `${API_BASE}/history`;        // GET (auth)
const ME_URL = `${API_BASE}/auth/me`;             // GET (auth) to verify token

const TOKEN_KEY = "eliyon_token";

// ================= GLOBAL STATE =================
let currentSpecies = "baby";
let currentMode = "emotion";
let currentInputMode = "upload"; // "upload" | "camera"
let selectedFile = null;
let cameraStream = null;
let authToken = null;
let currentUser = null;

// ================= TOKEN HELPERS =================
function saveToken(token) {
  authToken = token || null;
  if (token) {
    localStorage.setItem(TOKEN_KEY, token);
  } else {
    localStorage.removeItem(TOKEN_KEY);
  }
}

function getStoredToken() {
  return localStorage.getItem(TOKEN_KEY);
}

// ================= SMOOTH SCROLL =================
function scrollToSection(id) {
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: "smooth" });
}
window.scrollToSection = scrollToSection;

// ================= MOBILE NAV =================
const navToggle = document.getElementById("navToggle");
const navMobile = document.getElementById("navMobile");

if (navToggle && navMobile) {
  navToggle.addEventListener("click", () => {
    navMobile.classList.toggle("open");
  });
}

function closeMobileNav() {
  if (navMobile) navMobile.classList.remove("open");
}
window.closeMobileNav = closeMobileNav;

// ================= SPECIES CARD REVEAL =================
const speciesCards = document.querySelectorAll(".species-card");
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("in-view");
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.3 }
);
speciesCards.forEach((card) => observer.observe(card));

// ================= TABS: SPECIES + MODE =================
const speciesTabButtons = document.querySelectorAll(".tab-btn");
const modeTabButtons = document.querySelectorAll(".mode-btn");
const demoTitleSpecies = document.getElementById("demoTitleSpecies");
const demoTitleMode = document.getElementById("demoTitleMode");

function updateDemoTitle() {
  const speciesText =
    currentSpecies === "baby"
      ? "Kids"
      : currentSpecies === "dog"
      ? "Dogs"
      : "Cats";

  const modeText = currentMode === "emotion" ? "Emotion Scan" : "Health Scan";

  if (demoTitleSpecies) demoTitleSpecies.textContent = speciesText;
  if (demoTitleMode) demoTitleMode.textContent = modeText;
}

speciesTabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    speciesTabButtons.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentSpecies = btn.dataset.species;
    updateDemoTitle();
  });
});

modeTabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    modeTabButtons.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentMode = btn.dataset.mode;
    updateDemoTitle();
  });
});

// Called from species card buttons in HTML
function openDemo(species, mode) {
  currentSpecies = species;
  currentMode = mode;

  speciesTabButtons.forEach((b) =>
    b.classList.toggle("active", b.dataset.species === species)
  );
  modeTabButtons.forEach((b) =>
    b.classList.toggle("active", b.dataset.mode === mode)
  );

  updateDemoTitle();
  scrollToSection("demo");
}
window.openDemo = openDemo;

// ================= INPUT TOGGLE (UPLOAD vs CAMERA) =================
const inputButtons = document.querySelectorAll(".input-btn");
const inputModes = document.querySelectorAll(".input-mode");

inputButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    inputButtons.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");

    const type = btn.dataset.input; // "upload" | "camera"
    currentInputMode = type;

    inputModes.forEach((mode) => {
      const isTarget =
        (type === "upload" && mode.classList.contains("input-upload")) ||
        (type === "camera" && mode.classList.contains("input-camera"));
      mode.classList.toggle("active", isTarget);
    });

    if (type === "camera") {
      // camera will actually start when user clicks Start Camera button
    } else {
      stopCamera();
    }
  });
});

// ================= FILE UPLOAD + DROPZONE =================
const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const fileNameLabel = document.getElementById("fileName");

if (dropzone && fileInput) {
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith("image/")) {
        selectedFile = file;
        if (fileNameLabel) fileNameLabel.textContent = file.name;
      }
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      selectedFile = fileInput.files[0];
      if (fileNameLabel) fileNameLabel.textContent = selectedFile.name;
    } else {
      selectedFile = null;
      if (fileNameLabel) fileNameLabel.textContent = "No file selected";
    }
  });
}

// ================= CAMERA: LIVE PREVIEW (Rear camera preferred) =================
const cameraVideo = document.getElementById("cameraVideo");
const cameraCanvas = document.getElementById("cameraCanvas");
const startCameraBtn = document.getElementById("startCameraBtn");

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setScanStatus("Camera API not supported in this browser.", true);
    return;
  }
  if (cameraStream) return; // already running

  try {
    // Try to use rear/back camera first (phones/tablets)
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { exact: "environment" } },
        audio: false,
      });
    } catch (err) {
      console.warn("Rear camera not available, falling back to front camera.", err);
      // Fallback to generic camera (front or default webcam)
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
    }

    if (cameraVideo) {
      cameraVideo.srcObject = cameraStream;
      cameraVideo.play().catch(() => {});
    }
  } catch (err) {
    console.error("Error starting camera:", err);
    setScanStatus("Could not access camera. Check permissions.", true);
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = null;
  }
  if (cameraVideo) {
    cameraVideo.srcObject = null;
  }
}

// capture still image from live preview
async function captureFromCameraPreview() {
  if (!cameraVideo || !cameraCanvas || !cameraStream) {
    throw new Error("Camera is not started.");
  }

  const w = cameraVideo.videoWidth;
  const h = cameraVideo.videoHeight;
  if (!w || !h) {
    throw new Error("Camera not ready yet. Wait a second.");
  }

  cameraCanvas.width = w;
  cameraCanvas.height = h;
  const ctx = cameraCanvas.getContext("2d");
  ctx.drawImage(cameraVideo, 0, 0, w, h);

  const blob = await new Promise((resolve) =>
    cameraCanvas.toBlob((b) => resolve(b), "image/jpeg", 0.9)
  );
  if (!blob) {
    throw new Error("Failed to capture image from camera.");
  }

  return {
    file: new File([blob], "camera_capture.jpg", { type: "image/jpeg" }),
    url: URL.createObjectURL(blob),
  };
}

// ================= STATUS & RESULT RENDERING =================
const resultCard = document.getElementById("resultCard");
const resultBadges = document.getElementById("resultBadges");
const runScanBtn = document.getElementById("runScanBtn");
const scanStatus = document.getElementById("scanStatus");

function setScanStatus(msg, isError = false) {
  if (!scanStatus) return;
  scanStatus.textContent = msg || "";
  scanStatus.style.color = isError ? "#ff6b6b" : "var(--text-muted)";
}

function renderResult(data, isMock = false, snapshotUrl = null) {
  if (!resultCard || !data) return;

  const {
    species,
    mode,
    label,
    confidence,
    severity,
    ai_insight,
    image_url,
    breakdown,
  } = data;

  const sev = (severity || "clear").toLowerCase();
  let sevLabel = "Clear";
  if (sev === "mild") sevLabel = "Mild";
  else if (sev === "moderate") sevLabel = "Moderate";
  else if (sev === "severe") sevLabel = "Severe";

  const displayImg = snapshotUrl || image_url || null;

  resultCard.innerHTML = `
    <div class="result-main">
      ${
        displayImg
          ? `<div class="result-image-wrapper">
               <img src="${displayImg}" alt="Scan image" class="result-image" />
             </div>`
          : ""
      }
      <div class="result-text">
        <h4 class="result-label">${label || "Unknown"}</h4>
        <p class="result-meta">
          <span>${species || "baby"} • ${mode || "emotion"}</span>
          ${
            typeof confidence === "number"
              ? `<span>Confidence: ${(confidence * 100).toFixed(1)}%</span>`
              : ""
          }
        </p>
        <p class="result-insight">${ai_insight || "No explanation available."}</p>
        ${
          breakdown
            ? `<pre class="result-breakdown">${JSON.stringify(
                breakdown,
                null,
                2
              )}</pre>`
            : ""
        }
        ${
          isMock
            ? `<p class="demo-note" style="margin-top:8px;">Mock result (no backend / no login).</p>`
            : ""
        }
      </div>
    </div>
  `;

  if (resultBadges) {
    resultBadges.innerHTML = `
      <span class="badge badge-small">Severity: ${sevLabel}</span>
      <span class="badge badge-small">Mode: ${mode}</span>
      <span class="badge badge-small">Species: ${species}</span>
    `;
  }
}

function resetDemo() {
  if (resultCard) {
    resultCard.innerHTML = `
      <div class="result-placeholder">
        <i class="ri-magic-line"></i>
        <p>Run a scan to see predictions here.</p>
      </div>
    `;
  }
  if (resultBadges) resultBadges.innerHTML = "";
  if (fileNameLabel) fileNameLabel.textContent = "No file selected";
  if (fileInput) fileInput.value = "";
  selectedFile = null;
  setScanStatus("");
}
window.resetDemo = resetDemo;

// ================= MOCK RESULT (if not logged in) =================
function mockResult() {
  const emotionsBaby = ["Happy", "Crying", "Angry", "Calm"];
  const emotionsPet = ["Relaxed", "Anxious", "Alert", "Playful"];
  const healthLabels = [
    "Healthy",
    "Mild rash",
    "Allergy (unlikely)",
    "Ear infection (mild)",
  ];

  const isEmotion = currentMode === "emotion";
  const isBaby = currentSpecies === "baby";

  const label = isEmotion
    ? isBaby
      ? emotionsBaby[Math.floor(Math.random() * emotionsBaby.length)]
      : emotionsPet[Math.floor(Math.random() * emotionsPet.length)]
    : healthLabels[Math.floor(Math.random() * healthLabels.length)];

  const severity =
    label.toLowerCase().includes("healthy") ||
    label.toLowerCase().includes("calm")
      ? "clear"
      : label.toLowerCase().includes("mild")
      ? "mild"
      : "moderate";

  const conf = 0.7 + Math.random() * 0.2;

  const ai_insight = isEmotion
    ? "This is a mock emotion result. In real mode, Eliyon would use your YOLO emotion model here."
    : "This is a mock health result. In real mode, Eliyon would use your YOLO health models.";

  return {
    species: currentSpecies,
    mode: currentMode,
    label,
    confidence: conf,
    severity,
    ai_insight,
  };
}

// ================= HISTORY UI =================
const historyContainer = document.getElementById("historyContainer");

async function loadHistory() {
  if (!historyContainer) return;

  if (!authToken) {
    historyContainer.innerHTML = `
      <div class="result-placeholder">
        <i class="ri-time-line"></i>
        <p>Log in and run a few scans. Your past results will appear here.</p>
      </div>
    `;
    return;
  }

  try {
    const res = await fetch(HISTORY_URL, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${authToken}`,
      },
    });

    if (res.status === 401) {
      saveToken(null);
      historyContainer.innerHTML = `
        <div class="result-placeholder">
          <i class="ri-time-line"></i>
          <p>Your session expired. Please log in again to view history.</p>
        </div>
      `;
      return;
    }

    if (!res.ok) {
      throw new Error("Failed to fetch history");
    }

    const items = await res.json();
    renderHistory(items);
  } catch (err) {
    console.error("Error loading history:", err);
    historyContainer.innerHTML = `
      <div class="result-placeholder">
        <i class="ri-alert-line"></i>
        <p>Could not load history. Please check if the backend is running.</p>
      </div>
    `;
  }
}

function renderHistory(items) {
  if (!historyContainer) return;

  if (!Array.isArray(items) || items.length === 0) {
    historyContainer.innerHTML = `
      <div class="result-placeholder">
        <i class="ri-time-line"></i>
        <p>No scans found yet. Run a scan while logged in to see history here.</p>
      </div>
    `;
    return;
  }

  const cardsHtml = items
    .map((item) => {
      const label = item.label || "Unknown";
      const species = item.species || "unknown";
      const mode = item.mode || "unknown";
      const severity = item.severity || "clear";
      const confidence =
        typeof item.confidence === "number"
          ? Math.round(item.confidence * 100)
          : null;
      const aiInsight = item.ai_insight || "";
      const tsRaw = item.created_at || "";
      let ts = "";
      if (tsRaw) {
        const d = new Date(tsRaw);
        ts = !isNaN(d.getTime()) ? d.toLocaleString() : tsRaw;
      } else {
        ts = "N/A";
      }
      const imageUrl = item.image_url || null;

      let severityText = "No strong concern";
      let severityClass = "badge-severity-clear";
      if (severity === "mild") {
        severityText = "Mild concern";
        severityClass = "badge-severity-mild";
      } else if (severity === "moderate") {
        severityText = "Moderate concern";
        severityClass = "badge-severity-moderate";
      } else if (severity === "severe") {
        severityText = "High concern";
        severityClass = "badge-severity-severe";
      }

      return `
        <div class="history-card">
          <div class="history-main">
            ${
              imageUrl
                ? `<div class="history-image-wrapper">
                     <img src="${imageUrl}" alt="Scan image" class="history-image" />
                   </div>`
                : ""
            }
            <div class="history-text">
              <h4>${label}</h4>
              <p class="history-meta">
                <span>${species} • ${mode}</span>
                ${
                  confidence !== null
                    ? `<span>Conf: ${confidence}%</span>`
                    : ""
                }
              </p>
              <p class="history-insight">${aiInsight}</p>
              <p class="history-date">${ts}</p>
              <div style="display:flex; gap:6px; flex-wrap:wrap; margin-top:6px;">
                <span class="badge-pill ${severityClass}">${severityText}</span>
              </div>
            </div>
          </div>
        </div>
      `;
    })
    .join("");

  historyContainer.innerHTML = cardsHtml;
}

// ================= RUN SCAN =================
async function handleRunScan() {
  setScanStatus("");

  if (!authToken) {
    // Not logged in: show mock result but no backend usage
    setScanStatus("You are not logged in. Showing mock result.", true);
    const m = mockResult();
    renderResult(m, true);
    return;
  }

  let fileToSend = null;
  let snapshotUrl = null;

  try {
    if (currentInputMode === "upload") {
      if (!selectedFile) {
        setScanStatus("Please select a photo first.", true);
        return;
      }
      fileToSend = selectedFile;
    } else {
      // Camera mode: capture from live preview
      const capture = await captureFromCameraPreview();
      fileToSend = capture.file;
      snapshotUrl = capture.url;
      if (fileNameLabel) fileNameLabel.textContent = "Captured from camera";
    }

    if (!fileToSend) {
      setScanStatus("No image available to scan.", true);
      return;
    }

    if (runScanBtn) runScanBtn.disabled = true;
    setScanStatus("Running AI scan...");

    const formData = new FormData();
    formData.append("file", fileToSend);
    formData.append("species", currentSpecies);
    formData.append("mode", currentMode);

    const res = await fetch(PREDICT_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${authToken}`,
      },
      body: formData,
    });

    if (res.status === 401) {
      saveToken(null);
      currentUser = null;
      updateAuthUI();
      setScanStatus("Token invalid or expired. Please log in again.", true);
      return;
    }

    if (!res.ok) {
      console.error("Backend error:", await res.text());
      throw new Error("Backend returned an error");
    }

    const data = await res.json();
    renderResult(data, false, snapshotUrl);
    setScanStatus("Scan complete.");
    loadHistory();
  } catch (err) {
    console.error("Scan error:", err);
    setScanStatus("Backend not reachable. Showing mock result.", true);
    const m = mockResult();
    renderResult(m, true, snapshotUrl);
  } finally {
    if (runScanBtn) runScanBtn.disabled = false;
  }
}

// ================= AUTH: LOGIN / SIGNUP / LOGOUT =================
const loginForm = document.getElementById("loginForm");
const signupForm = document.getElementById("signupForm");
const loginStatus = document.getElementById("loginStatus");
const signupStatus = document.getElementById("signupStatus");

// navbar user UI
const navUser = document.getElementById("navUser");
const navUserName = document.getElementById("navUserName");
const logoutBtn = document.getElementById("logoutBtn");

function updateAuthUI() {
  if (currentUser && authToken) {
    if (navUser) navUser.style.display = "flex";
    if (navUserName) {
      navUserName.textContent = currentUser.name || currentUser.email || "User";
    }
    if (loginStatus) loginStatus.textContent = `Logged in as ${currentUser.email}`;
    if (signupStatus && signupStatus.textContent.startsWith("Account created")) {
      // keep that message
    }
    loadHistory();
  } else {
    if (navUser) navUser.style.display = "none";
    if (loginStatus) loginStatus.textContent = "";
    if (signupStatus && !signupStatus.textContent.startsWith("Account created")) {
      signupStatus.textContent = "";
    }
    if (historyContainer) {
      historyContainer.innerHTML = `
        <div class="result-placeholder">
          <i class="ri-time-line"></i>
          <p>Log in and run a few scans. Your past results will appear here.</p>
        </div>
      `;
    }
  }
}

if (loginForm) {
  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (loginStatus) loginStatus.textContent = "Logging in...";

    const email = document.getElementById("loginEmail")?.value.trim();
    const password = document.getElementById("loginPassword")?.value;

    if (!email || !password) {
      if (loginStatus)
        loginStatus.textContent = "Please enter email and password.";
      return;
    }

    try {
      const res = await fetch(LOGIN_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg || "Login failed");
      }

      const data = await res.json();
      const token = data.access_token;
      if (!token) throw new Error("No token received from server.");

      // save token & use it immediately
      saveToken(token);

      const meRes = await fetch(ME_URL, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!meRes.ok) throw new Error("Could not fetch user info.");
      currentUser = await meRes.json();

      if (loginStatus) loginStatus.textContent = `Logged in as ${currentUser.email}`;
      if (signupStatus && !signupStatus.textContent.startsWith("Account created")) {
        signupStatus.textContent = "";
      }

      updateAuthUI();
      scrollToSection("demo");
    } catch (err) {
      console.error("Login error:", err);
      if (loginStatus)
        loginStatus.textContent =
          "Login failed. Please check credentials or backend.";
    }
  });
}

if (signupForm) {
  signupForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (signupStatus) signupStatus.textContent = "Creating account...";

    const name = document.getElementById("signupName")?.value.trim();
    const email = document.getElementById("signupEmail")?.value.trim();
    const password = document.getElementById("signupPassword")?.value;

    if (!email || !password) {
      if (signupStatus)
        signupStatus.textContent = "Email and password are required.";
      return;
    }

    try {
      const res = await fetch(SIGNUP_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email, password }),
      });

      if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg || "Signup failed");
      }

      const data = await res.json();

      // Support BOTH behaviors:
      // 1) Backend returns access_token (auto-login)
      // 2) Backend returns only { message, user_id } (manual login)
      if (data.access_token) {
        const token = data.access_token;
        saveToken(token);

        const meRes = await fetch(ME_URL, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!meRes.ok) throw new Error("Could not fetch user info after signup");
        currentUser = await meRes.json();

        if (signupStatus)
          signupStatus.textContent = "Account created & logged in.";
        if (loginStatus) loginStatus.textContent = "";

        updateAuthUI();
        scrollToSection("demo");
      } else {
        // No token returned -> just show success, let user log in manually
        if (signupStatus)
          signupStatus.textContent =
            data.message || "Account created. Please log in now.";
      }
    } catch (err) {
      console.error("Signup error:", err);
      if (signupStatus)
        signupStatus.textContent =
          "Signup failed. Email may already exist or backend is down.";
    }
  });
}

if (logoutBtn) {
  logoutBtn.addEventListener("click", () => {
    saveToken(null);
    currentUser = null;
    updateAuthUI();
  });
}

// ================= FAQ ACCORDION =================
const faqItems = document.querySelectorAll(".faq-item");

faqItems.forEach((item) => {
  const question = item.querySelector(".faq-question");
  const answer = item.querySelector(".faq-answer");
  question.addEventListener("click", () => {
    const isOpen = item.classList.contains("open");
    faqItems.forEach((i) => {
      i.classList.remove("open");
      const ans = i.querySelector(".faq-answer");
      ans.style.maxHeight = "0px";
      ans.style.paddingBottom = "0px";
    });
    if (!isOpen) {
      item.classList.add("open");
      answer.style.maxHeight = answer.scrollHeight + "px";
      answer.style.paddingBottom = "10px";
    }
  });
});

// ================= INIT =================
document.addEventListener("DOMContentLoaded", () => {
  updateDemoTitle();

  if (startCameraBtn) {
    startCameraBtn.addEventListener("click", () => {
      startCamera();
    });
  }

  if (runScanBtn) {
    runScanBtn.addEventListener("click", () => {
      handleRunScan();
    });
  }

  // Load token from storage & verify
  const storedToken = getStoredToken();
  if (storedToken) {
    authToken = storedToken;
    fetch(ME_URL, {
      headers: { Authorization: `Bearer ${authToken}` },
    })
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data && data.email) {
          currentUser = data;
          updateAuthUI();
        } else {
          saveToken(null);
          currentUser = null;
          updateAuthUI();
        }
      })
      .catch((err) => {
        console.error("Error verifying token:", err);
        saveToken(null);
        currentUser = null;
        updateAuthUI();
      });
  } else {
    updateAuthUI();
  }
});
