// Initialize Telegram WebApp SDK
const tg = window.Telegram?.WebApp;
if (tg) {
  tg.ready();
  tg.expand();
  // Apply Telegram theme colors
  document.documentElement.style.setProperty('--bg-color', tg.themeParams.bg_color || '#050811');
}

// Global States
let activeTab = 'quick';
let quickPhotoBase64 = null;
let selectedDeepSlot = null;

// Deep slots dictionary
let deepSlots = {
  "Frontal": null,
  "Left Semi-profile": null,
  "Right Semi-profile": null,
  "Left Profile": null,
  "Right Profile": null
};

// Canvas references
const quickCanvas = document.getElementById('quick-canvas');
const quickCtx = quickCanvas.getContext('2d');
const quickPlaceholder = document.getElementById('quick-placeholder');

// --- Tab Switcher ---
function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
  
  if (tab === 'quick') {
    document.querySelector('.tab-btn:nth-child(1)').classList.add('active');
    document.getElementById('tab-quick').classList.add('active');
  } else {
    document.querySelector('.tab-btn:nth-child(2)').classList.add('active');
    document.getElementById('tab-deep').classList.add('active');
  }
}

// --- File Trigger Helper ---
function triggerFileInput(id) {
  document.getElementById(id).click();
}

// --- Quick Photo Handling ---
function loadQuickPhoto(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function(e) {
    quickPhotoBase64 = e.target.result;
    
    // Draw on Canvas
    const img = new Image();
    img.onload = function() {
      quickCanvas.width = img.width;
      quickCanvas.height = img.height;
      quickCtx.clearRect(0, 0, img.width, img.height);
      quickCtx.drawImage(img, 0, 0);
      quickPlaceholder.style.display = 'none';
      document.getElementById('btn-quick-run').disabled = false;
    };
    img.src = quickPhotoBase64;
  };
  reader.readAsDataURL(file);
}

// --- Run Quick Analysis API ---
async function runQuickAnalysis() {
  if (!quickPhotoBase64) return;

  const btn = document.getElementById('btn-quick-run');
  const calibration = document.getElementById('group-select').value;
  btn.disabled = true;
  btn.innerText = "ANALYZING...";

  try {
    const response = await fetch('/analyze-frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_base64: quickPhotoBase64,
        target_group: calibration,
        draw_hud: true
      })
    });

    const data = await response.json();
    displayQuickResults(data);
  } catch (err) {
    console.error(err);
    alert("Analysis failed. Make sure your local server is running and accessible.");
  } finally {
    btn.disabled = false;
    btn.innerText = "ANALYZE FACE";
  }
}

// Display results of Quick Analysis
function displayQuickResults(data) {
  const metrics = data.metrics;
  if (!metrics || !metrics.detected) {
    alert("No face detected! Please use a clear, front-facing selfie.");
    return;
  }

  // Draw HUD frame returned by server onto canvas
  const img = new Image();
  img.onload = function() {
    quickCanvas.width = img.width;
    quickCanvas.height = img.height;
    quickCtx.clearRect(0, 0, img.width, img.height);
    quickCtx.drawImage(img, 0, 0);
  };
  img.src = data.hud_image;

  document.getElementById('quick-results-section').style.display = 'flex';
  
  // Score & Badge
  document.getElementById('quick-score-val').innerText = metrics.ai_score.toFixed(2);
  const badge = document.getElementById('quick-rank-badge');
  badge.innerText = getTierText(metrics.ai_score);
  const color = getTierColor(metrics.ai_score);
  badge.style.borderColor = color;
  badge.style.color = color;

  // Potential & Percentile
  const geomFactor = metrics.overall_geom / 100.0;
  const potential = Math.min(10.0, metrics.ai_score + (10.0 - metrics.ai_score) * (0.20 + 0.25 * geomFactor));
  document.getElementById('quick-potential-val').innerText = `POTENTIAL: ${potential.toFixed(2)} / 10.0`;
  
  const topPct = getPercentile(metrics.ai_score);
  document.getElementById('quick-percentile-val').innerText = `TOP: ${topPct.toFixed(1)}%`;

  // Progress bars
  document.getElementById('quick-pb-sym').style.width = `${metrics.symmetry}%`;
  document.getElementById('quick-lbl-sym').innerText = `${metrics.symmetry.toFixed(1)}%`;

  document.getElementById('quick-pb-gr').style.width = `${metrics.golden_ratio}%`;
  document.getElementById('quick-lbl-gr').innerText = `${metrics.golden_ratio.toFixed(1)}%`;

  document.getElementById('quick-pb-geom').style.width = `${metrics.overall_geom}%`;
  document.getElementById('quick-lbl-geom').innerText = `${metrics.overall_geom.toFixed(1)}%`;

  // Log box
  let logs = `> [AI SCORE] Rating: ${metrics.ai_score.toFixed(2)}/10.0\n`;
  logs += `> [GEOMETRY] Analysis details:\n`;
  metrics.details.forEach(d => logs += `* ${d}\n`);
  document.getElementById('quick-log-box').innerText = logs;
}

// --- Deep Analysis Handling ---
function selectSlot(pose) {
  selectedDeepSlot = pose;
  // Reset border highlights
  document.querySelectorAll('.slot-card').forEach(card => card.classList.remove('active'));
  document.getElementById(`slot-${pose}`).classList.add('active');
  
  // Trigger file upload
  triggerFileInput('deep-file-input');
}

function loadDeepPhoto(event) {
  const file = event.target.files[0];
  if (!file || !selectedDeepSlot) return;

  const reader = new FileReader();
  reader.onload = async function(e) {
    const base64 = e.target.result;
    
    // Set slot visual to loading
    const slotCard = document.getElementById(`slot-${selectedDeepSlot}`);
    const scoreLbl = document.getElementById(`score-lbl-${selectedDeepSlot}`);
    scoreLbl.innerText = "Analyzing...";
    slotCard.classList.add('loaded');

    // Run quick analyze on slot to populate metrics and face crop
    const calibration = document.getElementById('group-select').value;
    try {
      const response = await fetch('/analyze-frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: base64,
          target_group: calibration,
          draw_hud: false
        })
      });

      const data = await response.json();
      if (!data.metrics || !data.metrics.detected) {
        alert("Face not detected in slot photo! Try another image.");
        scoreLbl.innerText = "Error";
        slotCard.classList.remove('loaded');
        deepSlots[selectedDeepSlot] = null;
      } else {
        deepSlots[selectedDeepSlot] = {
          faceCropB64: data.face_crop,
          metrics: data.metrics
        };
        scoreLbl.innerText = `${data.metrics.ai_score.toFixed(2)}/10.0`;
      }
    } catch (err) {
      console.error(err);
      scoreLbl.innerText = "Error";
      slotCard.classList.remove('loaded');
    }
    
    checkDeepReady();
  };
  reader.readAsDataURL(file);
}

function checkDeepReady() {
  const hasFrontal = deepSlots["Frontal"] !== null;
  const hasSide = Object.keys(deepSlots).some(k => k !== "Frontal" && deepSlots[k] !== null);
  document.getElementById('btn-deep-run').disabled = !(hasFrontal && hasSide);
}

function resetDeepSlots() {
  deepSlots = {
    "Frontal": null,
    "Left Semi-profile": null,
    "Right Semi-profile": null,
    "Left Profile": null,
    "Right Profile": null
  };
  document.querySelectorAll('.slot-card').forEach(card => {
    card.classList.remove('loaded');
    card.classList.remove('active');
  });
  document.querySelectorAll('.slot-score').forEach(lbl => lbl.innerText = "Empty");
  document.getElementById('btn-deep-run').disabled = true;
  document.getElementById('deep-results-section').style.display = 'none';
  document.getElementById('deep-console-card').style.display = 'none';
}

// --- Run Deep TTA Analysis ---
async function runDeepAnalysis() {
  const btn = document.getElementById('btn-deep-run');
  const calibration = document.getElementById('group-select').value;
  btn.disabled = true;

  document.getElementById('deep-console-card').style.display = 'flex';
  const progressInner = document.getElementById('deep-progress-inner');
  const progressLbl = document.getElementById('deep-progress-lbl');
  const consoleLog = document.getElementById('deep-console-log');
  
  consoleLog.innerText = "=== INITIALIZING DEEP ANALYSIS ===\n";
  
  const activePoses = Object.keys(deepSlots).filter(k => deepSlots[k] !== null);
  const total = activePoses.length;
  let step = 0;

  for (const pose of activePoses) {
    progressLbl.innerText = `Analyzing slot: ${pose}...`;
    consoleLog.innerText += `> [TTA] Initializing deep assessment on ${pose}\n`;
    
    const slot = deepSlots[pose];
    try {
      const response = await fetch('/predict-deep', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          face_crop_base64: slot.faceCropB64,
          target_group: calibration,
          is_webcam: false
        })
      });
      const data = await response.json();
      
      slot.metrics.ai_score = data.ai_score;
      slot.metrics.raw_score = data.raw_score;
      
      consoleLog.innerText += `  * TTA Configuration complete\n`;
      consoleLog.innerText += `  [RESULT] Score: ${data.ai_score.toFixed(2)}/10.0\n`;
      
      step++;
      progressInner.style.width = `${(step / total) * 100}%`;
    } catch (err) {
      consoleLog.innerText += `  [ERROR] Processing failed on slot ${pose}\n`;
    }
  }

  // Calculate Combined Certificate
  progressLbl.innerText = "Calculating combined certificate...";
  consoleLog.innerText += "\n[SYSTEM] Compiling combined metrics...\n";

  try {
    // Reformat slots payload
    const slotsPayload = {};
    Object.keys(deepSlots).forEach(k => {
      if (deepSlots[k]) {
        slotsPayload[k] = {
          score: deepSlots[k].metrics.ai_score,
          geom_score: deepSlots[k].metrics.overall_geom,
          symmetry: deepSlots[k].metrics.symmetry,
          golden_ratio: deepSlots[k].metrics.golden_ratio,
          details: deepSlots[k].metrics.details
        };
      } else {
        slotsPayload[k] = null;
      }
    });

    const response = await fetch('/calculate-combined', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        slots: slotsPayload,
        target_group: calibration
      })
    });
    
    const res = await response.json();
    displayDeepResults(res);
    progressLbl.innerText = "Complete!";
  } catch (err) {
    console.error(err);
    alert("Combined calculation failed.");
  } finally {
    btn.disabled = false;
  }
}

function displayDeepResults(res) {
  document.getElementById('deep-results-section').style.display = 'flex';
  
  document.getElementById('deep-score-val').innerText = res.ai_score.toFixed(2);
  const badge = document.getElementById('deep-rank-badge');
  badge.innerText = res.tier_text;
  badge.style.borderColor = res.tier_color;
  badge.style.color = res.tier_color;

  document.getElementById('deep-potential-val').innerText = `POTENTIAL: ${res.potential_score.toFixed(2)} / 10.0`;
  document.getElementById('deep-percentile-val').innerText = `TOP: ${res.top_pct.toFixed(1)}%`;

  // Bars
  document.getElementById('deep-pb-sym').style.width = `${res.symmetry}%`;
  document.getElementById('deep-lbl-sym').innerText = `${res.symmetry.toFixed(1)}%`;

  document.getElementById('deep-pb-gr').style.width = `${res.golden_ratio}%`;
  document.getElementById('deep-lbl-gr').innerText = `${res.golden_ratio.toFixed(1)}%`;

  document.getElementById('deep-pb-geom').style.width = `${res.geom_score}%`;
  document.getElementById('deep-lbl-geom').innerText = `${res.geom_score.toFixed(1)}%`;
}

// --- Help functions ---
function getTierText(score) {
  if (score < 3.0) return "SUB-3";
  if (score < 4.0) return "SUB";
  if (score < 5.0) return "LTN";
  if (score < 6.0) return "MTN";
  if (score < 7.0) return "HTN";
  if (score < 8.0) return "CHADLITE";
  return "CHAD";
}

function getTierColor(score) {
  if (score < 3.0) return "#ff3333";
  if (score < 4.0) return "#ff6666";
  if (score < 5.0) return "#ffcc00";
  if (score < 6.0) return "#00f0ff";
  if (score < 7.0) return "#00ff64";
  if (score < 8.0) return "#bf00ff";
  return "#ff007f";
}

function getPercentile(score) {
  const mean = 5.0;
  const std = 1.15;
  const z = (score - mean) / std;
  
  // Approximation of erf
  const t = 1.0 / (1.0 + 0.5 * Math.abs(z));
  const ans = 1.0 - t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  const cdf = z >= 0 ? 0.5 + 0.5 * ans : 0.5 - 0.5 * ans;
  return (1.0 - cdf) * 100.0;
}
