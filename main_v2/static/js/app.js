// app.js — Fall Detection Dashboard
const BACKEND = "http://localhost:5000";

// DOM refs
const fpsEl        = document.getElementById("fps");
const fallCountEl  = document.getElementById("fall-count");
const accuracyEl   = document.getElementById("accuracy");
const confBar      = document.getElementById("conf-bar");
const stabBar      = document.getElementById("stab-bar");
const fallStatus   = document.getElementById("fall-status");
const modelStatusEl= document.getElementById("model-status");

// ── Camera controls ────────────────────────────────────────
function startCamera() {
    fetch(`${BACKEND}/start`, { method: "POST" });
    document.getElementById("video-stream").src = `${BACKEND}/video_feed`;
    modelStatusEl.innerText = "ON";
    modelStatusEl.style.color = "";
}

function stopCamera() {
    fetch(`${BACKEND}/stop`, { method: "POST" });
    document.getElementById("video-stream").src = "";
    modelStatusEl.innerText = "OFF";
    modelStatusEl.style.color = "";
}

// ── Alert log ──────────────────────────────────────────────
const alertLog = [];
function pushLog(msg, isFall) {
    const ts = new Date().toLocaleTimeString([], { hour12: false });
    alertLog.unshift({ msg, isFall, ts });
    if (alertLog.length > 20) alertLog.pop();
    const ul = document.getElementById("alert-log");
    if (!ul) return;
    ul.innerHTML = alertLog.map(e =>
        `<li class="${e.isFall ? '' : 'safe'}">[${e.ts}] ${e.msg}</li>`
    ).join("");
}

// ── Status polling ─────────────────────────────────────────
let _wasFall         = false;
let connectionLost   = false;
let consecutiveErrors= 0;

// Track which alert event timestamps we've already logged
const _seenAlerts = new Set();

async function updateStatus() {
    try {
        const res = await fetch(`${BACKEND}/status`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // Metrics
        fpsEl.innerText       = data.fps;
        fallCountEl.innerText = data.fall_count;
        accuracyEl.innerText  = Math.round(data.confidence * 100) + "%";
        confBar.style.width   = Math.round(data.confidence * 100) + "%";
        stabBar.style.width   = Math.round(data.stability  * 100) + "%";

        // Source badge
        const badge = document.getElementById("source-badge");
        if (badge) badge.innerText = "source: " + (data.source || "—");

        // ── Process structured alert events from server ──────────
        // FIX: was never reading data.alerts; now we log each new event
        if (Array.isArray(data.alerts)) {
            for (const ev of data.alerts) {
                // Use person_id + rounded timestamp as a dedup key
                const key = `${ev.event_type}-${ev.person_id}-${Math.round(ev.timestamp)}`;
                if (!_seenAlerts.has(key)) {
                    _seenAlerts.add(key);
                    if (ev.event_type === "fall_alert") {
                        pushLog(
                            `⚠ FALL — Person #${ev.person_id} | conf: ${Math.round(ev.confidence * 100)}%`,
                            true
                        );
                    } else if (ev.event_type === "risk_warning") {
                        pushLog(
                            `⚡ RISK — Person #${ev.person_id} | conf: ${Math.round(ev.confidence * 100)}%`,
                            true
                        );
                    } else if (ev.event_type === "alert_resolved") {
                        pushLog(`✓ Resolved — Person #${ev.person_id} back to SAFE`, false);
                    }
                }
            }
        }

        // Fall banner + status
        const banner = document.getElementById("fall-alert-banner");
        if (data.fall) {
            if (banner) banner.classList.add("active");
            fallStatus.innerText = "⚠ FALL DETECTED";
            fallStatus.className = "fall";
            if (!_wasFall) {
                // Only beep on the rising edge (first frame of a new fall)
                try {
                    const ctx = new (window.AudioContext || window.webkitAudioContext)();
                    const osc = ctx.createOscillator();
                    osc.type = "square"; osc.frequency.value = 880;
                    osc.connect(ctx.destination);
                    osc.start(); osc.stop(ctx.currentTime + 0.3);
                } catch(_) {}
            }
            _wasFall = true;
        } else {
            if (banner) banner.classList.remove("active");
            fallStatus.innerText = "SAFE";
            fallStatus.className = "safe";
            if (_wasFall) pushLog("✓ Resolved — back to SAFE", false);
            _wasFall = false;
        }

        // Connection recovery
        if (connectionLost) {
            modelStatusEl.innerText    = "ON";
            modelStatusEl.style.color  = "#22d3ee";
            connectionLost             = false;
        }
        consecutiveErrors = 0;

    } catch (err) {
        consecutiveErrors++;
        if (consecutiveErrors >= 3 && !connectionLost) {
            connectionLost             = true;
            modelStatusEl.innerText    = "OFFLINE";
            modelStatusEl.style.color  = "#f43f5e";
            fpsEl.innerText            = "--";
            accuracyEl.innerText       = "--%";
            confBar.style.width        = "0%";
            stabBar.style.width        = "0%";
        }
    }
}

setInterval(updateStatus, 500);
updateStatus();

// ── Media list ─────────────────────────────────────────────
async function loadMediaList() {
    try {
        const res  = await fetch(`${BACKEND}/media_list`);
        const data = await res.json();
        const sel  = document.getElementById("source-select");
        if (!sel) return;
        [...sel.options].forEach(o => { if (o.value !== "webcam") o.remove(); });
        data.files.forEach(name => {
            const opt = document.createElement("option");
            opt.value = name;
            opt.text  = (/\.(jpg|jpeg|png|bmp)$/i.test(name) ? "🖼 " : "🎬 ") + name;
            sel.appendChild(opt);
        });
        toast(`${data.files.length} file(s) found`);
    } catch(e) { toast("Could not reach server", true); }
}

// ── Apply selected source ──────────────────────────────────
async function applySource() {
    const source = document.getElementById("source-select").value;
    try {
        const res  = await fetch(`${BACKEND}/set_source`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ source })
        });
        const data = await res.json();
        if (!res.ok) { toast(data.error || "Error", true); return; }
        document.getElementById("video-stream").src = `${BACKEND}/video_feed?t=${Date.now()}`;
        modelStatusEl.innerText = "ON";
        toast(`✓ Switched to: ${data.source}`);
    } catch(e) { toast("Switch failed — server running?", true); }
}

// ── File upload ────────────────────────────────────────────
function onFileSelected(input) {
    const file = input.files[0];
    if (!file) return;
    document.getElementById("upload-name").innerText = file.name;
    document.getElementById("btn-upload").disabled   = false;
}

async function uploadFile() {
    const input = document.getElementById("file-input");
    if (!input.files[0]) return;
    const btn = document.getElementById("btn-upload");
    btn.disabled = true; btn.innerText = "⏳ Uploading…";

    const fd = new FormData();
    fd.append("file", input.files[0]);
    try {
        // FIX: was calling /upload — corrected to /upload_media to match server route
        const res  = await fetch(`${BACKEND}/upload_media`, { method: "POST", body: fd });
        const data = await res.json();
        if (!res.ok) { toast(data.error || "Upload failed", true); return; }
        document.getElementById("video-stream").src = `${BACKEND}/video_feed?t=${Date.now()}`;
        modelStatusEl.innerText = "ON";
        toast(`✓ Uploaded & switched to: ${data.filename}`);
        input.value = "";
        document.getElementById("upload-name").innerText = "No file selected — pick a video or photo";
        await loadMediaList();
        document.getElementById("source-select").value = data.filename;
    } catch(e) { toast("Upload error", true); }
    finally { btn.disabled = false; btn.innerText = "⬆ Upload & Switch"; }
}

// ── Toast ──────────────────────────────────────────────────
let _toastTimer;
function toast(msg, isError = false) {
    const el = document.getElementById("toast");
    if (!el) return;
    el.innerText = msg;
    el.style.borderColor = isError ? "#f43f5e" : "rgba(168,85,247,.5)";
    el.classList.add("show");
    clearTimeout(_toastTimer);
    _toastTimer = setTimeout(() => el.classList.remove("show"), 3500);
}

// Auto-load file list on page open
loadMediaList();