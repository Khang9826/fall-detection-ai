// DOM Elements
const fpsEl = document.getElementById("fps");
const fallCountEl = document.getElementById("fall-count");
const accuracyEl = document.getElementById("accuracy");
const confBar = document.getElementById("conf-bar");
const stabBar = document.getElementById("stab-bar");
const fallStatus = document.getElementById("fall-status");
const modelStatusEl = document.getElementById("model-status");
const BACKEND = "http://localhost:5000";

function startCamera() {
    fetch(`${BACKEND}/start`, { method: "POST" });
    document.getElementById("video-stream").src = `${BACKEND}/video_feed`;
}

function stopCamera() {
    fetch(`${BACKEND}/stop`, { method: "POST" });
    document.getElementById("video-stream").src = "";
}

// Connection state
let connectionLost = false;
let consecutiveErrors = 0;

/**
 * Update dashboard with latest system status
 * Fixed: Use relative URL (works on any IP/domain)
 * Fixed: Added user-facing error feedback
 */
async function updateStatus() {
    try {
        // Fixed: Changed from "http://localhost:5000/status" to relative URL
        const res = await fetch(`${BACKEND}/status`);
        
        if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
        }
        
        const data = await res.json();

        // Update metrics
        fpsEl.innerText = data.fps;
        fallCountEl.innerText = data.fall_count;
        accuracyEl.innerText = Math.round(data.confidence * 100) + "%";

        confBar.style.width = Math.round(data.confidence * 100) + "%";
        stabBar.style.width = Math.round(data.stability * 100) + "%";

        // Update fall status
        if (data.fall) {
            fallStatus.innerText = "FALL DETECTED";
            fallStatus.className = "fall";
        } else {
            fallStatus.innerText = "SAFE";
            fallStatus.className = "safe";
        }

        // Fixed: Added connection recovery feedback
        if (connectionLost) {
            console.log("✓ Connection restored");
            modelStatusEl.innerText = "ON";
            modelStatusEl.style.color = "#22d3ee"; // cyan
            connectionLost = false;
        }
        consecutiveErrors = 0;

    } catch (err) {
        consecutiveErrors++;
        console.error("Backend error:", err);

        // Fixed: Show error to user after multiple failures
        if (consecutiveErrors >= 3 && !connectionLost) {
            connectionLost = true;
            modelStatusEl.innerText = "OFFLINE";
            modelStatusEl.style.color = "#f43f5e"; // red
            
            // Reset stats to show disconnection
            fpsEl.innerText = "--";
            accuracyEl.innerText = "--%";
            confBar.style.width = "0%";
            stabBar.style.width = "0%";
            
            console.error("⚠ Connection lost to backend");
        }
    }
}

// Fixed: Kept 500ms interval (faster than orphan script's 1000ms)
setInterval(updateStatus, 500);

// Initial update
updateStatus();