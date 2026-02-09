<div align="center">

<img width="1920" height="1008" alt="Screenshot 2026-01-07 011039" src="https://github.com/user-attachments/assets/91308fcd-4faf-4b58-bbfe-f3f7ec4258cd" />


# AEON: RNG Synchronicity Suite
### *Advanced Entropy Monitoring \u0026 Calibration for Parapsychological Research*

</div>

---

## 🚀 Recent Updates & Improvements

- **⚡ ESP32 Wireless Node Support**: Integrated hardware RNG support via UDP.
- **🧙 Startup Wizard**: New all-in-one startup screen for easier configuration, Wi-Fi setup, and firmware generation.
- **📊 Pop-out Waterfall**: Separate, scalable window for dedicated full-screen spectral analysis.
- **🎯 Auto Center**: Spectral visualization now automatically tracks and centers on the most prominent frequency peaks.
- **🔢 Adjustable Bin Size**: User-selectable FFT resolution (8–256 bins) for finer spectral detail.
- **🛠️ Firmware Generator**: Embedded GUI tool with auto-IP detection for seamless ESP32 node deployment.
- **🐛 Stability Fix**: Resolved a critical bug causing the visualization graphs to hang during long-term monitoring sessions. The suite now supports continuous multi-hour logging without interruption.

---

## 🌌 Overview
**AEON** is a specialized toolset designed for researchers exploring the boundaries of mind-matter interaction, telepathy, and anomalous cognition. By utilizing high-fidelity entropy monitoring and adaptive calibration, AEON allows users to detect and log "synchronicities"—non-random fluctuations in entropy streams that correlate with focused intent or external events.

---

## 📈 Real-Time Monitoring: Random Number Visualizer
### Features:
  - **Instantaneous Probability**: Monitor the real-time bias (p-value) of your entropy source.
  - **The Cumulative Walk**: Track long-term drift and deviation from the expected random mean.
  - **Frequency Morphing Waterfall**: Visualize the spectral evolution of the entropy stream using Welch's PSD method.
- **Intelligent Anomaly Engine**:
  - **Statistical Detection**: Real-time logging of Chi-Square significance and run-length streaks.
  - **FFT Analysis**: Detects persistent frequency peaks, spectral order, and unexpected power shifts.
---
<img width="827" height="1080" alt="Screenshot 2026-01-08 214321" src="https://github.com/user-attachments/assets/f2084c03-7aa6-415f-a8c8-128528bc61e9" />
The **Calibration Mode** (`CalibrationMode.py`) acts as a search engine for identifying high-bias channels across various parameter spaces.

- **Hidden Bit Sources**: Supports multiple "ground truth" generators for rigorous cross-validation:
  - **Internal Feedback**: Tests if the entropy stream predicts its own future state.
  - **PRNG Resonance**: Cross-references separate generators for synchronous behavior.
  - **Temporal Hashing**: Uses unpredictable but deterministic time-based hashes.
  - **Fractal Mapping**: Maps entropy to the complex-plane iterations of the Mandelbrot set.
- **Session Results**: Real-time Z-scoring, hit-rate analysis, and probability-of-chance calculations.


### 1. Requirements
- Python 3.8+
- `tkinter`, `matplotlib`, `numpy`, `scipy`

### 2. Launch
Initialize the monitoring suite:
```bash
python RandomNumberVisualizer.py
```

The **Startup Wizard** will appear on first run:
- Select your entropy source (OS Entropy or ESP32 Network Nodes)
- If using ESP32, configure Wi-Fi credentials and host IP (auto-detected)
- Generate firmware for multiple nodes if needed (all on one screen)
- Click **Start AEON** to begin monitoring

### 3. Core Features & Controls

**Main Controls:**
- **Start/Stop**: Toggle the real-time entropy stream.
- **Reset Walk**: Reinitialize the cumulative walk chart.
- **🔮 Calibration**: Open the multi-dimensional search engine to find high-resonance channels.
- **Pattern Len**: Adjust pattern sensitivity (4–32 bits) for anomaly detection.
- **Note-Taking**: Add timestamped markers to correlate with mental focus or external events.

**Visualization & Analysis:**
- **Triple-View Display**:
  - Instantaneous Probability: Real-time p-value bias
  - Cumulative Walk: Long-term drift and deviation tracking
  - Frequency Morphing Waterfall: Spectral evolution using Welch's PSD method
  
**Waterfall Controls** (in the Anomaly panel):
- **Bin Size**: Adjust frequency resolution (8–256 bins; default 64)
- **Auto Center**: When enabled, automatically center the waterfall on the strongest spectral peak
- **📊 Pop-out**: Open the waterfall in a separate full-screen window for extended analysis

**Source & Configuration** (Menu → Source):
- **Use OS Entropy (/dev/urandom)**: Switch to system entropy
- **Use ESP32 Network Nodes**: Switch to wireless entropy nodes
- **Generate ESP32 Firmware...**: Re-generate firmware for additional nodes (embedded in startup wizard for convenience)
- **📊 Open Pop-out Waterfall**: Launch extended waterfall visualization

**Node Status** (Anomaly Panel → "Node Status" tab):
- Real-time ESP32 node monitoring (IP, last seen, bit rate, total bits received)
- Visual indicator of which nodes are active and responsive

---

## 📜 Logging \u0026 Export

AEON automatically logs all sessions to the `rng_logs/` directory.
- **CSV Data**: Raw timestamps and bitstreams for external analysis.
- **JSON Metadata**: Comprehensive session summaries, including anomaly events and calibration results.

---
<div align="center">
Developed for the exploration of the intersection between intent and entropy.
</div>

## ⚡ ESP32 Node Support (Experimental)

AEON includes experimental support for reading entropy from small ESP32 "nodes" that send hardware RNG packets over UDP to the host running the visualizer.

### Getting Started with ESP32 Nodes

The **Startup Wizard** (shown when you launch AEON) includes an integrated ESP32 configuration panel:
1. Select **ESP32 Network Nodes** in the source selector
2. Enter your Wi-Fi SSID and password
3. The host IP is **auto-detected** (but you can override it)
4. Click **Generate Firmware** to create an Arduino sketch for your node
5. You can generate firmware for multiple nodes without closing the wizard

### Features
- Firmware generation directly in the startup wizard (no dialog popups needed)
- Auto-detected host IP address for convenience
- All configuration fields on a single panel
- Real-time node status monitoring in the UI ("Node Status" tab)
- Support for multiple simultaneous nodes with individual bit-rate tracking

### Important Notes / Known Issues (Connection Stability)
- This feature is **experimental**: connection stability depends heavily on the Wi‑Fi environment and router configuration (AP isolation, power-save modes, channel roaming, etc.).
- The provided firmware attempts to improve robustness by disabling Wi‑Fi power save and binding to the AP's BSSID/channel when found, but some networks still cause packet loss or disconnects.
- If you see frequent "NO ESP32 DATA" warnings in the UI, try the troubleshooting steps in `esp32_firmware/README.md` (firewall, reserved DHCP, disable client isolation, test with `test_udp_sender.py`).

If you rely on ESP32 nodes for long-running sessions, consider using a dedicated access point or wired gateway to reduce variability.

---

<div align="center">
Developed for the exploration of the intersection between intent and entropy.
</div>


