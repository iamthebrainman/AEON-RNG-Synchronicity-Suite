import tkinter as tk
import tkinter.ttk as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt  # For specgram
import os
import time
import json
import csv
import sqlite3
from collections import deque, Counter
import math
import numpy as np
from scipy.stats import chisquare
from scipy.signal import correlate, welch
from tkinter import simpledialog  # For session name prompt
from CalibrationMode import CalibrationWindow  # Calibration popup

# --- CONFIGURATION ---
WINDOW_SIZE = 1000   # How many bits to keep in the "probability" window
UPDATE_MS = 50       # Update speed (lower = smoother/faster)
BITS_PER_TICK = 10   # How many bits per update
LOG_DIR = "rng_logs"
PATTERN_FILE = "patterns_history.json"
PATTERN_LENGTH = 16  # Initial; now adjustable
AUTO_CORR_LAGS = 20  # For autocorrelation
STREAK_ALERT = 10    # Max streak to flag
CHI_SIG_THRESH = 0.01  # p < this = significant
AUTO_CORR_THRESH = 0.1  # |corr| > this = flag
WALK_MAX_VIS = 500   # For magnitude bar normalization

class RNGFluctuationMeter:
    def __init__(self, root):
        self.root = root
        self.root.title("RNG Fluctuation Meter - Triple View (FFT Enhanced)")
        self.root.geometry("1200x1000")  # Larger for FFT subplot

        # --- AUTO-START SESSION NAME PROMPT ---
        session_name = simpledialog.askstring("Session Name", "Enter session name (or leave blank for timestamp):")
        if not session_name:
            session_name = time.strftime("%Y%m%d_%H%M%S")
        else:
            session_name = "".join(c if c.isalnum() else "_" for c in session_name)  # Sanitize
        self.session_name = session_name
        # Total bits emitted since start (monotonic counter)
        self.total_bits = 0

        # --- LAYOUT: CONTROLS FIRST (BOTTOM) ---
        self.controls_frame = tk.Frame(root, bg="#1e1e1e", bd=2, relief=tk.RAISED)
        self.controls_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # BUTTONS SUB-FRAME
        btn_frame = tk.Frame(self.controls_frame, bg="#1e1e1e")
        btn_frame.pack(side=tk.LEFT, padx=5)

        self.btn = tk.Button(btn_frame, text="START", command=self.toggle, 
                             bg="#008000", fg="white", font=("Arial", 12, "bold"), width=8)
        self.btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.reset_btn = tk.Button(btn_frame, text="Reset Walk", command=self.reset_walk, 
                                   bg="#333333", fg="white", font=("Arial", 10))
        self.reset_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.calib_btn = tk.Button(btn_frame, text="🔮 Calibration", command=self.open_calibration,
                                   bg="#444499", fg="white", font=("Arial", 10, "bold"))
        self.calib_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # PATTERN SENSITIVITY SLIDER
        sens_frame = tk.Frame(self.controls_frame, bg="#1e1e1e")
        sens_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(sens_frame, text="Pattern Len:", bg="#1e1e1e", fg="#aaaaaa").pack(side=tk.LEFT)
        self.pattern_scale = tk.Scale(sens_frame, from_=4, to=32, orient=tk.HORIZONTAL, 
                                      command=self.update_pattern_length, bg="#2d2d2d", fg="white")
        self.pattern_scale.set(PATTERN_LENGTH)
        self.pattern_scale.pack(side=tk.LEFT, padx=5)

        # NOTE SUB-FRAME
        note_frame = tk.Frame(self.controls_frame, bg="#1e1e1e")
        note_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        tk.Label(note_frame, text="Note:", bg="#1e1e1e", fg="#aaaaaa").pack(side=tk.LEFT)
        self.note_entry = tk.Entry(note_frame, bg="#2d2d2d", fg="white", insertbackground="white")
        self.note_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.note_entry.bind("<Return>", lambda e: self.add_note())

        self.note_btn = tk.Button(note_frame, text="Add Note", command=self.add_note,
                                  bg="#444499", fg="white", font=("Arial", 10))
        self.note_btn.pack(side=tk.LEFT, padx=5)

        # COLORMAP CONTROLS
        cmap_frame = tk.Frame(self.controls_frame, bg="#1e1e1e")
        cmap_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(cmap_frame, text="Colormap:", bg="#1e1e1e", fg="#aaaaaa").pack(side=tk.LEFT)
        self.cmap_var = tk.StringVar(value="viridis")
        self.cmap_dropdown = tk.OptionMenu(cmap_frame, self.cmap_var, 
                                           "viridis", "plasma", "inferno", "magma", "cividis",
                                           "twilight", "twilight_shifted", "cool", "winter", "ocean",
                                           command=self.update_colormap)
        self.cmap_dropdown.config(bg="#2d2d2d", fg="white", highlightthickness=0)
        self.cmap_dropdown.pack(side=tk.LEFT, padx=5)
        
        self.reverse_var = tk.BooleanVar(value=False)
        self.reverse_check = tk.Checkbutton(cmap_frame, text="Reverse", variable=self.reverse_var,
                                           command=self.update_colormap, bg="#1e1e1e", fg="#aaaaaa",
                                           selectcolor="#2d2d2d", activebackground="#1e1e1e")
        self.reverse_check.pack(side=tk.LEFT, padx=5)

        # STATS / DIAGNOSTICS LABEL
        self.label = tk.Label(self.controls_frame, text="Ready", font=("Courier", 11, "bold"), 
                              bg="#1e1e1e", fg="#00ff00", justify=tk.LEFT)
        self.label.pack(side=tk.RIGHT, padx=10)

        # --- MAGNITUDE BAR (DEVIATION GAUGE) ---
        self.mag_frame = tk.Frame(root, bg="#1e1e1e")
        self.mag_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        tk.Label(self.mag_frame, text="Deviation\nMagnitude", bg="#1e1e1e", fg="#aaaaaa", font=("Arial", 8)).pack(side=tk.TOP)
        self.mag_canvas = tk.Canvas(self.mag_frame, width=30, height=400, bg="#2d2d2d")
        self.mag_canvas.pack(side=tk.TOP, pady=5)
        self.mag_bar = self.mag_canvas.create_rectangle(0, 0, 30, 0, fill="#00ff00")  # Starts empty

        # --- ANOMALY DETECTION PANEL (RIGHT SIDE) ---
        self.right_panel = tk.Frame(root, bg="#1e1e1e", width=300)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        tk.Label(self.right_panel, text="ANOMALY DETECTION LOG", 
                 bg="#1e1e1e", fg="#ffaa00", font=("Arial", 10, "bold")).pack(pady=5)
        
        # Treeview styling
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#2d2d2d", foreground="white", 
                       fieldbackground="#2d2d2d", borderwidth=0)
        style.configure("Treeview.Heading", background="#444444", foreground="white", 
                       borderwidth=1, relief="raised")
        style.map("Treeview", background=[("selected", "#555555")])
        
        # Notebook styling for tabs
        style.configure("TNotebook", background="#1e1e1e", borderwidth=0)
        style.configure("TNotebook.Tab", background="#2d2d2d", foreground="white", 
                       padding=[10, 5], borderwidth=1)
        style.map("TNotebook.Tab", background=[("selected", "#444444")], 
                 foreground=[("selected", "#00ff00")])
        
        # Create tabbed notebook
        self.anomaly_notebook = ttk.Notebook(self.right_panel)
        self.anomaly_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # --- TAB 1: STATISTICAL ANOMALIES ---
        stat_frame = tk.Frame(self.anomaly_notebook, bg="#1e1e1e")
        self.anomaly_notebook.add(stat_frame, text="Statistical")
        
        stat_tree_frame = tk.Frame(stat_frame, bg="#1e1e1e")
        stat_tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.stat_tree = ttk.Treeview(stat_tree_frame, columns=('Time', 'Type', 'Details'), 
                                      height=15, show='headings')
        self.stat_tree.heading('Time', text='Time')
        self.stat_tree.heading('Type', text='Type')
        self.stat_tree.heading('Details', text='Details')
        self.stat_tree.column('Time', width=70)
        self.stat_tree.column('Type', width=90)
        self.stat_tree.column('Details', width=120)
        self.stat_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        stat_scroll = ttk.Scrollbar(stat_tree_frame, orient=tk.VERTICAL, 
                                   command=self.stat_tree.yview)
        stat_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.stat_tree.configure(yscrollcommand=stat_scroll.set)
        
        # --- TAB 2: FFT PATTERNS ---
        fft_frame = tk.Frame(self.anomaly_notebook, bg="#1e1e1e")
        self.anomaly_notebook.add(fft_frame, text="FFT Patterns")
        
        fft_tree_frame = tk.Frame(fft_frame, bg="#1e1e1e")
        fft_tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fft_tree = ttk.Treeview(fft_tree_frame, columns=('Time', 'Type', 'Details'), 
                                     height=15, show='headings')
        self.fft_tree.heading('Time', text='Time')
        self.fft_tree.heading('Type', text='Type')
        self.fft_tree.heading('Details', text='Details')
        self.fft_tree.column('Time', width=70)
        self.fft_tree.column('Type', width=90)
        self.fft_tree.column('Details', width=120)
        self.fft_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        fft_scroll = ttk.Scrollbar(fft_tree_frame, orient=tk.VERTICAL, 
                                  command=self.fft_tree.yview)
        fft_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.fft_tree.configure(yscrollcommand=fft_scroll.set)
        
        # --- TAB 3: PATTERN REPEATS ---
        pattern_frame = tk.Frame(self.anomaly_notebook, bg="#1e1e1e")
        self.anomaly_notebook.add(pattern_frame, text="Patterns")
        
        pattern_tree_frame = tk.Frame(pattern_frame, bg="#1e1e1e")
        pattern_tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.pattern_tree = ttk.Treeview(pattern_tree_frame, columns=('Time', 'Type', 'Details'), 
                                        height=15, show='headings')
        self.pattern_tree.heading('Time', text='Time')
        self.pattern_tree.heading('Type', text='Type')
        self.pattern_tree.heading('Details', text='Details')
        self.pattern_tree.column('Time', width=70)
        self.pattern_tree.column('Type', width=90)
        self.pattern_tree.column('Details', width=120)
        self.pattern_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        pattern_scroll = ttk.Scrollbar(pattern_tree_frame, orient=tk.VERTICAL, 
                                      command=self.pattern_tree.yview)
        pattern_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.pattern_tree.configure(yscrollcommand=pattern_scroll.set)
        
        # Keep reference to old anomaly_tree for backward compatibility
        self.anomaly_tree = self.stat_tree  # Default to statistical tab
        
        # Frequency drift indicator
        self.drift_label = tk.Label(self.right_panel, text="Freq Drift: 0.0000 Hz/s", 
                                   bg="#1e1e1e", fg="#00ff00", font=("Courier", 9, "bold"))
        self.drift_label.pack(pady=5)

        # --- DATA SETUP ---
        # Anomaly tracking
        self.anomaly_db = []  # Persistent storage
        self.fft_history_db = []  # Store FFT spectra long-term
        self.pattern_anomaly_logged = {}  # Track which patterns have been logged and at what count
        self.window = deque(maxlen=WINDOW_SIZE)
        self.bit_stream = deque(maxlen=self.pattern_scale.get())  # Dynamic
        self.signal_buffer = deque(maxlen=512)  # For FFT analysis (±1 values)
        self.cumulative_val = 0
        self.timestamps = deque(maxlen=300)
        self.p_history = deque(maxlen=300)
        self.cumulative_history = deque(maxlen=300)
        
        # FFT baseline for anomaly detection
        self.fft_baseline_mean = None   # np.array of shape (64,)
        self.fft_baseline_var = None    # np.array of shape (64,)
        self.fft_baseline_alpha = 0.01  # EMA smoothing factor
        self.fft_frame_counter = 0      # Throttle FFT anomaly checks
        self.last_fft_anomaly_time = 0  # Debounce FFT-PATTERN logging

        # Pattern Memory
        self.patterns_history = self.load_patterns()
        self.recent_patterns = deque(maxlen=10)
        self.current_log_file = None
        self.current_csv_writer = None
        self.log_path = None

        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

        # --- FIGURE SETUP (THREE SUBPLOTS) ---
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor='#121212')
        self.fig.subplots_adjust(bottom=0.1, top=0.95, hspace=0.3, left=0.1, right=0.95)

        # Top: Probability
        self.ax1 = self.fig.add_subplot(311, facecolor='#1e1e1e')
        self.line_p, = self.ax1.plot([], [], color='#4da6ff', linewidth=1.5)
        self.ax1.set_title("Instantaneous Probability (Last 1000 bits)", fontsize=10, color='white')
        self.ax1.tick_params(colors='white')
        self.ax1.set_ylim(0.4, 0.6)
        self.ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        self.ax1.grid(True, alpha=0.1, color='white')

        # Middle: Cumulative Walk
        self.ax2 = self.fig.add_subplot(312, facecolor='#1e1e1e')
        self.line_cum, = self.ax2.plot([], [], color='#ff4d4d', linewidth=2)
        self.ax2.set_title("Cumulative Deviation (The Walk)", fontsize=10, color='white')
        self.ax2.tick_params(colors='white')
        self.ax2.set_xlabel("Seconds Ago", color='white')
        self.ax2.axhline(y=0, color='white', linewidth=1, alpha=0.5)
        self.ax2.grid(True, alpha=0.1, color='white')

        # Bottom: Frequency Morphing Waterfall (Long-term spectral evolution)
        self.ax3 = self.fig.add_subplot(313, facecolor='#1e1e1e')
        self.ax3.set_title("Frequency Morphing (Long-term Spectral Evolution)", fontsize=10, color='white')
        self.ax3.tick_params(colors='white')
        self.ax3.set_xlabel("Frequency Bin", color='white')
        self.ax3.set_ylabel("Time (Recent)", color='white')
        self.ax3.grid(False)
        # Waterfall: 100 time slices × 64 frequency bins
        self.waterfall_data = np.zeros((100, 64))
        self.waterfall_img = self.ax3.imshow(self.waterfall_data, aspect='auto', cmap='plasma',
                                            extent=[0, 64, 0, 100], origin='lower', interpolation='bilinear')
        self.fig.colorbar(self.waterfall_img, ax=self.ax3, label='Power', shrink=0.5)

        for spine in self.ax1.spines.values(): spine.set_color('#444444')
        for spine in self.ax2.spines.values(): spine.set_color('#444444')
        for spine in self.ax3.spines.values(): spine.set_color('#444444')

        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.running = False

    def load_patterns(self):
        if os.path.exists(PATTERN_FILE):
            try:
                with open(PATTERN_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_patterns(self):
        with open(PATTERN_FILE, 'w') as f:
            json.dump(self.patterns_history, f)

    def add_note(self):
        note_text = self.note_entry.get().strip()
        if not note_text:
            return
        
        timestamp = time.time()
        if self.current_csv_writer:
            self.current_csv_writer.writerow([timestamp, "NOTE", note_text])
            self.current_log_file.flush()
        
        self.recent_patterns.appendleft(f"NOTE: {note_text}")
        self.note_entry.delete(0, tk.END)
        print(f"Added Note: {note_text}")

    def get_random_bit(self):
        return int.from_bytes(os.urandom(1), "big") & 1

    def reset_walk(self):
        self.cumulative_val = 0
        self.cumulative_history.clear()
        self.timestamps.clear()
        self.p_history.clear()
        self.window.clear()
        self.bit_stream.clear()
        self.signal_buffer.clear()
        self.waterfall_data = np.zeros_like(self.waterfall_data)
        self.waterfall_img.set_data(self.waterfall_data)
        self.canvas.draw()
    
    def open_calibration(self):
        """Open calibration mode popup window"""
        CalibrationWindow(self.root, self)

    def update_pattern_length(self, val):
        length = int(val)
        self.bit_stream = deque(self.bit_stream, maxlen=length)  # Resize

    def update_colormap(self, *args):
        """Update the colormap based on dropdown and reverse checkbox"""
        cmap_name = self.cmap_var.get()
        if self.reverse_var.get():
            cmap_name = cmap_name + "_r"
        self.waterfall_img.set_cmap(cmap_name)
        self.canvas.draw()

    def update(self):
        if not self.running:
            return

        # 1. Generate Data
        now = time.time()
        new_bits = []
        for _ in range(BITS_PER_TICK):
            b = self.get_random_bit()
            self.window.append(b)
            self.bit_stream.append(b)
            signal_val = 2*b - 1  # ±1 for signal processing
            new_bits.append(signal_val)
            self.signal_buffer.append(signal_val)  # Accumulate for FFT
            
            # Log bits
            if self.current_csv_writer:
                self.current_csv_writer.writerow([now, "BIT", b])

            # Walk Logic
            if b == 1:
                self.cumulative_val += 1
            else:
                self.cumulative_val -= 1
        # Track total bits emitted
        self.total_bits += len(new_bits)

        # 2. Pattern Matching (Hex Signature)
        if len(self.bit_stream) == self.pattern_scale.get():
            bit_str = "".join(map(str, self.bit_stream))
            signature = hex(int(bit_str, 2))
            
            if signature not in self.patterns_history:
                self.patterns_history[signature] = {"count": 0, "first_seen": now}
            
            self.patterns_history[signature]["count"] += 1
            self.patterns_history[signature]["last_seen"] = now

            if self.patterns_history[signature]["count"] > 10:
                msg = f"Pattern Matched: {signature} (x{self.patterns_history[signature]['count']})"
                if not self.recent_patterns or self.recent_patterns[0] != msg:
                    self.recent_patterns.appendleft(msg)

        # 3. Other Patterns: Autocorrelation & Streaks
        if len(self.window) == WINDOW_SIZE:
            p = sum(self.window) / WINDOW_SIZE
            
            # Autocorrelation (normalized)
            signal = np.array(self.window) * 2 - 1  # ±1
            autocorr = correlate(signal, signal, mode='full')[WINDOW_SIZE - AUTO_CORR_LAGS : WINDOW_SIZE + AUTO_CORR_LAGS]
            autocorr = autocorr.astype(np.float64)  # Convert to float before division
            autocorr /= autocorr[AUTO_CORR_LAGS]  # Normalize by lag-0
            
            # Only search POSITIVE lags (indices AUTO_CORR_LAGS+1 to end)
            positive_lags = autocorr[AUTO_CORR_LAGS + 1:]  # Skip lag-0, only positive
            max_corr = np.max(np.abs(positive_lags))
            peak_lag = np.argmax(np.abs(positive_lags)) + 1  # +1 because we want lag 1-20, not 0-19
            auto_str = f"AutoCorr: lag{peak_lag} ({max_corr:.2f})" if max_corr > AUTO_CORR_THRESH else ""

            # Run-Length Streaks
            streaks = Counter()
            current = self.window[0]
            count = 1
            for b in list(self.window)[1:]:
                if b == current:
                    count += 1
                else:
                    streaks[f"{current}s"] = max(streaks.get(f"{current}s", 0), count)
                    current = b
                    count = 1
            streaks[f"{current}s"] = max(streaks.get(f"{current}s", 0), count)
            max_streak = max(streaks.values(), default=0)
            streak_str = f"Streak: {max_streak} ({max(streaks, key=streaks.get)})" if max_streak > STREAK_ALERT else ""

            # Chi-Square p-value
            observed = np.bincount(self.window)
            if len(observed) < 2:
                observed = np.append(observed, [0])
            expected = np.array([WINDOW_SIZE/2, WINDOW_SIZE/2])
            chi_stat, chi_p = chisquare(observed, expected)
            chi_str = f"Chi p: {chi_p:.4f}" + (" << SIG" if chi_p < CHI_SIG_THRESH else "")

            self.timestamps.append(now)
            self.p_history.append(p)
            self.cumulative_history.append(self.cumulative_val)

            # 4. Update Lines
            time_deltas = [t - now for t in self.timestamps]
            self.line_p.set_data(time_deltas, list(self.p_history))
            self.line_cum.set_data(time_deltas, list(self.cumulative_history))

            # 5. Frequency Morphing Waterfall \u0026 Anomaly Detection
            # Compute PSD using Welch's method for long-term frequency tracking
            if len(self.signal_buffer) >= 512:
                signal = np.array(list(self.signal_buffer)[-512:])
                
                # Welch's method for power spectral density (more stable than raw FFT)
                freqs, psd = welch(signal, nperseg=128, noverlap=64, nfft=256)
                
                # NEW: FFT-domain anomaly detection hook
                self.detect_fft_anomalies(freqs[:64], psd[:64], now)
                
                # Store in long-term DB
                self.fft_history_db.append({
                    'timestamp': now,
                    'freqs': freqs[:64],  # Keep first 64 bins
                    'psd': psd[:64]
                })
                
                # Update waterfall display
                psd_norm = (psd[:64] - psd[:64].min()) / (psd[:64].max() - psd[:64].min() + 1e-10)
                self.waterfall_data = np.roll(self.waterfall_data, -1, axis=0)
                self.waterfall_data[-1, :] = psd_norm
                self.waterfall_img.set_data(self.waterfall_data)
                
                # Calculate frequency drift (dominant frequency movement)
                if len(self.fft_history_db) >= 2:
                    # Find dominant frequency in current and previous
                    curr_dom = freqs[np.argmax(psd[:64])]
                    prev_psd = self.fft_history_db[-2]['psd']
                    prev_freqs = self.fft_history_db[-2]['freqs']
                    prev_dom = prev_freqs[np.argmax(prev_psd)]
                    
                    # Drift rate in Hz per second
                    time_diff = now - self.fft_history_db[-2]['timestamp']
                    if time_diff > 0:
                        drift_rate = (curr_dom - prev_dom) / time_diff
                        self.drift_label.config(text=f"Freq Drift: {drift_rate:.4f} Hz/s")
                        
                        # Color code the drift label
                        if abs(drift_rate) > 0.1:
                            self.drift_label.config(fg="#ff0000")  # Red = high drift
                        elif abs(drift_rate) > 0.01:
                            self.drift_label.config(fg="#ffaa00")  # Orange = moderate
                        else:
                            self.drift_label.config(fg="#00ff00")  # Green = stable
            
            # ANOMALY DETECTION & LOGGING (with debouncing to prevent spam)
            anomalies = []
            
            # Initialize last_logged tracker if not exists
            if not hasattr(self, 'last_anomaly_logged'):
                self.last_anomaly_logged = {}
            
            # Chi-square anomaly (debounce: log only if not logged in last 5 seconds)
            if chi_p < CHI_SIG_THRESH:
                last_chi_time = self.last_anomaly_logged.get('CHI-SIG', 0)
                if now - last_chi_time > 5.0:  # 5 second debounce
                    anomalies.append(("CHI-SIG", f"p={chi_p:.4f}"))
                    self.last_anomaly_logged['CHI-SIG'] = now
            
            # Autocorrelation anomaly (debounce: log only if lag or magnitude changed significantly)
            if max_corr > AUTO_CORR_THRESH:
                last_autocorr_key = f"lag{peak_lag}"
                last_autocorr_time = self.last_anomaly_logged.get('AUTO-CORR', 0)
                last_autocorr_lag = self.last_anomaly_logged.get('AUTO-CORR-LAG', -1)
                
                # Log if: different lag OR 5 seconds passed OR correlation increased by >0.05
                if (peak_lag != last_autocorr_lag) or (now - last_autocorr_time > 5.0):
                    anomalies.append(("AUTO-CORR", f"lag{peak_lag}={max_corr:.3f}"))
                    self.last_anomaly_logged['AUTO-CORR'] = now
                    self.last_anomaly_logged['AUTO-CORR-LAG'] = peak_lag
            
            # Streak anomaly (debounce: log only if streak length increased)
            if max_streak > STREAK_ALERT:
                last_streak = self.last_anomaly_logged.get('STREAK-LEN', 0)
                if max_streak > last_streak:  # Only log if streak got longer
                    anomalies.append(("STREAK", f"{max_streak} {max(streaks, key=streaks.get)}"))
                    self.last_anomaly_logged['STREAK-LEN'] = max_streak
                    self.last_anomaly_logged['STREAK-TIME'] = now
                elif now - self.last_anomaly_logged.get('STREAK-TIME', 0) > 10.0:
                    # Reset if 10 seconds passed without streak
                    self.last_anomaly_logged['STREAK-LEN'] = 0
            
            # Pattern repeat anomaly - only log NEW patterns or significant increases
            for sig, meta in self.patterns_history.items():
                current_count = meta['count']
                
                # Check if this pattern is significant (> 5 occurrences)
                if current_count > 5 and sig != "0xelf":
                    # Check if we've logged this pattern before
                    if sig not in self.pattern_anomaly_logged:
                        # First time this pattern crosses threshold - LOG IT
                        anomalies.append(("PATTERN", f"{sig}x{current_count} [NEW]"))
                        self.pattern_anomaly_logged[sig] = current_count
                        break  # Only log one pattern per tick
                    else:
                        # Pattern was logged before - only log if count increased significantly
                        last_logged_count = self.pattern_anomaly_logged[sig]
                        count_increase = current_count - last_logged_count
                        
                        if count_increase >= 10:  # Threshold: log every 10 new occurrences
                            anomalies.append(("PATTERN", f"{sig}x{current_count} [+{count_increase}]"))
                            self.pattern_anomaly_logged[sig] = current_count
                            break  # Only log one pattern per tick
            
            # Log anomalies to treeview and DB
            for anomaly_type, details in anomalies:
                time_str = time.strftime("%H:%M:%S", time.localtime(now))
                
                # Route to correct tab based on anomaly type
                if anomaly_type == "PATTERN":
                    target_tree = self.pattern_tree
                else:  # CHI-SIG, AUTO-CORR, STREAK
                    target_tree = self.stat_tree
                
                target_tree.insert('', 0, values=(time_str, anomaly_type, details))
                
                # Store in persistent DB
                self.anomaly_db.append({
                    'timestamp': now,
                    'type': anomaly_type,
                    'details': details,
                    'p': p,
                    'entropy': -(p * math.log2(p) + (1-p) * math.log2(1-p)) if 0 < p < 1 else 0,
                    'walk': self.cumulative_val
                })
                
                # Keep only last 100 anomalies in each treeview
                if len(target_tree.get_children()) > 100:
                    target_tree.delete(target_tree.get_children()[-1])

            # 6. Handle Axis Scrolling
            self.ax1.set_xlim(min(time_deltas) if time_deltas else -10, 0)
            self.ax2.set_xlim(min(time_deltas) if time_deltas else -10, 0)
            self.ax3.set_ylim(0, 100)  # Fixed time depth for waterfall
            
            self.ax2.relim()
            self.ax2.autoscale_view()
            self.canvas.draw()
            
            # Diagnostics
            p1 = p
            p0 = 1 - p
            entropy = 0
            if p0 > 0 and p1 > 0:
                entropy = -(p0 * math.log2(p0) + p1 * math.log2(p1))
            
            status = "RANDOM"
            if entropy < 0.95: status = "ORDERED?"
            if entropy < 0.80: status = "SYNCHRONICITY!"

            diag_text = f"p(1): {p:.4f} | Entropy: {entropy:.4f} | {status}\n"
            diag_text += f"Walk: {self.cumulative_val} | Log: {os.path.basename(self.log_path or '')}\n"
            diag_text += f"{chi_str} | {auto_str} | {streak_str}\n"
            if self.recent_patterns:
                diag_text += f"Latest: {self.recent_patterns[0]}"
            self.label.config(text=diag_text)

            # Update Magnitude Bar
            norm_dev = abs(self.cumulative_val) / WALK_MAX_VIS
            height = min(1, norm_dev) * 400
            color = "#00ff00" if self.cumulative_val > 0 else "#ff0000"
            self.mag_canvas.itemconfig(self.mag_bar, fill=color)
            self.mag_canvas.coords(self.mag_bar, 0, 400 - height, 30, 400)  # From bottom up

        self.root.after(UPDATE_MS, self.update)

    def detect_fft_anomalies(self, freqs, psd, now):
        """
        Update a rolling baseline over FFT power and push FFT-PATTERN anomalies
        into the same anomaly log (Treeview + anomaly_db) when certain bins or
        bands deviate significantly from that baseline.

        Features:
        - Throttled execution (every 3rd frame) to minimize FPS impact
        - Band grouping for contiguous anomalous bins
        - Spectral entropy and band power ratio computation
        - Persistent peak tracking and drift detection
        - Debounced logging to prevent spam

        freqs: 1D np.array of length 64
        psd:   1D np.array of length 64
        """
        # Throttle: only run every 3rd FFT frame to avoid lag
        self.fft_frame_counter += 1
        if self.fft_frame_counter % 3 != 0:
            # Still update baseline even when not checking anomalies
            if self.fft_baseline_mean is not None:
                alpha = self.fft_baseline_alpha
                delta = psd - self.fft_baseline_mean
                self.fft_baseline_mean += alpha * delta
                self.fft_baseline_var = (1 - alpha) * (self.fft_baseline_var + alpha * delta**2)
            return

        # Initialize baseline on first call
        if self.fft_baseline_mean is None:
            self.fft_baseline_mean = psd.copy()
            self.fft_baseline_var = np.ones_like(psd) * 1e-6
            return

        alpha = self.fft_baseline_alpha

        # Exponential moving average for mean and variance
        delta = psd - self.fft_baseline_mean
        self.fft_baseline_mean += alpha * delta
        self.fft_baseline_var = (1 - alpha) * (self.fft_baseline_var + alpha * delta**2)

        std = np.sqrt(self.fft_baseline_var + 1e-12)
        z_scores = (psd - self.fft_baseline_mean) / std

        # --- ADDITIONAL FFT-DOMAIN FEATURES ---
        
        # 1. Spectral Entropy (measure of frequency distribution randomness)
        psd_norm = psd / (np.sum(psd) + 1e-12)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        
        # 2. Band Power Ratios (low/mid/high frequency energy distribution)
        low_band = np.sum(psd[:16])    # Bins 0-15
        mid_band = np.sum(psd[16:48])  # Bins 16-47
        high_band = np.sum(psd[48:])   # Bins 48-63
        total_power = low_band + mid_band + high_band + 1e-12
        band_ratio = (low_band / total_power, mid_band / total_power, high_band / total_power)
        
        # 3. Peak Bin Indices (top 3 strongest frequencies)
        peak_bins = np.argsort(psd)[-3:][::-1]  # Descending order
        peak_freqs = freqs[peak_bins]
        
        # --- ANOMALY DETECTION ---
        
        # Debounce: don't log if we logged an FFT anomaly in the last 3 seconds
        if now - self.last_fft_anomaly_time < 3.0:
            return
        
        anomaly_detected = False
        anomaly_type = ""
        details = ""
        
        # Rule 1: Group contiguous anomalous bins (|z| > 4) into bands
        anomalous_mask = np.abs(z_scores) > 4.0
        if np.any(anomalous_mask):
            # Find contiguous regions
            anomalous_indices = np.where(anomalous_mask)[0]
            
            # Group into bands (contiguous bins within 2 indices of each other)
            bands = []
            current_band = [anomalous_indices[0]]
            
            for idx in anomalous_indices[1:]:
                if idx - current_band[-1] <= 2:  # Within 2 bins = same band
                    current_band.append(idx)
                else:
                    bands.append(current_band)
                    current_band = [idx]
            bands.append(current_band)
            
            # Find the strongest band
            strongest_band = max(bands, key=lambda b: np.sum(np.abs(z_scores[b])))
            band_start = freqs[strongest_band[0]]
            band_end = freqs[strongest_band[-1]]
            band_peak_z = np.max(np.abs(z_scores[strongest_band]))
            
            anomaly_detected = True
            anomaly_type = "FFT-BAND"
            details = f"{band_start:.1f}-{band_end:.1f}Hz z={band_peak_z:.1f}"
        
        # Rule 2: Persistent peak (same bin in top-3 for multiple checks)
        # Track last peak bins
        if not hasattr(self, 'last_peak_bins'):
            self.last_peak_bins = peak_bins
        else:
            # Check if any peak persisted
            persistent = np.intersect1d(self.last_peak_bins, peak_bins)
            if len(persistent) >= 2 and not anomaly_detected:
                # At least 2 peaks persisted
                persistent_freqs = freqs[persistent]
                anomaly_detected = True
                anomaly_type = "FFT-PERSIST"
                details = f"peaks@{persistent_freqs[0]:.1f},{persistent_freqs[1]:.1f}Hz"
            self.last_peak_bins = peak_bins
        
        # Rule 3: Spectral entropy anomaly (very low = ordered, very high = noisy)
        if spectral_entropy < 4.5 and not anomaly_detected:
            anomaly_detected = True
            anomaly_type = "FFT-ORDER"
            details = f"entropy={spectral_entropy:.2f}"
        elif spectral_entropy > 5.8 and not anomaly_detected:
            anomaly_detected = True
            anomaly_type = "FFT-CHAOS"
            details = f"entropy={spectral_entropy:.2f}"
        
        # Rule 4: Extreme band power imbalance
        if not anomaly_detected:
            if band_ratio[0] > 0.7:  # >70% power in low band
                anomaly_detected = True
                anomaly_type = "FFT-LOWSHIFT"
                details = f"low={band_ratio[0]:.2f}"
            elif band_ratio[2] > 0.5:  # >50% power in high band
                anomaly_detected = True
                anomaly_type = "FFT-HIGHSHIFT"
                details = f"high={band_ratio[2]:.2f}"
        
        
        # Log the anomaly if detected
        if anomaly_detected:
            time_str = time.strftime("%H:%M:%S", time.localtime(now))
            
            # Insert into FFT tab Treeview and anomaly_db
            self.fft_tree.insert('', 0, values=(time_str, anomaly_type, details))
            self.anomaly_db.append({
                'timestamp': now,
                'type': anomaly_type,
                'details': details,
                'p': None,
                'entropy': spectral_entropy,
                'walk': self.cumulative_val
            })
            
            # Keep only last 100 anomalies in FFT treeview
            if len(self.fft_tree.get_children()) > 100:
                self.fft_tree.delete(self.fft_tree.get_children()[-1])
            
            # Update debounce timer
            self.last_fft_anomaly_time = now

    def toggle(self):
        self.running = not self.running
        if self.running:
            self.btn.config(text="STOP", bg="#cc0000")
            
            # Start Logging
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            self.log_path = os.path.join(LOG_DIR, f"rng_log_{self.session_name}_{timestamp_str}.csv")
            self.current_log_file = open(self.log_path, 'w', newline='')
            self.current_csv_writer = csv.writer(self.current_log_file)
            self.current_csv_writer.writerow(["timestamp", "type", "value"])
            
            if len(self.window) < WINDOW_SIZE:
                for _ in range(WINDOW_SIZE):
                    self.window.append(self.get_random_bit())
            self.update()
        else:
            self.btn.config(text="START", bg="#008000")
            # Stop Logging & Export JSON
            if self.current_log_file:
                self.current_log_file.close()
                self.current_log_file = None
                self.current_csv_writer = None
            
            # Export Walk Data
            export_path = self.log_path.replace('.csv', '.json')
            export_data = {
                'timestamps': list(self.timestamps),
                'cumulative_history': list(self.cumulative_history),
                'p_history': list(self.p_history),
                'patterns_history': self.patterns_history
            }
            with open(export_path, 'w') as f:
                json.dump(export_data, f)
            print(f"Exported walk data to {export_path}")
            
            # Save long-term anomaly and FFT data
            self.save_longterm_data()
            
            self.save_patterns()

    def save_longterm_data(self):
        """Save anomalies and FFT history to SQLite"""
        db_path = os.path.join(LOG_DIR, "rng_longterm.db")
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Create tables
        c.execute('''CREATE TABLE IF NOT EXISTS anomalies
                     (timestamp REAL, type TEXT, details TEXT, p REAL, entropy REAL, walk INTEGER)''')
        c.execute('''CREATE TABLE IF NOT EXISTS fft_history
                     (timestamp REAL, freq_bin INTEGER, power REAL)''')
        
        # Store anomalies
        for anomaly in self.anomaly_db:
            c.execute("INSERT INTO anomalies VALUES (?, ?, ?, ?, ?, ?)",
                     (anomaly['timestamp'], anomaly['type'], anomaly['details'],
                      anomaly['p'], anomaly['entropy'], anomaly['walk']))
        
        # Store FFT spectra (downsampled to save space)
        for record in self.fft_history_db:
            for i, (freq, power) in enumerate(zip(record['freqs'], record['psd'])):
                if i % 4 == 0:  # Store every 4th bin to save space
                    c.execute("INSERT INTO fft_history VALUES (?, ?, ?)",
                             (record['timestamp'], freq, power))
        
        conn.commit()
        conn.close()
        print(f"Saved {len(self.anomaly_db)} anomalies and {len(self.fft_history_db)} FFT records to {db_path}")

if __name__ == "__main__":
    root = tk.Tk()
    root.configure(bg="#121212")
    app = RNGFluctuationMeter(root)
    root.mainloop()