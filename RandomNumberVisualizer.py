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
from entropy_sources import OsEntropySource, Esp32NodeSource
import firmware_gen
import tkinter.messagebox as mbox
import threading

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

        # --- ENTROPY SOURCE SETUP ---
        self.entropy_source = OsEntropySource()
        self.entropy_source.open()

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

        self.label = tk.Label(self.controls_frame, text="Ready", font=("Courier", 11, "bold"), 
                              bg="#1e1e1e", fg="#00ff00", justify=tk.LEFT)
        self.label.pack(side=tk.RIGHT, padx=10)

        # --- MENU BAR ---
        self.menubar = tk.Menu(root)
        self.root.config(menu=self.menubar)
        
        self.device_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Source", menu=self.device_menu)
        self.device_menu.add_command(label="Use OS Entropy (/dev/urandom)", command=self.set_source_os)
        # NOTE: ESP32 network node support is experimental. See README and
        # `esp32_firmware/README.md` for known connection-stability issues and
        # troubleshooting steps (router settings, DHCP reservations, UDP/firewall).
        self.device_menu.add_command(label="Use ESP32 Network Nodes", command=self.set_source_esp32)
        self.device_menu.add_separator()
        self.device_menu.add_command(label="Generate ESP32 Firmware...", command=self.open_firmware_gen)
        self.device_menu.add_separator()
        self.device_menu.add_command(label="📊 Open Pop-out Waterfall", command=self.open_waterfall_popup)

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
        
        # --- TAB 4: NODE STATUS (ESP32) ---
        node_frame = tk.Frame(self.anomaly_notebook, bg="#1e1e1e")
        self.anomaly_notebook.add(node_frame, text="Node Status")
        
        node_tree_frame = tk.Frame(node_frame, bg="#1e1e1e")
        node_tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.node_tree = ttk.Treeview(node_tree_frame, columns=('ID', 'Status', 'Bitrate', 'Last Seen', 'Source IP'), 
                                     height=15, show='headings')
        self.node_tree.heading('ID', text='Node ID')
        self.node_tree.heading('Status', text='Status')
        self.node_tree.heading('Bitrate', text='Bitrate')
        self.node_tree.heading('Last Seen', text='Last Seen')
        self.node_tree.heading('Source IP', text='Source IP')
        self.node_tree.column('ID', width=70)
        self.node_tree.column('Status', width=70)
        self.node_tree.column('Bitrate', width=80)
        self.node_tree.column('Last Seen', width=80)
        self.node_tree.column('Source IP', width=100)
        self.node_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        node_scroll = ttk.Scrollbar(node_tree_frame, orient=tk.VERTICAL, 
                                   command=self.node_tree.yview)
        node_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.node_tree.configure(yscrollcommand=node_scroll.set)
        
        # Keep reference to old anomaly_tree for backward compatibility
        self.anomaly_tree = self.stat_tree  # Default to statistical tab
        
        # Frequency drift indicator
        self.drift_label = tk.Label(self.right_panel, text="Freq Drift: 0.0000 Hz/s", 
                                   bg="#1e1e1e", fg="#00ff00", font=("Courier", 9, "bold"))
        self.drift_label.pack(pady=5)

        # --- FFT sensitivity & display controls ---
        fft_ctrl = tk.Frame(self.right_panel, bg="#1e1e1e")
        fft_ctrl.pack(pady=5, fill=tk.X)

        tk.Label(fft_ctrl, text="Bins:", bg="#1e1e1e", fg="#aaaaaa").pack(side=tk.LEFT, padx=(5,0))
        self.waterfall_bins = 64
        self.bins_scale = tk.Scale(fft_ctrl, from_=16, to=128, orient=tk.HORIZONTAL, bg="#2d2d2d", fg="white",
                                  command=lambda v: self.set_waterfall_bins(int(v)))
        self.bins_scale.set(64)
        self.bins_scale.pack(side=tk.LEFT, padx=5)

        self.auto_center_waterfall = False
        self.auto_center_var = tk.BooleanVar(value=False)
        self.auto_center_check = tk.Checkbutton(fft_ctrl, text="Auto-center", variable=self.auto_center_var,
                            command=lambda: setattr(self, 'auto_center_waterfall', self.auto_center_var.get()),
                            bg="#1e1e1e", fg="#aaaaaa", selectcolor="#2d2d2d")
        self.auto_center_check.pack(side=tk.LEFT, padx=5)
        
        self.popout_btn = tk.Button(fft_ctrl, text="📊 Pop-out", command=self.open_waterfall_popup,
                                   bg="#449944", fg="white", font=("Arial", 8, "bold"))
        self.popout_btn.pack(side=tk.RIGHT, padx=5)

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
        
        # Timing diagnostics for fluctuation investigation
        self.last_tick_time = time.time()
        self.tick_deltas = deque(maxlen=1000)
        self.bit_rate_history = deque(maxlen=1000)
        
        # Center-band oscillation tracking
        self.mid_band_history = deque(maxlen=4000) # ~200 seconds at 20fps
        self.last_osc_check = time.time()
        
        # FFT baseline for anomaly detection
        self.fft_baseline_mean = None   # np.array of shape (64,)
        self.fft_baseline_var = None    # np.array of shape (64,)
        self.fft_baseline_alpha = 0.01  # EMA smoothing factor
        self.fft_frame_counter = 0      # Throttle FFT anomaly checks
        self.last_fft_anomaly_time = 0  # Debounce FFT-PATTERN logging
        self.auto_center_waterfall = False
        self.waterfall_popup = None

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

        # Make FFT clickable
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)

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
        return self.entropy_source.read_bit()

    def set_source_os(self):
        if self.entropy_source:
            self.entropy_source.close()
        self.entropy_source = OsEntropySource()
        self.entropy_source.open()
        print("Switched to OS Entropy Source")

    def set_source_esp32(self):
        if self.entropy_source:
            self.entropy_source.close()
        self.entropy_source = Esp32NodeSource()
        self.entropy_source.open()
        print("Switched to ESP32 Node Source")

    def open_firmware_gen(self):
        """Show a dialog to generate ESP32 firmware"""
        self.root.update() # Force refresh before first dialog
        ssid = simpledialog.askstring("Firmware Gen", "Wi-Fi SSID:")
        if not ssid: return
        
        self.root.update() # Force refresh to show next dialog
        pwd = simpledialog.askstring("Firmware Gen", "Wi-Fi Password:", show='*')
        if pwd is None: return
        
        self.root.update()
        host = simpledialog.askstring("Firmware Gen", "Host IP (your PC):", initialvalue="192.168.1.100")
        if not host: return
        
        path = firmware_gen.generate_firmware(ssid, pwd, host)
        mbox.showinfo("Success", f"Firmware code generated at:\n{path}\n\nFlash this to your ESP32 via Arduino IDE.")

    def open_waterfall_popup(self):
        """Open waterfall in separate window with extended lookback"""
        if self.waterfall_popup is not None:
            try: self.waterfall_popup.lift(); return
            except: self.waterfall_popup = None

        popup = tk.Toplevel(self.root)
        popup.title("Frequency Morphing - Extended Spectrum View")
        popup.geometry("1400x900")
        popup.configure(bg="#121212")

        # F11 for Fullscreen
        popup.attributes('-fullscreen', False)
        popup.bind('<F11>', lambda e: popup.attributes('-fullscreen', not popup.attributes('-fullscreen')))
        popup.bind('<Escape>', lambda e: popup.attributes('-fullscreen', False))

        controls = tk.Frame(popup, bg="#1e1e1e")
        controls.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(controls, text="Lookback (Sec):", bg="#1e1e1e", fg="#aaaaaa").pack(side=tk.LEFT)
        depth_var = tk.IntVar(value=300)
        depth_scale = tk.Scale(controls, from_=100, to=3600, orient=tk.HORIZONTAL,
                              variable=depth_var, bg="#2d2d2d", fg="white", length=200,
                              command=lambda v: self.resize_waterfall(int(v), popup))
        depth_scale.set(300)
        depth_scale.pack(side=tk.LEFT, padx=10)

        tk.Label(controls, text="Colormap:", bg="#1e1e1e", fg="#aaaaaa").pack(side=tk.LEFT, padx=(20, 0))
        popup_cmap = tk.OptionMenu(controls, self.cmap_var, 
                                   "viridis", "plasma", "inferno", "magma", "cividis",
                                   "twilight", "cool", "winter", "ocean",
                                   command=self.update_colormap)
        popup_cmap.config(bg="#2d2d2d", fg="white", highlightthickness=0)
        popup_cmap.pack(side=tk.LEFT, padx=5)

        tk.Label(controls, text="Press F11 for Fullscreen", bg="#1e1e1e", fg="#00ff00").pack(side=tk.RIGHT, padx=10)

        popup_fig = Figure(figsize=(14, 8), dpi=100, facecolor='#121212')
        popup_ax = popup_fig.add_subplot(111, facecolor='#1e1e1e')
        
        # Extended data buffer for popup
        popup.waterfall_data = np.zeros((300, self.waterfall_bins))
        
        curr_cmap = self.cmap_var.get()
        if self.reverse_var.get():
            curr_cmap += "_r"
            
        popup.waterfall_img = popup_ax.imshow(popup.waterfall_data, aspect='auto', cmap=curr_cmap,
                             extent=[0, self.waterfall_bins, 0, 300], origin='lower', interpolation='bilinear')
        popup_ax.set_title("Long-term Spectral Evolution", color='white', fontsize=14)
        popup_ax.tick_params(colors='white')
        popup_fig.colorbar(popup.waterfall_img, ax=popup_ax, label='Relative Power')
        
        popup_canvas = FigureCanvasTkAgg(popup_fig, master=popup)
        popup_canvas.draw()
        popup_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        popup.waterfall_canvas = popup_canvas
        popup.waterfall_ax = popup_ax
        self.waterfall_popup = popup

        def on_close():
            self.waterfall_popup = None
            try:
                popup.destroy()
            except:
                pass
        popup.protocol("WM_DELETE_WINDOW", on_close)

    def resize_waterfall(self, new_depth, popup):
        """Resize the vertical depth of the waterfall storage and display"""
        if not popup: return
        old_data = popup.waterfall_data
        new_data = np.zeros((new_depth, self.waterfall_bins))
        copy_rows = min(old_data.shape[0], new_depth)
        new_data[-copy_rows:] = old_data[-copy_rows:]
        
        popup.waterfall_data = new_data
        popup.waterfall_img.set_data(new_data)
        popup.waterfall_img.set_extent([0, self.waterfall_bins, 0, new_depth])
        popup.waterfall_ax.set_ylim(0, new_depth)
        popup.waterfall_canvas.draw()

    def set_waterfall_bins(self, new_bins):
        """Update the frequency resolution/range shown"""
        self.waterfall_bins = new_bins
        # Re-init main waterfall data
        rows = self.waterfall_data.shape[0]
        self.waterfall_data = np.zeros((rows, self.waterfall_bins))
        self.waterfall_img.set_data(self.waterfall_data)
        self.waterfall_img.set_extent([0, self.waterfall_bins, 0, rows])
        
        if self.waterfall_popup:
            prows = self.waterfall_popup.waterfall_data.shape[0]
            self.waterfall_popup.waterfall_data = np.zeros((prows, self.waterfall_bins))
            self.waterfall_popup.waterfall_img.set_data(self.waterfall_popup.waterfall_data)
            self.waterfall_popup.waterfall_img.set_extent([0, self.waterfall_bins, 0, prows])
        
        # Reset FFT baseline for anomaly detection to prevent shape mismatch
        self.fft_baseline_mean = None
        self.fft_baseline_var = None
        
        # Clear signal buffer too to force a fresh FFT window
        self.signal_buffer.clear()

    def on_plot_click(self, event):
        """Handle clicks on the plots - if clicking waterfall, pop it out"""
        if event.inaxes == self.ax3:
            self.open_waterfall_popup()

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
        
        # Update main window
        self.waterfall_img.set_cmap(cmap_name)
        self.canvas.draw_idle()
        
        # Update popup window if it exists (Tethered)
        if self.waterfall_popup:
            try:
                self.waterfall_popup.waterfall_img.set_cmap(cmap_name)
                self.waterfall_popup.waterfall_canvas.draw_idle()
            except:
                self.waterfall_popup = None

    def update(self):
        if not self.running:
            return

        # 1. Generate Data
        now = time.time()
        
        # Timing Diagnostic
        dt = now - self.last_tick_time
        self.tick_deltas.append(dt)
        self.last_tick_time = now
        
        new_bits = []
        for _ in range(BITS_PER_TICK):
            b = self.get_random_bit()
            if b is None:
                # If even one bit is missing from hardware, pause this tick
                # Update status immediately to show stoppage
                self.label.config(text=f"SOURCE: {self.entropy_source.get_name()} | !! NO DATA !!", fg="#ff0000")
                self.root.after(UPDATE_MS, self.update)
                return 

            
            if b is not None:
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
        # Filter None values to prevent crash
        valid_window = [x for x in self.window if x is not None]
        if len(valid_window) == WINDOW_SIZE:
            p = sum(valid_window) / WINDOW_SIZE
            
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
            if len(self.signal_buffer) >= max(512, self.waterfall_bins * 8):
                fft_buffer_size = max(512, self.waterfall_bins * 8)
                signal = np.array(list(self.signal_buffer)[-fft_buffer_size:])
                
                # Use nfft based on bins to get appropriate resolution
                nfft_val = max(256, self.waterfall_bins * 4)
                freqs, psd = welch(signal, nperseg=nfft_val//2, noverlap=nfft_val//4, nfft=nfft_val)
                
                # Keep only what we need for the waterfall display
                psd_view = psd[:self.waterfall_bins]
                freqs_view = freqs[:self.waterfall_bins]
                
                # NEW: FFT-domain anomaly detection hook
                self.detect_fft_anomalies(freqs_view, psd_view, now)
                
                # Store in long-term DB
                self.fft_history_db.append({
                    'timestamp': now,
                    'freqs': freqs_view.copy(),
                    'psd': psd_view.copy()
                })
                
                # Auto-center stabilization: subtract rolling mean if enabled
                if getattr(self, 'auto_center_waterfall', False) and self.fft_baseline_mean is not None:
                    # Resize baseline if bins changed
                    if len(self.fft_baseline_mean) != len(psd_view):
                        self.fft_baseline_mean = psd_view.copy()
                        self.fft_baseline_var = np.ones_like(psd_view) * 1e-6
                    
                    # Show deviation from mean (stabilized view)
                    psd_norm = psd_view - self.fft_baseline_mean
                    # Re-normalize for visibility
                    psd_norm = (psd_norm - psd_norm.min()) / (psd_norm.max() - psd_norm.min() + 1e-10)
                else:
                    # Absolute power normalization
                    psd_norm = (psd_view - psd_view.min()) / (psd_view.max() - psd_view.min() + 1e-10)
                
                # Update main waterfall
                self.waterfall_data = np.roll(self.waterfall_data, -1, axis=0)
                self.waterfall_data[-1, :] = psd_norm
                self.waterfall_img.set_data(self.waterfall_data)
                
                # Update popup waterfall if open
                if self.waterfall_popup:
                    try:
                        self.waterfall_popup.waterfall_data = np.roll(self.waterfall_popup.waterfall_data, -1, axis=0)
                        self.waterfall_popup.waterfall_data[-1, :] = psd_norm
                        self.waterfall_popup.waterfall_img.set_data(self.waterfall_popup.waterfall_data)
                        self.waterfall_popup.waterfall_canvas.draw_idle()
                    except:
                        self.waterfall_popup = None
                
                # Calculate frequency drift (dominant frequency movement)
                if len(self.fft_history_db) >= 2:
                    # Find dominant frequency in current and previous
                    curr_dom = freqs_view[np.argmax(psd_view)]
                    prev_psd = self.fft_history_db[-2]['psd']
                    prev_freqs = self.fft_history_db[-2]['freqs']
                    
                    # Safety check for bin count changes
                    if len(prev_psd) == len(psd_view):
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
                    else:
                        self.drift_label.config(text="Freq Drift: --- Hz/s", fg="#555555")
            
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

            if len(self.anomaly_db) > 100 or len(self.fft_history_db) > 100:
                self.flush_to_db()

            # 6. Update Node Status UI (if using ESP32)
            if isinstance(self.entropy_source, Esp32NodeSource):
                stats = self.entropy_source.get_node_stats()
                # Clear and repopulate or sync
                # For simplicity, we'll refresh the list every tick
                existing_items = {self.node_tree.item(item)['values'][0]: item for item in self.node_tree.get_children()}
                
                for node_id, data in stats.items():
                    node_hex = hex(node_id)
                    last_seen_str = time.strftime("%H:%M:%S", time.localtime(data['last_seen']))
                    bitrate_str = f"{data['bitrate']:.1f} b/s"
                    source_ip = data.get('ip', 'Unknown')
                    status = "ONLINE" if (now - data['last_seen']) < 4.0 else "OFFLINE"
                    
                    if node_hex in existing_items:
                        self.node_tree.item(existing_items[node_hex], values=(node_hex, status, bitrate_str, last_seen_str, source_ip))
                    else:
                        self.node_tree.insert('', tk.END, values=(node_hex, status, bitrate_str, last_seen_str, source_ip))

            # 7. Handle Axis Scrolling
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

            source_name = self.entropy_source.get_name()
            starvation_warn = ""
            if hasattr(self.entropy_source, 'starvation_count') and self.entropy_source.starvation_count > 0:
                if (int(now * 2) % 2) == 0: # Faster flashing
                    starvation_warn = " | !! NO ESP32 DATA !!"
            
            diag_text = f"SOURCE: {source_name}{starvation_warn}\n"
            diag_text += f"p(1): {p:.4f} | Entropy: {entropy:.4f} | {status}\n"
            diag_text += f"Walk: {self.cumulative_val} | Log: {os.path.basename(self.log_path or '')}\n"
            diag_text += f"{chi_str} | {auto_str} | {streak_str}\n"
            # Fluctuation Diagnostic readout
            avg_dt = np.mean(self.tick_deltas) if self.tick_deltas else 0.05
            actual_bps = (BITS_PER_TICK / avg_dt) if avg_dt > 0 else 0
            diag_text += f"Tick Δ: {avg_dt*1000:.1f}ms | Actual: {actual_bps:.1f} bps\n"
            
            self.label.config(text=diag_text)
            
            # Flash label red if starving
            if starvation_warn:
                self.label.config(fg="#ff0000")
            else:
                self.label.config(fg="#00ff00")

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
        if self.fft_baseline_mean is None or len(self.fft_baseline_mean) != len(psd):
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
        num_bins = len(psd)
        q1, q3 = num_bins // 4, 3 * num_bins // 4
        low_band = np.sum(psd[:q1])
        mid_band = np.sum(psd[q1:q3])
        high_band = np.sum(psd[q3:])
        total_power = low_band + mid_band + high_band + 1e-12
        band_ratio = (low_band / total_power, mid_band / total_power, high_band / total_power)
        
        # Track mid-band power for oscillation detection
        self.mid_band_history.append(mid_band / total_power)
        
        # Periodically check for slow oscillations (e.g. 100s period)
        if now - self.last_osc_check > 10.0 and len(self.mid_band_history) > 2000:
            self.last_osc_check = now
            # Use autocorrelation on the power signal to find periodicities
            m_data = np.array(self.mid_band_history)
            m_data = m_data - np.mean(m_data) # Zero-center
            
            # Look for 100s peak (at ~20Hz updates, 100s is lag 2000)
            # We check a range around 100s
            search_window = 500 # Look at lags 1500 to 2500
            if len(m_data) > 2500:
                acorr = correlate(m_data, m_data, mode='full')
                acorr = acorr[len(acorr)//2:] # Positive lags
                # Normalize
                acorr /= acorr[0]
                
                # Search for peak around 100s (lag 2000)
                sub_acorr = acorr[1500:2500]
                max_lag_idx = np.argmax(sub_acorr)
                max_val = sub_acorr[max_lag_idx]
                
                if max_val > 0.4: # Significant periodic logic
                    detected_period = (max_lag_idx + 1500) * (UPDATE_MS / 1000.0)
                    self.stat_tree.insert('', 0, values=(time.strftime("%H:%M:%S"), "OSC-DET", f"MidBand {detected_period:.1f}s cycle"))
                    print(f"DEBUG: Found oscillation at {detected_period:.1f}s period (val={max_val:.3f})")
        
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
            # Stop Logging & Export JSON (Backgrounded)
            log_path_copy = self.log_path
            csv_file_copy = self.current_log_file
            
            self.current_log_file = None
            self.current_csv_writer = None
            
            def final_save():
                if csv_file_copy:
                    csv_file_copy.close()
                
                # Export Walk Data
                export_path = log_path_copy.replace('.csv', '.json')
                export_data = {
                    'timestamps': list(self.timestamps),
                    'cumulative_history': list(self.cumulative_history),
                    'p_history': list(self.p_history),
                    'patterns_history': self.patterns_history
                }
                with open(export_path, 'w') as f:
                    json.dump(export_data, f)
                
                # Final flush of current memory buffers
                self.flush_to_db()
                self.save_patterns()
                print(f"Final data export to {export_path} complete.")

            threading.Thread(target=final_save, daemon=True).start()

    def flush_to_db(self):
        """Move current memory buffers to SQLite and clear them to prevent RAM bloat"""
        if not self.anomaly_db and not self.fft_history_db:
            return
            
        anomalies_to_save = list(self.anomaly_db)
        fft_to_save = list(self.fft_history_db)
        
        self.anomaly_db.clear()
        self.fft_history_db.clear()
        
        def save_task():
            db_path = os.path.join(LOG_DIR, "rng_longterm.db")
            try:
                conn = sqlite3.connect(db_path)
                c = conn.cursor()
                
                # Create tables
                c.execute('''CREATE TABLE IF NOT EXISTS anomalies
                             (timestamp REAL, type TEXT, details TEXT, p REAL, entropy REAL, walk INTEGER)''')
                c.execute('''CREATE TABLE IF NOT EXISTS fft_history
                             (timestamp REAL, freq_bin INTEGER, power REAL)''')
                
                # Store anomalies
                for anomaly in anomalies_to_save:
                    c.execute("INSERT INTO anomalies VALUES (?, ?, ?, ?, ?, ?)",
                             (anomaly['timestamp'], anomaly['type'], anomaly['details'],
                              anomaly['p'], anomaly['entropy'], anomaly['walk']))
                
                # Store FFT spectra (downsampled to save space)
                for record in fft_to_save:
                    for i, (freq, power) in enumerate(zip(record['freqs'], record['psd'])):
                        if i % 4 == 0:  # Store every 4th bin to save space
                            c.execute("INSERT INTO fft_history VALUES (?, ?, ?)",
                                     (record['timestamp'], freq, power))
                
                conn.commit()
                conn.close()
                # print(f"Flushed {len(anomalies_to_save)} anomalies and {len(fft_to_save)} FFT records to disk.")
            except Exception as e:
                print(f"Error flushing to DB: {e}")

        # Run save in background thread
        threading.Thread(target=save_task, daemon=True).start()

    def save_longterm_data(self):
        # Redirected to flush_to_db() which is non-blocking
        self.flush_to_db()

if __name__ == "__main__":
    root = tk.Tk()
    root.configure(bg="#121212")
    app = RNGFluctuationMeter(root)
    root.mainloop()