"""
AEON Startup Wizard
Displays a welcome screen with:
  - Overview of AEON features
  - ESP32 firmware generator with all fields on one panel
  - Source selector (OS Entropy vs ESP32 Network)
  - Auto-detected host IP
"""

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
import socket
import os
from entropy_sources import OsEntropySource, Esp32NodeSource
import firmware_gen


def get_local_ip():
    """Attempt to auto-detect the host's IP address."""
    try:
        # Connect to a public DNS to infer local IP (no packets sent)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        return "192.168.1.100"  # Fallback


class StartupWizard:
    def __init__(self, root):
        self.root = root
        self.root.title("AEON - Startup Wizard")
        self.root.geometry("700x700")
        self.root.configure(bg="#1e1e1e")
        
        # Result holders
        self.entropy_source = None
        self.wizard_closed = False
        
        # Make window modal
        self.root.transient()
        self.root.grab_set()
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the startup wizard UI."""
        # Header
        header = tk.Frame(self.root, bg="#333333")
        header.pack(fill=tk.X, padx=0, pady=0)
        
        title = tk.Label(header, text="🌌 AEON: RNG Synchronicity Suite", 
                        font=("Arial", 18, "bold"), bg="#333333", fg="#00ff00")
        title.pack(pady=10)
        
        subtitle = tk.Label(header, text="Advanced Entropy Monitoring & Synchronicity Detection",
                           font=("Arial", 10), bg="#333333", fg="#aaaaaa")
        subtitle.pack(pady=5)
        
        # Main content
        content = tk.Frame(self.root, bg="#1e1e1e")
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # --- Quick Start Instructions ---
        intro_frame = tk.LabelFrame(content, text="Quick Start", 
                                   bg="#1e1e1e", fg="#ffaa00", font=("Arial", 10, "bold"))
        intro_frame.pack(fill=tk.X, pady=10)
        
        intro_text = tk.Label(intro_frame, 
            text="Choose your entropy source and configure ESP32 nodes (optional).\n"
                 "• OS Entropy: Use system /dev/urandom (stable, no hardware required)\n"
                 "• ESP32 Network: Stream entropy from wireless hardware nodes (experimental)",
            bg="#1e1e1e", fg="#aaaaaa", justify=tk.LEFT, wraplength=600)
        intro_text.pack(anchor=tk.W, padx=10, pady=10)
        
        # --- Entropy Source Selection ---
        source_frame = tk.LabelFrame(content, text="Entropy Source", 
                                    bg="#1e1e1e", fg="#00ff00", font=("Arial", 10, "bold"))
        source_frame.pack(fill=tk.X, pady=10)
        
        self.source_var = tk.StringVar(value="os")
        
        os_radio = tk.Radiobutton(source_frame, text="OS Entropy (/dev/urandom)", 
                                 variable=self.source_var, value="os", 
                                 bg="#1e1e1e", fg="white", selectcolor="#2d2d2d",
                                 activebackground="#1e1e1e", activeforeground="white",
                                 command=self._on_source_changed)
        os_radio.pack(anchor=tk.W, padx=20, pady=5)
        
        esp32_radio = tk.Radiobutton(source_frame, text="ESP32 Network Nodes", 
                                    variable=self.source_var, value="esp32",
                                    bg="#1e1e1e", fg="white", selectcolor="#2d2d2d",
                                    activebackground="#1e1e1e", activeforeground="white",
                                    command=self._on_source_changed)
        esp32_radio.pack(anchor=tk.W, padx=20, pady=5)
        
        # --- ESP32 Configuration Panel (shown when ESP32 selected) ---
        self.esp32_config_frame = tk.LabelFrame(content, text="ESP32 Configuration", 
                                               bg="#1e1e1e", fg="#00ccff", font=("Arial", 10, "bold"))
        self.esp32_config_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # SSID
        ssid_label = tk.Label(self.esp32_config_frame, text="Wi-Fi SSID:", 
                             bg="#1e1e1e", fg="white")
        ssid_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.ssid_entry = tk.Entry(self.esp32_config_frame, bg="#2d2d2d", fg="white",
                                  insertbackground="white", width=40)
        self.ssid_entry.grid(row=0, column=1, padx=10, pady=5)
        
        # Password
        pwd_label = tk.Label(self.esp32_config_frame, text="Wi-Fi Password:", 
                            bg="#1e1e1e", fg="white")
        pwd_label.grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.pwd_entry = tk.Entry(self.esp32_config_frame, bg="#2d2d2d", fg="white",
                                 insertbackground="white", show="*", width=40)
        self.pwd_entry.grid(row=1, column=1, padx=10, pady=5)
        
        # Host IP
        ip_label = tk.Label(self.esp32_config_frame, text="Host IP Address:", 
                           bg="#1e1e1e", fg="white")
        ip_label.grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        
        auto_ip = get_local_ip()
        self.ip_entry = tk.Entry(self.esp32_config_frame, bg="#2d2d2d", fg="white",
                                insertbackground="white", width=40)
        self.ip_entry.insert(0, auto_ip)
        self.ip_entry.grid(row=2, column=1, padx=10, pady=5)
        
        ip_hint = tk.Label(self.esp32_config_frame, text=f"(auto-detected: {auto_ip})",
                          bg="#1e1e1e", fg="#888888", font=("Arial", 8))
        ip_hint.grid(row=3, column=1, sticky=tk.W, padx=10)
        
        # Node ID
        nodeid_label = tk.Label(self.esp32_config_frame, text="Node ID (hex):", 
                               bg="#1e1e1e", fg="white")
        nodeid_label.grid(row=4, column=0, sticky=tk.W, padx=10, pady=5)
        self.nodeid_entry = tk.Entry(self.esp32_config_frame, bg="#2d2d2d", fg="white",
                                    insertbackground="white", width=40)
        self.nodeid_entry.insert(0, "0xAE01")
        self.nodeid_entry.grid(row=4, column=1, padx=10, pady=5)
        
        # Generate Firmware Button
        gen_fw_btn = tk.Button(self.esp32_config_frame, text="Generate Firmware",
                              bg="#0088ff", fg="white", font=("Arial", 10, "bold"),
                              command=self._generate_firmware)
        gen_fw_btn.grid(row=5, column=0, columnspan=2, pady=15)
        
        # Status label for firmware generation
        self.fw_status_label = tk.Label(self.esp32_config_frame, text="",
                                       bg="#1e1e1e", fg="#ffaa00")
        self.fw_status_label.grid(row=6, column=0, columnspan=2, padx=10, pady=5)
        
        # Hide ESP32 config by default
        self.esp32_config_frame.pack_forget()
        
        # --- Buttons (Bottom) ---
        button_frame = tk.Frame(self.root, bg="#1e1e1e")
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        start_btn = tk.Button(button_frame, text="Start AEON", 
                             bg="#008000", fg="white", font=("Arial", 12, "bold"),
                             command=self._on_start, width=15)
        start_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = tk.Button(button_frame, text="Cancel", 
                              bg="#333333", fg="white", font=("Arial", 10),
                              command=self._on_cancel, width=15)
        cancel_btn.pack(side=tk.LEFT, padx=5)
    
    def _on_source_changed(self):
        """Toggle ESP32 config panel visibility."""
        if self.source_var.get() == "esp32":
            self.esp32_config_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        else:
            self.esp32_config_frame.pack_forget()
    
    def _generate_firmware(self):
        """Generate ESP32 firmware from current config."""
        ssid = self.ssid_entry.get().strip()
        pwd = self.pwd_entry.get().strip()
        host_ip = self.ip_entry.get().strip()
        node_id_str = self.nodeid_entry.get().strip()
        
        if not ssid or not pwd or not host_ip:
            messagebox.showerror("Error", "Please fill in SSID, Password, and Host IP.")
            return
        
        try:
            # Parse node ID (hex string like "0xAE01")
            if node_id_str.startswith("0x") or node_id_str.startswith("0X"):
                node_id = int(node_id_str, 16)
            else:
                node_id = int(node_id_str)
        except ValueError:
            messagebox.showerror("Error", f"Invalid Node ID: {node_id_str}")
            return
        
        try:
            path = firmware_gen.generate_firmware(ssid, pwd, host_ip, node_id=node_id)
            self.fw_status_label.config(text=f"✓ Generated: {os.path.basename(path)}", fg="#00ff00")
            messagebox.showinfo("Success", 
                f"Firmware generated at:\n{path}\n\n"
                f"Open this file in Arduino IDE and flash to your ESP32.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate firmware:\n{str(e)}")
    
    def _on_start(self):
        """Start AEON with selected configuration."""
        source_type = self.source_var.get()
        
        if source_type == "os":
            self.entropy_source = OsEntropySource()
        else:  # esp32
            self.entropy_source = Esp32NodeSource()
        
        self.entropy_source.open()
        self.wizard_closed = True
        self.root.destroy()
    
    def _on_cancel(self):
        """Cancel startup."""
        self.wizard_closed = False
        self.root.destroy()
    
    def get_entropy_source(self):
        """Return the selected entropy source (or None if cancelled)."""
        if self.wizard_closed:
            return self.entropy_source
        return None
