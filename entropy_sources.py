import os
import socket
import threading
import struct
import time
from abc import ABC, abstractmethod
from collections import deque

class EntropySource(ABC):
    @abstractmethod
    def open(self):
        """Initialize the source (e.g., open file or socket)."""
        pass

    @abstractmethod
    def get_name(self):
        """Return a display name for the source."""
        pass

    @abstractmethod
    def read_bit(self):
        """Read a single bit (0 or 1) from the source."""
        pass

    @abstractmethod
    def close(self):
        """Clean up resources."""
        pass

class OsEntropySource(EntropySource):
    def open(self):
        pass

    def get_name(self):
        return "OS Entropy (/dev/urandom)"

    def read_bit(self):
        # Current implementation: os.urandom(1) & 1
        return int.from_bytes(os.urandom(1), "big") & 1

    def close(self):
        pass


class Esp32NodeSource(EntropySource):
    """Entropy source that listens for UDP packets from ESP32 nodes.

    Packet format (binary):
        - NodeID: 4 bytes (uint32 little-endian)
        - SeqNum: 4 bytes (uint32 little-endian)
        - Payload: arbitrary random bytes (converted to bits, little-end bit-order)

    Notes / behavior:
        - This is *experimental*. Network reliability depends on Wi‑Fi and router settings.
        - The listener keeps a tiny per-node buffer (default maxlen=20 bits) and is intentionally
          low-latency: older bits may be dropped in favor of fresher data.
        - `read_bit()` returns an `int` (0 or 1) when available, or `None` to indicate starvation
          (no data available). The application intentionally does not fall back to OS entropy
          when starved so that the UI can surface missing-node conditions.
        - If you experience dropouts, increase the host's UDP receive buffer, use a dedicated
          AP, reserve DHCP leases, or reduce node payload rate in the firmware.
    """   

    def __init__(self, port=5000):
        self.port = port
        self.sock = None
        self.running = False
        self.nodes = {}  # node_id -> deque of bits
        self.node_status = {}  # node_id -> {'last_seen': time, 'bits_total': 0, 'bitrate': 0, 'last_stat_time': 0, 'last_bits_count': 0}
        self.starvation_count = 0
        self.thread = None
        self.lock = threading.Lock()

    def get_name(self):
        count = len(self.node_status)
        return f"ESP32 Network ({count} nodes)"

    def open(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', self.port))
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Minimize OS buffer to drop old packets immediately (~1KB)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        except Exception as e:
            print(f"Warning: Could not set SO_RCVBUF: {e}")
            
        self.sock.settimeout(1.0) # 1s timeout for periodic checks
        self.running = True
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()

    def _listen(self):
        print(f"DEBUG: UDP Listener started on port {self.port}")
        while self.running:
            try:
                data, addr = self.sock.recvfrom(2048)
                print(f"DEBUG: Packet received from {addr} ({len(data)} bytes)")
                if len(data) < 8:
                    continue
                
                # Header: NodeID (4 bytes), SeqNum (4 bytes)
                node_id, seq = struct.unpack('<II', data[:8])
                payload = data[8:]
                
                with self.lock:
                    if node_id not in self.nodes:
                        # Ultra-low latency: Keep only latest ~2 frames (20 bits)
                        # This ensures we are always reading the freshest data, even if it means dropping older bits
                        self.nodes[node_id] = deque(maxlen=20) 
                        self.node_status[node_id] = {'last_seen': 0, 'bits_total': 0, 'bitrate': 0, 'ip': addr[0]}
                    else:
                        # Update IP if it changes (e.g. DHCP)
                        self.node_status[node_id]['ip'] = addr[0]
                    
                    # Convert bytes to bits
                    bits = []
                    for b in payload:
                        for i in range(8):
                            bits.append((b >> i) & 1)
                    
                    # Check for overflow and log if needed
                    current_len = len(self.nodes[node_id])
                    if current_len + len(bits) > self.nodes[node_id].maxlen:
                        # We are dropping data
                        pass 

                    self.nodes[node_id].extend(bits)
                    now = time.time()
                    self.node_status[node_id]['last_seen'] = now
                    self.node_status[node_id]['bits_total'] += len(bits)

                    # Update bitrate every 1 second
                    elapsed = now - self.node_status[node_id].get('last_stat_time', 0)
                    if elapsed >= 1.0:
                        bits_diff = self.node_status[node_id]['bits_total'] - self.node_status[node_id].get('last_bits_count', 0)
                        self.node_status[node_id]['bitrate'] = bits_diff / elapsed
                        self.node_status[node_id]['last_stat_time'] = now
                        self.node_status[node_id]['last_bits_count'] = self.node_status[node_id]['bits_total']
            except socket.timeout:
                continue
            except socket.error as e:
                if self.running:
                    print(f"UDP Listener Socket Error: {e}")
                break
            except Exception as e:
                if self.running:
                    print(f"UDP Listener Error: {e}")
                break

    def read_bit(self):
        """
        Reads a bit from any available node. 
        In a multi-node setup, AEON might want to multiplex them.
        For now, we'll round-robin or take the first available.
        """
        with self.lock:
            for node_id in list(self.nodes.keys()):
                if self.nodes[node_id]:
                    return self.nodes[node_id].popleft()
        
        # Starvation: No bits available in any node buffer
        self.starvation_count += 1
        return None # No fallback to OS entropy, allows app to 'freeze' as requested

    def get_node_stats(self):
        with self.lock:
            return dict(self.node_status)

    def close(self):
        self.running = False
        if self.sock:
            self.sock.close()
        if self.thread:
            self.thread.join()
