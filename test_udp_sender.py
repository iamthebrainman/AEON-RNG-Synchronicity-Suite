import socket
import time
import struct
import random

HOST = "127.0.0.1" # Send to localhost first to test listener
PORT = 5000
NODE_ID = 0xDEAD

print(f"Test Sender: target {HOST}:{PORT}")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

counter = 0
while True:
    # Header: NodeID (4 bytes), SeqNum (4 bytes)
    # Payload: 64 integers (256 bytes)
    header = struct.pack('<II', NODE_ID, counter)
    
    # Generate random bytes
    payload = random.randbytes(256)
    
    packet = header + payload
    
    sock.sendto(packet, (HOST, PORT))
    print(f"Sent packet #{counter}")
    
    counter += 1
    time.sleep(0.1)
