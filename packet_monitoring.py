from scapy.all import *
import pandas as pd
import datetime

class PacketMonitor:
    def __init__(self, interface='eth0'):
        self.interface = interface
        self.packet_data = []

    def packet_callback(self, packet):
        packet_info = {
            'timestamp': datetime.datetime.now(),
            'src_ip': packet[IP].src if IP in packet else 'N/A',
            'dst_ip': packet[IP].dst if IP in packet else 'N/A',
            'src_port': packet[IP].sport if IP in packet and TCP in packet else 'N/A',
            'dst_port': packet[IP].dport if IP in packet and TCP in packet else 'N/A',
            'protocol': packet[IP].proto if IP in packet else 'N/A',
            'length': len(packet)
        }
        self.packet_data.append(packet_info)

        # Print captured packet information
        print(packet_info)

        # Anomaly detection logic (example: detect unusually large packets)
        if packet_info['length'] > 1500:
            print(f"Potential anomaly detected: Large packet of size {packet_info['length']} bytes")

    def start_monitoring(self):
        print(f"Starting packet capture on interface {self.interface}...")
        sniff(iface=self.interface, prn=self.packet_callback, store=0)

    def save_to_csv(self, filename='packet_data.csv'):
        df = pd.DataFrame(self.packet_data)
        df.to_csv(filename, index=False)
        print(f"Packet data saved to {filename}")

# Example usage
if __name__ == "__main__":
    packet_monitor = PacketMonitor(interface='eth0')  # Change interface if needed
    try:
        packet_monitor.start_monitoring()
    except KeyboardInterrupt:
        print("Packet capture stopped.")
        packet_monitor.save_to_csv()
