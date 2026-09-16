"""
System configuration and default constants for Network Simulation.
"""

# Default Simulation Parameters
DEFAULT_RANDOM_SEED: int = 42
DEFAULT_SIMULATION_TICKS: int = 20

# Default Link Characteristics
DEFAULT_BANDWIDTH_MBPS: float = 1000.0  # 1 Gbps default link capacity
DEFAULT_LINK_DELAY_MS: float = 2.0      # 2 ms propagation delay
DEFAULT_BUFFER_CAPACITY_PACKETS: int = 1000

# Packet Data Defaults
DEFAULT_PACKET_SIZE_BYTES: int = 1500   # Standard MTU size in bytes

# Congestion Threshold
CONGESTION_THRESHOLD_UTILIZATION: float = 0.85
