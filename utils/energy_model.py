import numpy as np
# Energy Model Parameters
E_ELEC = 50e-9      # 50 nJ/bit - Energy consumed by transmitter/receiver circuitry
E_FS = 10e-12       # 10 pJ/bit/m² - Free space amplification energy (short distance)
E_MP = 0.0013e-12   # 0.0013 pJ/bit/m⁴ - Multi-path fading amplification energy (long distance)
EDA = 5e-9          # 5 nJ/bit - Data aggregation/fusion energy
PACKET_SIZE = 2000  # bits - Size of each data packet
D0 = np.sqrt(E_FS / E_MP)  # Threshold distance to switch between propagation models

def tx_energy(d):
    """
    Calculate transmission energy based on distance
    Args:
        d: distance between transmitter and receiver
    Returns:
        Energy consumed for transmission (Joules)
    """
    if d < D0:
        # Free space model (short distance)
        return PACKET_SIZE * (E_ELEC + E_FS * d**2)
    else:
        # Multi-path fading model (long distance)
        return PACKET_SIZE * (E_ELEC + E_MP * d**4)

def rx_energy():
    """
    Calculate reception energy
    Returns:
        Energy consumed for receiving a packet (Joules)
    """
    return PACKET_SIZE * E_ELEC

def fusion_energy(m):
    """
    Calculate data aggregation/fusion energy
    Args:
        m: number of packets to aggregate
    Returns:
        Energy consumed for data fusion (Joules)
    """
    return m * PACKET_SIZE * EDA