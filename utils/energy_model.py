import numpy as np

# Energy Model Parameters
E_ELEC = 50e-9      # 50 nJ/bit - circuitry energy
E_FS = 10e-12       # 10 pJ/bit/m^2 - free-space amplifier
E_MP = 0.0013e-12   # 0.0013 pJ/bit/m^4 - multi-path amplifier
EDA = 5e-9          # 5 nJ/bit - data aggregation energy
PACKET_SIZE = 2000  # bits - payload packet size
D0 = np.sqrt(E_FS / E_MP)


def tx_energy(d, bits=PACKET_SIZE):
    """
    Calculate transmission energy for scalar or vector distance.

    Args:
        d: distance between transmitter and receiver
        bits: packet size in bits
    Returns:
        Energy consumed for transmission (Joules)
    """
    d = np.asarray(d, dtype=np.float32)
    bits = float(bits)
    energy = np.where(
        d < D0,
        bits * (E_ELEC + E_FS * d ** 2),
        bits * (E_ELEC + E_MP * d ** 4),
    )
    if energy.ndim == 0:
        return float(energy)
    return energy.astype(np.float32)


def rx_energy(bits=PACKET_SIZE):
    """
    Calculate reception energy.
    """
    return float(bits) * E_ELEC


def fusion_energy(m, bits=PACKET_SIZE):
    """
    Calculate data aggregation/fusion energy.
    """
    return float(m) * float(bits) * EDA


def control_tx_energy(d, bits):
    """
    Transmission energy for short control/status packets.
    """
    return tx_energy(d, bits=bits)


def control_rx_energy(bits):
    """
    Reception energy for short control packets.
    """
    return rx_energy(bits=bits)
