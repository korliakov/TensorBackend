import numpy as np


def sigmax():
    """Returns the Pauli X matrix.

    Returns:
        np.ndarray: 2x2 Pauli X matrix [[0, 1], [1, 0]].
    """
    return np.array([[0,1],[1,0]])

def sigmay():
    """Returns the Pauli Y matrix.

    Returns:
        np.ndarray: 2x2 Pauli Y matrix [[0, -1j], [1j, 0]].
    """
    return np.array([[0, -1j],[1j, 0]])

def sigmaz():
    """Returns the Pauli Z matrix.

    Returns:
        np.ndarray: 2x2 Pauli Z matrix [[1, 0], [0, -1]].
    """
    return np.array([[1, 0], [0, -1]])

def H():
    """Returns the Hadamard gate matrix.

    Returns:
        np.ndarray: 2x2 Hadamard matrix normalized by 1/sqrt(2).
    """
    return np.array([[1,1],[1,-1]])/np.sqrt(2)

def CNOT():
    """Returns the CNOT (controlled-X) gate matrix.

    Returns:
        np.ndarray: 4x4 CNOT gate matrix.
    """
    return np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,0,1],
                     [0,0,1,0]])

def SWAP():
    """Returns the SWAP gate matrix.

    Returns:
        np.ndarray: 4x4 SWAP gate matrix.
    """
    return np.array([[1,0,0,0],
                     [0,0,1,0],
                     [0,1,0,0],
                     [0,0,0,1]])

def Toffoli():
    """Returns the Toffoli (CCNOT) gate matrix.

    Returns:
        np.ndarray: 8x8 Toffoli gate matrix.
    """
    return np.array([[1,0,0,0,0,0,0,0],
                     [0,1,0,0,0,0,0,0],
                     [0,0,1,0,0,0,0,0],
                     [0,0,0,1,0,0,0,0],
                     [0,0,0,0,1,0,0,0],
                     [0,0,0,0,0,1,0,0],
                     [0,0,0,0,0,0,0,1],
                     [0,0,0,0,0,0,1,0]])

def multiToffoli(num_qubits):
    """Returns a multi-controlled Toffoli gate matrix.

    Args:
        num_qubits (int): Number of qubits for the multi-controlled Toffoli gate.

    Returns:
        np.ndarray: 2^num_qubits x 2^num_qubits multi-Toffoli gate matrix.
    """
    matrix = np.eye(2**num_qubits)
    matrix[-1, -2] = matrix[-2,-1] = 1
    matrix[-1, -1] = matrix[-2,-2] = 0
    return matrix


def Kraus_bit_flip_channel(p):
    """Returns Kraus operators for the bit flip channel.

    Args:
        p (float): Probability of bit flip error (0 <= p <= 1).

    Returns:
        np.ndarray: Array of Kraus operators for bit flip channel.
    """
    return np.array([np.eye(2) * np.sqrt(1 - p), sigmax() * np.sqrt(p)])

def Kraus_phase_flip_channel(p):
    """Returns Kraus operators for the phase flip channel.

    Args:
        p (float): Probability of phase flip error (0 <= p <= 1).

    Returns:
        np.ndarray: Array of Kraus operators for phase flip channel.
    """
    return np.array([np.eye(2) * np.sqrt(1 - p), sigmaz() * np.sqrt(p)])


def Kraus_bit_phase_flip_channel(p):
    """Returns Kraus operators for the bit-phase flip channel.

    Args:
        p (float): Probability of bit-phase flip error (0 <= p <= 1).

    Returns:
        np.ndarray: Array of Kraus operators for bit-phase flip channel.
    """
    return np.array([np.eye(2) * np.sqrt(1 - p), sigmay() * np.sqrt(p)])


def Kraus_depolarizing_channel(p):
    """Returns Kraus operators for the depolarizing channel.

    Args:
        p (float): Probability of depolarizing error (0 <= p <= 1).

    Returns:
        np.ndarray: Array of Kraus operators for depolarizing channel.
    """
    return np.array([np.eye(2) * np.sqrt(1 - 0.75 * p), sigmax() * np.sqrt(p) / 2, sigmay() * np.sqrt(p) / 2,
             sigmaz() * np.sqrt(p) / 2])

def Kraus_amplitude_damping_channel(p):
    """Returns Kraus operators for the amplitude damping channel.

    Args:
        p (float): Probability of amplitude damping (0 <= p <= 1).

    Returns:
        np.ndarray: Array of Kraus operators for amplitude damping channel.
    """
    return np.array([np.array([[1, 0], [0, np.sqrt(1 - p)]]), np.array([[0, np.sqrt(p)], [0, 0]])])


def Kraus_phase_damping_channel(p):
    """Returns Kraus operators for the phase damping channel.

    Args:
        p (float): Probability of phase damping (0 <= p <= 1).

    Returns:
        np.ndarray: Array of Kraus operators for phase damping channel.
    """
    return np.array([np.array([[1, 0], [0, np.sqrt(1 - p)]]), np.array([[0, 0], [0, np.sqrt(p)]])])