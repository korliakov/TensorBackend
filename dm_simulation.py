import numpy as np


def gate_action(gate, tensor_dm, qubit_idx):
    """
    Applies unitary evolution on density matrix

    Args:
        gate (np.ndarray): matrix representation of the gate
        tensor_dm (np.ndarray): tensor representation of density matrix
        qubit_idx (Union[List[int], np.ndarray]): list of qubit indexes being altered by gate

    Returns:
        (np.ndarray) Density matrix after unitary evolution
    """
    qubit_idx = np.array(qubit_idx)
    tensor_dm_ndim = tensor_dm.ndim
    len_qubit_idx = qubit_idx.size


    tensor_gate = gate.reshape(len_qubit_idx * 2 * [2])
    tensor_gate_ndim = tensor_gate.ndim
    assert len_qubit_idx == int((np.log2(gate.size)/2))

    dm_contraction_idx = np.arange(tensor_dm_ndim)
    gate_contraction_idx_1 = np.arange(tensor_dm_ndim, tensor_dm_ndim + tensor_gate_ndim)
    gate_contraction_idx_1[len_qubit_idx:] = dm_contraction_idx[qubit_idx]

    gate_contraction_idx_2 = np.arange(tensor_dm_ndim + tensor_gate_ndim, tensor_dm_ndim + 2 * tensor_gate_ndim)
    gate_contraction_idx_2[len_qubit_idx:] = dm_contraction_idx[qubit_idx + int(tensor_dm_ndim / 2)]

    result_idx = np.arange(tensor_dm_ndim)
    result_idx[qubit_idx] = gate_contraction_idx_1[:len_qubit_idx]
    result_idx[qubit_idx + int(tensor_dm_ndim / 2)] = gate_contraction_idx_2[:len_qubit_idx]
    result = np.einsum(tensor_gate, gate_contraction_idx_1, tensor_dm, dm_contraction_idx, tensor_gate.conj(),
                       gate_contraction_idx_2, result_idx)

    return result


def channel_action(Kraus_channel, tensor_dm, qubit_idx):
    """
    Applies non-unitary evolution (described by Kraus operators) on density matrix


    Args:
        Kraus_channel (np.ndarray): tensor representation of quantum channel. First index enumerates Kraus operators
        tensor_dm (np.ndarray): tensor representation of density matrix
        qubit_idx (Union[List[int], np.ndarray]): list of qubit indexes being altered by channel

    Returns:
        (np.ndarray) Density matrix after non-unitary evolution
    """
    qubit_idx = np.array(qubit_idx)
    Kraus_num = Kraus_channel.shape[0]
    len_qubit_idx = qubit_idx.size
    assert len_qubit_idx == int(np.log2(Kraus_channel.shape[1]))
    tensor_channel = Kraus_channel.reshape([Kraus_num] + len_qubit_idx * 2 * [2])
    tensor_channel_ndim = tensor_channel.ndim

    tensor_dm_ndim = tensor_dm.ndim
    dm_contraction_idx = np.arange(tensor_dm_ndim)

    channel_contraction_idx_1 = np.arange(tensor_dm_ndim, tensor_dm_ndim+tensor_channel_ndim)
    channel_contraction_idx_1[len_qubit_idx+1:] = dm_contraction_idx[qubit_idx]

    channel_contraction_idx_2 = np.arange(tensor_dm_ndim+tensor_channel_ndim, tensor_dm_ndim+2*tensor_channel_ndim)
    channel_contraction_idx_2[len_qubit_idx+1:] = dm_contraction_idx[qubit_idx + int(tensor_dm_ndim / 2)]
    channel_contraction_idx_2[0] = channel_contraction_idx_1[0]

    result_idx = np.arange(tensor_dm_ndim)
    result_idx[qubit_idx] = channel_contraction_idx_1[1:len_qubit_idx+1]
    result_idx[qubit_idx + int(tensor_dm_ndim / 2)] = channel_contraction_idx_2[1:len_qubit_idx+1]
    result = np.einsum(tensor_channel, channel_contraction_idx_1, tensor_dm, dm_contraction_idx, tensor_channel.conj(),
                       channel_contraction_idx_2, result_idx)

    return result

def generate_random_dm_matrix(num_qubits):
    """
    Generates random density matrix (not Haar's random) for a given number of qubits

    Args:
        num_qubits (int): number of qubits

    Returns:
        (np.ndarray) Random density matrix with shape (2^num_qubits, 2^num_qubits)

    """
    dim = 2**num_qubits
    dm = np.random.random(dim*dim).reshape([dim,dim]) + 1j*np.random.random(dim*dim).reshape([dim,dim])
    dm = dm + dm.conjugate().T
    dm = dm/dm.trace()
    return dm

def generate_fock_dm(dim, n):
    """
    Generates density matrix of the given Fock state

    Args:
        dim (int): Hilbert space dimension
        n (int): number of basis vector (number of photons)

    Returns:
        (np.ndarray) Density matrix with shape (dim, dim)
    """
    matrix = np.zeros((dim,dim))
    matrix[n,n]=1
    return matrix

def ptrace(tensor_dm, qubit_idx):
    """
    Computes partial trace of given density matrix with given qubits remaining.

    Args:
        tensor_dm (np.ndarray): tensor representation of initial density matrix
        qubit_idx (Union[List[int], np.ndarray]): list of remaining indexes

    Returns:
        (np.ndarray) Reduced density matrix
    """

    qubit_idx = np.array(qubit_idx)
    len_qubit_idx = qubit_idx.size
    tensor_dm_ndim = tensor_dm.ndim
    sum_idx = np.delete(np.arange(int(tensor_dm_ndim/2)), qubit_idx)


    contraction_1 = np.arange(int(tensor_dm_ndim/2))
    contraction_2 = np.arange(int(tensor_dm_ndim/2), tensor_dm_ndim)
    contraction_2[sum_idx] = contraction_1[sum_idx]

    result_1 = np.delete(contraction_1, sum_idx)
    result_2 = np.delete(contraction_2, sum_idx)

    final = np.einsum(tensor_dm, np.hstack((contraction_1, contraction_2)), np.hstack((result_1, result_2)))
    return final

def tensor_to_matrix(operator):
    """
    Converts operator from tensor representation to matrix representation

    Args:
        operator (np.ndarray): tensor representation of operator

    Returns:
        (np.ndarray) matrix representation of operator
    """
    num_qubits = int(operator.ndim / 2)
    op = operator.reshape([2**num_qubits, 2**num_qubits])
    return op


def matrix_to_tensor(operator):
    """
    Converts operator from matrix representation to tensor representation

    Args:
        operator (np.ndarray): matrix representation of operator

    Returns:
        (np.ndarray) tensor representation of operator
    """
    num_qubits = int(np.log2(operator.shape[0]))
    op = operator.reshape(2 * num_qubits * [2])
    return op



def tensor_dm_trace(tensor_dm) -> float:
    """
    Computes trace of denity matrix in tensor representation

    Args:
        tensor_dm (np.ndarray): density matrix in tensor representation

    Returns:
        (float) Trace of given density matrix in tensor form
    """
    return np.trace(tensor_to_matrix(tensor_dm))
