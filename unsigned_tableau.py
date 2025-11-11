import numpy as np
from custom_circuit import CustomCircuit

XZ_DICT = {
    '00': 'I',
    '10': 'X',
    '01': 'Z',
    '11': 'Y',
}

class UnsignedTableau:
    """A class representing an unsigned tableau for Clifford circuit simulation.

    The tableau tracks Pauli operators through Clifford circuits using binary
    matrices for X and Z components.
    """

    def __init__(self, xs, zs):
        """Initializes the UnsignedTableau with X and Z matrices.

        Args:
            xs (list | np.ndarray): Binary matrix representing X components of Pauli operators.
            zs (list | np.ndarray): Binary matrix representing Z components of Pauli operators.

        Raises:
            ValueError: If matrices have incompatible shapes or invalid values.
        """
        self.xs = np.array(xs)
        self.zs = np.array(zs)
        self.check_xz()
        self.n_qubits = self.xs.shape[1]
        self.n_rows = self.xs.shape[0]
        self.circuit = CustomCircuit()

    @classmethod
    def generate_random_Clifford(cls, n_qubits):
        """Generates a random Clifford circuit using the decomposition method.
        The method samples n-qubit Clifford gates uniformly with O(n^2) time complexity.
        Original work - https://arxiv.org/pdf/2008.06011

        Args:
            n_qubits (int): Number of qubits for the Clifford circuit.

        Returns:
            CustomCircuit: A compiled random Clifford circuit.
        """
        compiled_circuit = CustomCircuit()
        for n in range(n_qubits, 0, -1):
            while cls.check_comm_strings(S1 := cls.generate_random_Pauli(n), S2 := cls.generate_random_Pauli(n)):
                pass
            T = S1 + S2
            T._step1(0)
            T._step23(0)
            T._step4()
            T._step5()
            compiled_circuit += T.circuit.shift_qubits(n_qubits-n)

        return compiled_circuit


    def _step1(self, string_num):
        """First step from https://arxiv.org/pdf/2008.06011"""
        S = self[string_num]
        z = S.zs[0]
        x = S.xs[0]
        for i in range(self.n_qubits):
            if z[i]:
                if x[i]:
                    self.apply_S(i)
                else:
                    self.apply_H(i)

    def _step23(self, string_num):
        """Second and third steps from https://arxiv.org/pdf/2008.06011"""
        S = self[string_num]
        J = S.xs[0].nonzero()[0]
        while len(J) > 1:
            self.apply_CNOT(J[0], J[1])
            J = np.delete(J, 1)
        self.apply_SWAP(0, J[0])

    def _step4(self):
        """Fourth step from https://arxiv.org/pdf/2008.06011"""
        x = np.array([[False]*self.n_qubits])
        z = x.copy()
        z[0, 0] = True
        if self[1] != type(self)(x, z):
            self.apply_H(0)
            self._step1(1)
            self._step23(1)
            self.apply_H(0)

    def _step5(self):
        """Fifth step from https://arxiv.org/pdf/2008.06011"""
        signs = np.random.choice([0,1], 2)
        if signs[0] == 0:
            if signs[1] == 1:
                self.apply_X(0)
            else:
                self.apply_I(0)
        else:
            if signs[1] == 1:
                self.apply_Y(0)
            else:
                self.apply_Z(0)

    @classmethod
    def check_comm_strings(cls, first, second):
        """Checks if two Pauli strings commute with each other.

        Args:
            first (UnsignedTableau): First Pauli string.
            second (UnsignedTableau): Second Pauli string.

        Returns:
            bool: True if the strings commute, False otherwise.
        """
        x1z2 = np.bitwise_and(first.xs, second.zs)
        x2z1 = np.bitwise_and(second.xs, first.zs)
        xor = np.bitwise_xor(x1z2, x2z1)
        return np.bitwise_not(xor).sum() % 2 == first.n_qubits % 2



    def commutes(self, other):
        """Checks if this tableau commutes with another tableau.

        Args:
            other (UnsignedTableau): Another tableau to check commutativity with.

        Returns:
            bool: True if the tableaus commute, False otherwise.
        """
        return type(self).check_comm_strings(self, other)

    @classmethod
    def generate_random_Pauli(cls, n_qubits):
        """Generates a random Pauli string of specified length.

        Args:
            n_qubits (int): Number of qubits for the Pauli string.

        Returns:
            UnsignedTableau: A random Pauli string tableau.
        """
        return cls(np.random.choice([0,1], (1,n_qubits)), np.random.choice([0,1], (1,n_qubits)))

    def check_xz(self):
        """Validates the X and Z matrices for proper format and values.

        Raises:
            ValueError: If matrices have incompatible dimensions or invalid values.
        """
        if not all((self.xs.ndim == 2, self.zs.ndim == 2)):
            raise ValueError("Incompatible dimensionalities")
        if self.xs.shape != self.zs.shape:
            raise ValueError("Incompatible xs and zs shapes")
        if all((arg.dtype == np.int64 and ((arg == 0) | (arg == 1)).all()) for arg in [self.xs, self.zs]):
            self.xs = self.xs.astype(bool)
            self.zs = self.zs.astype(bool)
        elif all(arg.dtype == bool for arg in [self.xs, self.zs]):
            pass
        else:
            raise ValueError("Inappropriate values for args")


    def apply_X(self, qubit):
        """Applies an X gate to the specified qubit.

        Args:
            qubit (int): Target qubit index.
        """
        self.circuit.add_gate(['X', qubit])

    def apply_Y(self, qubit):
        """Applies an Y gate to the specified qubit.

        Args:
            qubit (int): Target qubit index.
        """
        self.circuit.add_gate(['Y', qubit])

    def apply_Z(self, qubit):
        """Applies an Z gate to the specified qubit.

        Args:
            qubit (int): Target qubit index.
        """
        self.circuit.add_gate(['Z', qubit])

    def apply_I(self, qubit):
        """Applies an I gate to the specified qubit.
        Does nothing.

        Args:
            qubit (int): Target qubit index.
        """
        pass

    def apply_H(self, qubit):
        """Applies an H gate to the specified qubit.

        Args:
            qubit (int): Target qubit index.
        """
        x_col = self.xs[:, qubit].copy()
        self.xs[:, qubit] = self.zs[:, qubit].copy()
        self.zs[:, qubit] = x_col
        self.circuit.add_gate(['H', qubit])

    def apply_S(self, qubit):
        """Applies an S gate to the specified qubit.

        Args:
            qubit (int): Target qubit index.
        """
        x_col = self.xs[:, qubit].copy()
        self.zs[:, qubit] = np.bitwise_xor(self.zs[:, qubit].copy(), x_col)
        self.circuit.add_gate(['S', qubit])

    def apply_SWAP(self, qubit1, qubit2):
        """Applies a SWAP gate between two qubits.

        Args:
            qubit1 (int): First qubit index.
            qubit2 (int): Second qubit index.
        """
        x_col = self.xs[:, qubit1].copy()
        z_col = self.zs[:, qubit1].copy()

        self.xs[:, qubit1] = self.xs[:, qubit2].copy()
        self.xs[:, qubit2] = x_col
        self.zs[:, qubit1] = self.zs[:, qubit2].copy()
        self.zs[:, qubit2] = z_col
        self.circuit.add_gate(['SWAP', [int(qubit1), int(qubit2)]])

    def apply_CNOT(self, qubit1, qubit2):
        """Applies a CNOT gate between two qubits.

        Args:
            qubit1 (int): First qubit index.
            qubit2 (int): Second qubit index.
        """
        x_col = self.xs[:, qubit1].copy()
        z_col = self.zs[:, qubit2].copy()

        self.xs[:, qubit2] = np.bitwise_xor(self.xs[:, qubit2], x_col)
        self.zs[:, qubit1] = np.bitwise_xor(self.zs[:, qubit1], z_col)
        self.circuit.add_gate(['CNOT', [int(qubit1), int(qubit2)]])

    def to_string(self):
        """Converts the tableau to human-readable Pauli strings.

        Returns:
            list: List of strings representing Pauli operators.
        """
        res = []
        for x, z in zip(self.xs, self.zs):
            res.append(self.get_string_from_row(x, z))
        return res

    def get_string_from_row(self, x, z):
        """Converts a single row to Pauli string representation.

        Args:
            x (numpy.ndarray): X components for the row.
            z (numpy.ndarray): Z components for the row.

        Returns:
            str: Pauli string representation.
        """
        string = ''
        for el_x, el_z in zip(x, z):
            string = string + XZ_DICT[str(int(el_x)) + str(int(el_z))]
        return string

    def stack(self, other):
        """Stacks another tableau vertically with this one.

        Args:
            other (UnsignedTableau): Tableau to stack with this one.
        """
        self.xs = np.vstack((self.xs, other.xs))
        self.zs = np.vstack((self.zs, other.zs))

    def __add__(self, other):
        """Adds two tableaus by vertical stacking.

        Args:
            other (UnsignedTableau): Tableau to add.

        Returns:
            UnsignedTableau: New tableau containing both inputs.
        """
        return type(self)(np.vstack((self.xs, other.xs)), np.vstack((self.zs, other.zs)))

    def __mul__(self, other):
        """Multiplies two tableaus by element-wise XOR.

        Args:
            other (UnsignedTableau): Tableau to multiply.

        Returns:
            UnsignedTableau: New tableau with XORed components.
        """
        return type(self)(np.bitwise_xor(self.xs, other.xs), np.bitwise_xor(self.zs, other.zs))

    def __getitem__(self, item):
        """Gets a subset of the tableau rows.

        Args:
            item (int, slice, list): Indexing specification.

        Returns:
            UnsignedTableau: Subset of the original tableau.
        """
        if isinstance(item, int):
            return type(self)(self.xs[None, item], self.zs[None, item])
        elif isinstance(item, slice) or isinstance(item, list):
            return type(self)(self.xs[item], self.zs[item])

    def __eq__(self, other):
        """Checks if two tableaus are equal.

        Args:
            other (UnsignedTableau): Tableau to compare with.

        Returns:
            bool: True if tableaus are equal, False otherwise.
        """
        return (self.xs == other.xs).all() and (self.zs == other.zs).all()
