class CustomCircuit:
    """A custom quantum circuit representation for storing quantum gates and operations.

    This class provides a flexible structure for building quantum circuits
    by adding gates and manipulating qubit indices.
    """
    def __init__(self):
        self.circuit = []

    def add_gate(self, op):
        """Adds a quantum gate operation to the circuit.

        Args:
            op (list): A list where op[0] is the gate name (str) and
                      op[1] is the qubit index/indices (int or iterable of ints).

        Raises:
            ValueError: If the operation format is incorrect, gate name is not string,
                       or qubit indices are not integers.
        """
        if not len(op) == 2:
            raise ValueError('Incorrect operation')
        if not isinstance(op[0], str):
            raise ValueError('Gate name must be str')
        if isinstance(op[1], int):
            idxs = [op[1]]
            self.circuit.append([op[0], idxs])
        else:
            try:
                idxs = list(op[1])
            except TypeError:
                raise ValueError('Gate index/s must be int or iterable of ints')
            else:
                if not all(isinstance(el, int) for el in idxs):
                    raise ValueError('Gate index/s must be int or iterable of ints')
                else:
                    self.circuit.append([op[0], idxs])

    def shift_qubits(self, shift):
        """Shifts all qubit indices in the circuit by a specified amount.

        Args:
            shift (int): The amount to add to all qubit indices.

        Returns:
            CustomCircuit: Returns self for method chaining.
        """
        for op in self.circuit:
            op[1] = [idx + shift for idx in op[1]]
        return self

    def __add__(self, other):
        """Concatenates two circuits together.

        Args:
            other (CustomCircuit): Another circuit to append to this one.

        Returns:
            CustomCircuit: New circuit containing gates from both input circuits.
        """
        new_circ = type(self)()
        new_circ.circuit.extend(self.circuit.copy())
        new_circ.circuit.extend(other.circuit.copy())
        return new_circ

    def __str__(self):
        """Returns a string representation of the circuit.

        Returns:
            str: String representation showing each gate and its qubit indices.
        """
        return "".join(f"{el[0]} {' '.join(str(idx) for idx in el[1])}"  + '\n' for el in self.circuit)