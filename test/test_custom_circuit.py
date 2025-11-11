import unittest

from custom_circuit import CustomCircuit


class TestCustomCircuit(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_add_gate(self):
        circ1 = CustomCircuit()
        circ1.add_gate(('H', 1))
        circ1.add_gate(('CNOT', (1, 3)))
        self.assertEqual(str(circ1), "H 1\nCNOT 1 3\n")

    def test_magic_add(self):
        circ1 = CustomCircuit()
        circ2 = CustomCircuit()
        circ1.add_gate(('H', [1]))
        circ2.add_gate(('H', 1))
        circ2.add_gate(('CNOT', (1, 3)))
        circ2 += circ1
        self.assertEqual(str(circ2), "H 1\nCNOT 1 3\nH 1\n")

    def test_shift_qubits(self):
        circ1 = CustomCircuit()
        circ1.add_gate(('H', [1]))
        circ1.add_gate(('H', 1))
        circ1.add_gate(('CNOT', (1, 3)))
        circ1.shift_qubits(1)
        self.assertEqual(str(circ1), "H 2\nH 2\nCNOT 2 4\n")


