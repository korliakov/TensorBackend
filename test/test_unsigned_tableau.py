import unittest

from unsigned_tableau import UnsignedTableau


class TestCustomCircuit(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_generate_random_Clifford(self):
        for i in range(1, 10):
            UnsignedTableau.generate_random_Clifford(i)

    def test_generate_random_Pauli(self):
        for i in range(1, 10):
            UnsignedTableau.generate_random_Pauli(i)

    def test_check_comm_strings_1(self):
        t1 = UnsignedTableau([[1,1]], [[0,0]])
        t2 = UnsignedTableau([[1,1]], [[0,0]])
        self.assertTrue(t1.commutes(t2))

    def test_check_comm_strings_2(self):
        t1 = UnsignedTableau([[1,1]], [[0,0]])
        t2 = UnsignedTableau([[1,1]], [[1,0]])
        self.assertFalse(t1.commutes(t2))

    def test_apply_H(self):
        t1 = UnsignedTableau([[1, 1]], [[0, 0]])
        t1.apply_H(0)
        t2 = UnsignedTableau([[0, 1]], [[1, 0]])
        self.assertEqual(t1, t2)

    def test_apply_S(self):
        t1 = UnsignedTableau([[1, 1]], [[0, 0]])
        t1.apply_S(0)
        t2 = UnsignedTableau([[1, 1]], [[1, 0]])
        self.assertEqual(t1, t2)

    def test_apply_SWAP(self):
        t1 = UnsignedTableau([[1, 0]], [[0, 0]])
        t1.apply_SWAP(0,1)
        t2 = UnsignedTableau([[0, 1]], [[0, 0]])
        self.assertEqual(t1, t2)

    def test_to_string(self):
        t1 = UnsignedTableau([[1, 0]], [[0, 0]])
        self.assertEqual(t1.to_string(), ['XI'])

    def test_eq(self):
        t1 = UnsignedTableau([[1, 0]], [[0, 0]])
        t2 = UnsignedTableau([[1, 0]], [[0, 0]])
        self.assertTrue(t1 == t2)

    def test_add(self):
        t1 = UnsignedTableau([[1, 0]], [[0, 0]])
        t2 = UnsignedTableau([[1, 0]], [[0, 0]])
        t3 = UnsignedTableau([[1, 0], [1, 0]], [[0, 0], [0, 0]])
        self.assertTrue(t1 + t2 == t3)

    def test_mul(self):
        t1 = UnsignedTableau([[1, 0]], [[0, 0]])
        t2 = UnsignedTableau([[1, 0]], [[0, 0]])
        t3 = UnsignedTableau([[0, 0]], [[0, 0]])
        self.assertTrue(t1 * t2 == t3)

    def test_get_item(self):
        t1 = UnsignedTableau([[1, 0], [0, 0]], [[0, 0], [0, 0]])
        t2 = UnsignedTableau([[0, 0]], [[0, 0]])
        self.assertTrue(t1[1] == t2)
