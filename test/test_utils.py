import unittest
from inspect import trace

import qutip as q
from gates_channels import sigmax, sigmay, CNOT, multiToffoli, H
from dm_simulation import *


class TestUtils(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ptrace(self):
        rho = generate_random_dm_matrix(7)
        final = ptrace(rho.reshape(7*2*[2]), [2,0,5])
        q_final = q.Qobj(rho, dims=[7*[2], 7*[2]]).ptrace([2,0,5]).full()
        self.assertAlmostEqual(0, (np.abs(final.reshape(2**3, 2**3)-q_final)).sum(), places=7)


    def test_matrix_to_tensor(self):
        rho = generate_random_dm_matrix(7)
        self.assertAlmostEqual(0, (np.abs(rho.reshape(2*7*[2]) - matrix_to_tensor(rho))).sum(), places=7)

    def test_tensor_to_matrix(self):
        op = np.random.randn(32,32).reshape(2*5*[2])
        self.assertAlmostEqual(0, (np.abs(op.reshape(32,32) - tensor_to_matrix(op))).sum(), places=7)


    def test_trace(self):
        rho = generate_random_dm_matrix(7)
        self.assertAlmostEqual(0, np.abs(rho.trace() - trace(matrix_to_tensor(rho))).sum(), places=7)
