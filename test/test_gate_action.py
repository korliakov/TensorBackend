import unittest

from dm_simulation import *
import qutip as q
from gates_channels import sigmax, sigmay, CNOT, multiToffoli, H


class TestGateAction(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_id_one_qubit_gate(self):
        rho = generate_random_dm_matrix(1)
        final = gate_action(np.eye(2), rho, np.array([0]))
        self.assertAlmostEqual(0, (np.abs(final-rho)).sum(), places=7)

    def test_id_multi_qubit_gate(self):
        rho = generate_random_dm_matrix(7).reshape(7 * 2 * [2])
        final = gate_action(np.eye(2**7), rho, np.array([0,1,2,3,4,5,6]))
        self.assertAlmostEqual(0, (np.abs(final-rho)).sum(), places=7)

    def test_sigmax_on_i_qubit(self):
        rho = generate_random_dm_matrix(5).reshape(5 * 2 * [2])
        final = gate_action(sigmax(), rho, np.array([3]))
        q_gate = q.tensor(q.qeye(2),q.qeye(2),q.qeye(2),q.Qobj(sigmax(),dims=[[2],[2]]),q.qeye(2))
        q_final = q_gate*q.Qobj(rho.reshape(2*[2**5]), dims=[5*[2], 5*[2]])*q_gate.dag()
        self.assertAlmostEqual(0, (np.abs(final.reshape(2*[2**5])-q_final.full())).sum(), places=7)


    def test_sigmay_on_i_qubit(self):
        rho = generate_random_dm_matrix(5).reshape(5 * 2 * [2])
        final = gate_action(sigmay(), rho, np.array([3]))
        q_gate = q.tensor(q.qeye(2),q.qeye(2),q.qeye(2),q.Qobj(sigmay(),dims=[[2],[2]]),q.qeye(2))
        q_final = q_gate*q.Qobj(rho.reshape(2*[2**5]), dims=[5*[2], 5*[2]])*q_gate.dag()
        self.assertAlmostEqual(0, (np.abs(final.reshape(2*[2**5])-q_final.full())).sum(), places=7)

    def test_CNOT_on_ij_qubit(self):
        rho = generate_random_dm_matrix(3).reshape(3 * 2 * [2])
        final = gate_action(CNOT(), rho, np.array([1,2]))
        q_gate = q.tensor(q.qeye(2), q.Qobj(CNOT(),dims=[2*[2], 2*[2]]))
        q_final = q_gate*q.Qobj(rho.reshape(2*[2**3]), dims=[3*[2], 3*[2]])*q_gate.dag()
        self.assertAlmostEqual(0, (np.abs(final.reshape(2*[2**3])-q_final.full())).sum(), places=7)

    def test_multiToffoli(self):
        dm_0 = generate_fock_dm(2,0)
        dm_1 = generate_fock_dm(2,1)
        dm_pl = H()@dm_0@H()
        dm_m = H()@dm_1@H()
        rho = np.einsum('ab,cd,ef,gh->acegbdfh', dm_pl,dm_0,dm_1,dm_1)
        final = gate_action(multiToffoli(4), rho, np.array([0,3,2,1]))
        ideal_final = np.einsum('ab,ef,gh->aegbfh', np.array([[0.5,0,0,0.5],[0,0,0,0],[0,0,0,0],[0.5,0,0,0.5]]),dm_1,dm_1).reshape(4*2*[2])
        self.assertAlmostEqual(0, np.abs(final-ideal_final).sum(), places=7)


if __name__ == '__main__':
    unittest.main()
