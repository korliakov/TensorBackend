import unittest

import numpy as np
from dm_simulation import *
import qutip as q
from gates_channels import sigmax, sigmay, CNOT, multiToffoli, H, Kraus_depolarizing_channel


class TestChannelAction(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_Id_channel(self):
        channel = np.array([np.eye(2)])
        rho = generate_random_dm_matrix(1)
        final = channel_action(channel, rho, np.array([0]))
        self.assertAlmostEqual(0, np.abs(final-rho).sum(), places=7)

    def test_sigmay_channel_on_i_qubit(self):
        rho = generate_random_dm_matrix(5).reshape(5 * 2 * [2])
        final = channel_action(sigmay()[np.newaxis, :], rho, np.array([3]))
        q_gate = q.tensor(q.qeye(2),q.qeye(2),q.qeye(2),q.Qobj(sigmay(),dims=[[2],[2]]),q.qeye(2))
        q_final = q_gate*q.Qobj(rho.reshape(2*[2**5]), dims=[5*[2], 5*[2]])*q_gate.dag()
        self.assertAlmostEqual(0, (np.abs(final.reshape(2*[2**5])-q_final.full())).sum(), places=7)

    def test_CNOT_on_ij_qubit(self):
        rho = generate_random_dm_matrix(3).reshape(3 * 2 * [2])
        final = channel_action(CNOT()[np.newaxis, :], rho, np.array([1,2]))
        q_gate = q.tensor(q.qeye(2), q.Qobj(CNOT(),dims=[2*[2], 2*[2]]))
        q_final = q_gate*q.Qobj(rho.reshape(2*[2**3]), dims=[3*[2], 3*[2]])*q_gate.dag()
        self.assertAlmostEqual(0, (np.abs(final.reshape(2*[2**3])-q_final.full())).sum(), places=7)

    def test_multiToffoli(self):
        dm_0 = generate_fock_dm(2,0)
        dm_1 = generate_fock_dm(2,1)
        dm_pl = H()@dm_0@H()
        rho = np.einsum('ab,cd,ef,gh->acegbdfh', dm_pl,dm_0,dm_1,dm_1)
        final = channel_action(multiToffoli(4)[np.newaxis, :], rho, np.array([0,3,2,1]))
        ideal_final = np.einsum('ab,ef,gh->aegbfh', np.array([[0.5,0,0,0.5],[0,0,0,0],[0,0,0,0],[0.5,0,0,0.5]]),dm_1,dm_1).reshape(4*2*[2])
        self.assertAlmostEqual(0, np.abs(final-ideal_final).sum(), places=7)

    def test_empty_Kraus(self):
        channel = Kraus_depolarizing_channel(0.0)
        rho = generate_random_dm_matrix(3).reshape(3 * 2 * [2])
        final = channel_action(channel, rho, np.array([1]))
        self.assertAlmostEqual(0, np.abs(final-rho).sum(), places=7)

    def test_multiqubit_strong_depol(self):
        noise = Kraus_depolarizing_channel(1.0)
        channel = np.einsum('nij, kl, ab-> nikajlb', noise, np.eye(2), np.eye(2)).reshape(-1, 8, 8)
        r0 = generate_random_dm_matrix(1)
        r1 = generate_random_dm_matrix(1)
        r2 = generate_random_dm_matrix(1)
        r3 = generate_random_dm_matrix(1)
        r4 = generate_random_dm_matrix(1)

        rho = np.einsum('ab,cd,ef,gh,ij->acegibdfhj', r0,r1,r2,r3,r4)
        final = channel_action(channel, rho, np.array([0,1,2]))
        ideal_rho = np.einsum('ab,cd,ef,gh,ij->acegibdfhj', 0.5*np.eye(2),r1,r2,r3,r4)
        self.assertAlmostEqual(0, np.abs(final-ideal_rho).sum(), places=7)


