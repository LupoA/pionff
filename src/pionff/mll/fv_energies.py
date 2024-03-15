import numpy as np
from pionff.utils.kinematics import k_to_E_2particles
from pionff.utils.read_luscher_table import read_luscher_function


def zero_phase_shift(*args):
    """
    dummy function for the non-interacting theory
    """
    return 0


def pi_n_minus_delta(q, n, L, m_pi, phase_shift, *phase_args):  #
    """
    the quantisation condition is phi(q) = n pi - phase_shift(k)
    this function returns n pi - phase_shift(q) where q is dimensionless
    k has dimension induced by L
    """
    res = np.pi * n
    k = 2 * np.pi * q / L
    e = k_to_E_2particles(k=k, m=m_pi)
    delta = phase_shift(e, *phase_args)
    res -= delta
    return res


def find_zero(qsq, phi_q_normalised, n, L, m_pi, phase_shift, *phase_args):
    """
    finds the intersection of phi(q) and n pi - phase_shift(k(q))
    thus solving the quantisation condition for the values q_0 and k_0
    returns q_0, k_0, the index of the arrays for k and q such that q[indx]=q_0
    """
    q = np.sqrt(qsq)
    delta_array = pi_n_minus_delta(q, n, L, m_pi, phase_shift, *phase_args)
    diff_array = delta_array - (phi_q_normalised * np.pi * qsq)
    zero_index = np.argmin(np.abs(diff_array))
    k = q[zero_index] * 2 * np.pi / L
    return q[zero_index], k, zero_index


def get_energies(L, n, m_pi, phase_shift, *phase_args):
    """
    Returns values that have satisfied the quantisation condition
    in order:
    q, k, index, Energy
    q is dimensionless, k dimensionfull
    index is such that q_array[index]=q, k_array[index]=k
    """
    qsq, phi_q_normalised, tan_phi, zeta00 = read_luscher_function()
    q = np.sqrt(qsq)
    delta_array = pi_n_minus_delta(q, n, L, m_pi, phase_shift, *phase_args)
    diff_array = delta_array - (phi_q_normalised * np.pi * qsq)
    zero_index = np.argmin(np.abs(diff_array))
    k = q[zero_index] * 2 * np.pi / L
    energy = k_to_E_2particles(k=k, m=m_pi)
    return q[zero_index], k, zero_index, energy
