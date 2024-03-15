import numpy as np
from scipy.misc import derivative
from pionff.utils.kinematics import e_to_k_2particles, k_to_E_2particles
from pionff.utils.read_luscher_table import read_luscher_function
from pionff.mll.fv_energies import get_energies


def get_derivatives(
    e_star, indx_point, qsq, phi_q_normalised, m_pi, phase_shift, *phase_args
):
    phi_q = phi_q_normalised * np.pi * qsq
    q = np.sqrt(qsq)
    d_phi_dq = np.gradient(phi_q, q)
    d_delta_d_e = derivative(phase_shift, e_star, dx=1e-6, args=phase_args)
    k_star = e_to_k_2particles(e_star, m_pi)
    d_delta_d_k = d_delta_d_e * 4 * k_star / e_star
    return d_phi_dq[indx_point], d_delta_d_k


def get_amplitudes(n, L, m_pi, phase_shift, absfpi, phase_args=(), absfpi_args=()):
    qsq, phi_q_normalised, _, _ = read_luscher_function()
    q_n, k_n, indx_n, energy_n = get_energies(L, n, m_pi, phase_shift, *phase_args)
    e_n = k_to_E_2particles(k=k_n, m=m_pi)
    d_phi_dq, d_delta_d_k = get_derivatives(
        e_n, indx_n, qsq, phi_q_normalised, m_pi, phase_shift, *phase_args
    )
    _der_bundle = 1 / ((q_n * d_phi_dq) + (k_n * d_delta_d_k))
    _abs_fpi = absfpi(e_n, *absfpi_args)
    a_n_sq = _der_bundle * _abs_fpi * _abs_fpi * 2
    a_n_sq *= k_n**5
    a_n_sq /= np.pi * e_n * e_n
    return a_n_sq, k_n, absfpi(e_n, *absfpi_args)
