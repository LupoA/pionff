import numpy as np
from pionff.params import mass_pic_GeV, mass_pi0_GeV, mass_rho0_GeV, width_rho0_neutral, g_ppr_0, g_ppr_c
from pionff.formfactors.gounaris_sakurai import g_from_gamma, argFpi, mSquare

def test_g():
    g0 = g_from_gamma(m_pi=mass_pi0_GeV, m_rho=mass_rho0_GeV, gamma_rho=width_rho0_neutral)
    assert g0 - g_ppr_0
    diff = 90 - (57.2958 * argFpi(e=mass_rho0_GeV+1e-8, m_pi=mass_pi0_GeV, m_rho=mass_rho0_GeV, g_ppr=g_ppr_0))
    assert diff < 1e-3
    diff = mass_rho0_GeV - np.sqrt(mSquare(e=mass_rho0_GeV, m_pi=mass_pi0_GeV, m_rho=mass_rho0_GeV, g_ppr=g_ppr_0))
    assert diff < 1e-8
