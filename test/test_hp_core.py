import numpy as np
from pionff.hp.core import _zeta_c15, _zeta_226
from pionff.params import mass_pi0_GeV, Lref_fm, gev_fm_conversion


def test_auxiliary_zeta_function():
    L_gev = Lref_fm * gev_fm_conversion
    n_values = [1, np.sqrt(2), np.sqrt(3)]
    k_range = np.linspace(1e-5, 10, 100)
    for n_mod in n_values:
        for i in range(len(k_range)):
            z226 = _zeta_226(
                k3=k_range[i], n_mod=n_mod, L=L_gev, m_pi_sq=mass_pi0_GeV**2
            )
            zc15 = _zeta_c15(
                k3=k_range[i], n_mod=n_mod, L=L_gev, m_pi_sq=mass_pi0_GeV**2
            )

            relative_diff = abs(zc15 - z226) / zc15
            assert (
                relative_diff < 1e-8
            ), "Failed comparing different definition of HP auxiliary function (zeta)."
    return
