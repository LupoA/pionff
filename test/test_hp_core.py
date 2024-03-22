import numpy as np
from pionff.hp.core import _zeta_c15, _zeta_226
from pionff.params import (
    mass_pi0_GeV,
    Lref_fm,
    gev_fm_conversion,
)


def test_auxiliary_zeta_function():
    L_gev = Lref_fm * gev_fm_conversion
    n_values = [1, np.sqrt(2), np.sqrt(3)]
    k_range = np.linspace(1e-5, 10, 100)
    for n_mod in n_values:
        for i in range(len(k_range)):
            z226 = _zeta_226(k3=k_range[i], n_mod=n_mod, L=L_gev, msq=mass_pi0_GeV**2)
            zc15 = _zeta_c15(k3=k_range[i], n_mod=n_mod, L=L_gev, msq=mass_pi0_GeV**2)

            relative_diff = abs(zc15 - z226) / zc15
            assert (
                relative_diff < 1e-8
            ), "Failed comparing different definition of HP auxiliary function (zeta)."
    return


def plot_zeta():
    import matplotlib.pyplot as plt

    xaxis_size = 100
    zeta_v225 = np.zeros(xaxis_size)
    zeta_vc15 = np.zeros(xaxis_size)
    k_range = np.linspace(0.00001, 10, xaxis_size)
    for i in range(xaxis_size):
        zeta_v225[i] = _zeta_226(
            k3=k_range[i],
            n_mod=np.sqrt(2),
            L=Lref_fm * gev_fm_conversion,
            msq=mass_pi0_GeV**2,
        )
        zeta_vc15[i] = _zeta_c15(
            k3=k_range[i],
            n_mod=np.sqrt(2),
            L=Lref_fm * gev_fm_conversion,
            msq=mass_pi0_GeV**2,
        )
    plt.plot(k_range, zeta_v225, label="eq 2.25")
    plt.plot(k_range, zeta_vc15, label="eq c15")
    plt.xlabel(r"$k_3 \; $ $[1/fm]$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_auxiliary_zeta_function()
