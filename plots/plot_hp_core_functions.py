import numpy as np
import matplotlib.pyplot as plt
from pionff.hp.core import _zeta_c15, _zeta_226, matcal_T_pole
from pionff.formfactors.gounaris_sakurai import phase_shift as gs_delta
from pionff.formfactors.gkpry import phase_shift as gkpry_delta
from pionff.params import (
    mass_pi0_GeV,
    Lref_fm,
    gev_fm_conversion,
    g_ppr_0,
    mass_rho0_GeV,
)


def plot_zeta():
    xaxis_size = 100
    zeta_v225 = np.zeros(xaxis_size)
    zeta_vc15 = np.zeros(xaxis_size)
    k_range = np.linspace(0.00001, 10, xaxis_size)
    for i in range(xaxis_size):
        zeta_v225[i] = _zeta_226(
            k3=k_range[i],
            n_mod=np.sqrt(2),
            L=Lref_fm * gev_fm_conversion,
            m_pi_sq=mass_pi0_GeV**2,
        )
        zeta_vc15[i] = _zeta_c15(
            k3=k_range[i],
            n_mod=np.sqrt(2),
            L=Lref_fm * gev_fm_conversion,
            m_pi_sq=mass_pi0_GeV**2,
        )
    plt.plot(k_range, zeta_v225, label="eq 2.25")
    plt.plot(k_range, zeta_vc15, label="eq c15")
    plt.xlabel(r"$k_3 \; $ $[1/fm]$")
    plt.legend()
    plt.show()


def plot_tpole():
    krange = np.linspace(1e-4, 10, 100)
    y_gs = []
    y_gkpry = []
    for i in range(len(krange)):
        y_gs.append(
            matcal_T_pole(
                krange[i],
                1,
                Lref_fm * gev_fm_conversion,
                mass_pi0_GeV,
                gs_delta,
                mass_pi0_GeV,
                mass_rho0_GeV,
                g_ppr_0,
            )
        )
        y_gkpry.append(
            matcal_T_pole(
                krange[i],
                1,
                Lref_fm * gev_fm_conversion,
                mass_pi0_GeV,
                gkpry_delta,
                mass_pi0_GeV,
                mass_rho0_GeV,
            )
        )
    plt.plot(krange, y_gs, label="GS")
    plt.plot(krange, y_gkpry, label="GKPRY")
    plt.legend(fontsize="large")
    plt.xlabel(r"$k$ [GeV]", fontsize="x-large")
    plt.xticks(fontsize="large")
    plt.yticks(fontsize="large")
    plt.show()


if __name__ == "__main__":
    plot_tpole()
    plot_zeta()
