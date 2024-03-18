import numpy as np
import matplotlib.pyplot as plt
from pionff.params import mass_pi0_GeV, mass_rho0_GeV, g_ppr_0
from pionff.formfactors.omnes import omnes_function
from pionff.formfactors.gounaris_sakurai import (
    absFpi,
    argFpi,
)


def plotOmega():
    s_range = np.linspace(-0.9 * 0.9, 1, 300)  # 4*mass_pi0_GeV**2, 100)

    plt.title("Gounaris Sakurai FF")
    # plt.plot(s_range, y_axis_plot, label=r'$|\Omega(s)|$')
    plt.plot(
        s_range,
        omnes_function(
            s_range, 4 * mass_pi0_GeV**2, argFpi, mass_pi0_GeV, mass_rho0_GeV, g_ppr_0
        ),
        label=r"$|\Omega(s)|$",
        color="k",
    )
    plt.plot(
        np.linspace((2 * mass_pi0_GeV) ** 2, (0.9**2), 100),
        absFpi(
            np.sqrt(np.linspace((2 * mass_pi0_GeV) ** 2, (0.9**2), 100)),
            m_pi=mass_pi0_GeV,
            m_rho=mass_rho0_GeV,
            g_ppr=g_ppr_0,
        ),
        label=r"$|F_\pi|$",
        ls="-.",
        color="r",
    )
    plt.xlabel(r"$s \; [GeV]^2 $", fontsize="large")
    plt.xticks(fontsize="large")
    plt.yticks(fontsize="large")
    plt.legend(fontsize="large")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plotOmega()
