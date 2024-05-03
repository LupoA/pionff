import numpy as np
import matplotlib.pyplot as plt
from pionff.params import mass_pi0_GeV, mass_rho0_GeV, g_ppr_0
from pionff.formfactors.gounaris_sakurai import (
    absFpi,
    cot_delta_11,
    argFpi,
    cmplx_Fpi,
    mSquare,
    h_function,
    h_prime,
    h_prime_ee,
)
from pionff.formfactors.omnes import omnes_function


def plotGS():
    e_range = np.linspace(mass_pi0_GeV * 2, 1.1, 150)  # GeV
    plt.plot(
        e_range,
        absFpi(e_range, m_pi=mass_pi0_GeV, m_rho=mass_rho0_GeV, g_ppr=g_ppr_0) ** 2,
        label="GS",
    )
    plt.xlabel(r"$E \; [GeV] $")
    plt.ylabel(r"$ |F_{\pi}(E)|^2 $")
    plt.grid()
    plt.legend()
    plt.show()
    plt.close()

    e_range = np.linspace(mass_pi0_GeV * 2 + 1e-6, 0.75, 150)  # GeV
    plt.plot(
        e_range,
        1
        / cot_delta_11(e_range, m_pi=mass_pi0_GeV, m_rho=mass_rho0_GeV, g_ppr=g_ppr_0),
        label="GS",
    )
    plt.xlabel(r"$E \; [GeV] $")
    plt.ylabel(r"$\tan \; \delta_{11}(E) $")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()

    e_range = np.linspace(mass_pi0_GeV * 2 + 1e-6, 1.2, 150)  # GeV
    plt.plot(
        e_range,
        57.2958
        * argFpi(e_range, m_pi=mass_pi0_GeV, m_rho=mass_rho0_GeV, g_ppr=g_ppr_0),
        label="GS",
    )
    plt.xlabel(r"$E \; [GeV] $")
    plt.ylabel(r"${\rm arg} \; F_{\pi}(E) \;$ (degrees)")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()


def plot_cmplx():
    s_full_range = np.linspace(0, 1.1**2, 100)
    e_full_range = np.sqrt(s_full_range)
    e_y = np.zeros_like(e_full_range)

    Z = (e_full_range**2) + 1j * (e_y**2)
    W = cmplx_Fpi(e=np.sqrt(Z), m_pi=mass_pi0_GeV, m_rho=mass_rho0_GeV, g_ppr=g_ppr_0)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(e_full_range, np.abs(W), label=r"$|F_{\pi}(E)|$ cmpl", color="blue")
    plt.plot(
        e_full_range,
        absFpi(e_full_range, m_pi=mass_pi0_GeV, m_rho=mass_rho0_GeV, g_ppr=g_ppr_0),
        label="GS",
        color="g",
    )
    plt.xlabel(r"E $ [GeV]")
    plt.legend()
    # Plot the argument of f(z)
    plt.subplot(1, 2, 2)
    plt.plot(e_full_range, np.angle(W), label=r"arg$(F_{\pi})(E)$ cmpl", color="red")
    plt.plot(
        e_full_range,
        argFpi(e_full_range, m_pi=mass_pi0_GeV, m_rho=mass_rho0_GeV, g_ppr=g_ppr_0),
        label="GS",
        color="g",
    )
    plt.xlabel(r"E $ [GeV]")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


def plot_h_function():
    e_range = np.linspace(mass_pi0_GeV * 2 + 1e-6, 1.1, 150)  # GeV
    plt.plot(e_range, h_function(e_range, mass_pi0_GeV), color="k", label="h ")
    plt.plot(
        e_range,
        h_prime(e_range, mass_pi0_GeV),
        color="b",
        label=r"$\partial h / \partial E$",
    )
    plt.plot(
        e_range,
        h_prime_ee(e_range, mass_pi0_GeV) * (2 * e_range),
        color="r",
        ls="--",
        label=r"$\partial h / \partial E^2 \; 2E$",
    )
    plt.xlabel(r"$E \; [GeV] $")
    plt.grid()
    plt.legend()
    plt.show()
    plt.close()


def plotMsquare():
    e_range = np.linspace(mass_pi0_GeV * 2, 1.1, 150)  # GeV
    plt.plot(
        e_range,
        mSquare(e_range, m_pi=mass_pi0_GeV, m_rho=mass_rho0_GeV, g_ppr=g_ppr_0),
        color="k",
        label=r"$M^2(E)$",
    )
    plt.xlabel(r"$E \; [GeV] $")
    plt.ylabel(r"$ M^2(E) [Gev^2]$")
    plt.legend()
    plt.show()
    plt.close()


def plotOmnes():
    s_range = np.linspace(-6 * mass_pi0_GeV**2, 1.1, 150)  # GeV
    plt.plot(
        s_range,
        omnes_function(
            s_range,
            4 * mass_pi0_GeV * mass_pi0_GeV,
            argFpi,
            mass_pi0_GeV,
            mass_rho0_GeV,
            g_ppr_0,
        )
        ** 2,
        label="GS",
    )
    plt.xlabel(r"$E \; [GeV] $")
    plt.ylabel(r"$ |\Omega(E)|^2 $")
    plt.grid()
    plt.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":
    plotGS()
    plotOmnes()
