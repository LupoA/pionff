from pionff.hp.fv_corrections import correction_amu
from pionff.formfactors.gounaris_sakurai import phase_shift as gs_delta
from pionff.formfactors.gkpry import phase_shift as gkpry_delta
from pionff.params import (
    mass_pi0_GeV,
    Lref_fm,
    gev_fm_conversion,
    g_ppr_0,
    mass_rho0_GeV,
    mass_muon_GeV,
)


def gs(x0cut_fm):
    delta_amu, err = correction_amu(
        x0cut_fm * gev_fm_conversion,
        Lref_fm * gev_fm_conversion,
        mass_muon_GeV,
        mass_pi0_GeV,
        gs_delta,
        mass_pi0_GeV,
        mass_rho0_GeV,
        g_ppr_0,
    )

    print("GS")
    print(delta_amu * 1e10)
    print(err * 1e10)
    return delta_amu


def gkpry(x0cut_fm):
    delta_amu, err = correction_amu(
        x0cut_fm * gev_fm_conversion,
        Lref_fm * gev_fm_conversion,
        mass_muon_GeV,
        mass_pi0_GeV,
        gkpry_delta,
        mass_pi0_GeV,
        mass_rho0_GeV,
    )

    print("GKPRY")
    print(delta_amu * 1e10)
    print(err * 1e10)
    return delta_amu


if __name__ == "__main__":
    x0cut_fm = 1.682
    a = gkpry(x0cut_fm)
    b = gs(x0cut_fm)
    print("diff =", abs(a - b) / a)
