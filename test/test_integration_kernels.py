import numpy as np
from pionff.utils.amu_kernels import kernel_2002_12347 as kernelBMW
from pionff.utils.amu_kernels import kernelK_2004_03935 as kernelHP
from pionff.utils.amu_kernels import kernelEQ
from pionff.params import gev_fm_conversion, mass_muon_GeV


def test_kernel_tmr():
    x0_list_fm = np.linspace(1e-1, 5, 100)
    x0_list_gev = x0_list_fm * gev_fm_conversion
    for x0 in x0_list_gev:
        relative_diff = (
            kernelBMW(x0, mass_muon=mass_muon_GeV)
            - kernelHP(x0, mass_muon=mass_muon_GeV)
        ) / kernelBMW(x0, mass_muon=mass_muon_GeV)
        assert abs(relative_diff) < 1e-4


def test_kernel_eq():
    e_list = [0.1, 1, 5, 10, 100]
    qsq_list = [0.1, 1, 10, 100, 1e3]
    for e in e_list:
        for qsq in qsq_list:
            relative_diff = (
                kernelEQ(e, qsq, mass_muon=mass_muon_GeV, xmin=0, xmax=np.inf)
                - kernelEQ(e, qsq, mass_muon=mass_muon_GeV, xmin=1e-12, xmax=np.inf)
            ) / kernelEQ(e, qsq, mass_muon=mass_muon_GeV, xmin=0, xmax=np.inf)
            assert abs(relative_diff) < 1e-4
