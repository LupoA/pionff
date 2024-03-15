import matplotlib.pyplot as plt
from pionff.utils.read_luscher_table import read_luscher_function

qsq, phi_q_normalised, tan_phi, zeta00 = read_luscher_function()

plt.plot(qsq, phi_q_normalised, label=r"$\phi(q)/(\pi q^2)$")
plt.xlabel(r"$q^2$")
plt.legend()
plt.show()
plt.close()

plt.plot(qsq, tan_phi, label=r"$\tan (\phi(q)) $")
plt.xlabel(r"$q^2$")
plt.ylim(top=1e2, bottom=-1e2)
plt.xlim(left=0, right=1e1)
plt.legend()
plt.show()
plt.close()

qsq_filtered = []
zeta00_filtered = []
for q, z in zip(qsq, zeta00):
    if q <= 10:  # if q <= 40 and q.is_integer():
        qsq_filtered.append(q)
        zeta00_filtered.append(z)
plt.plot(qsq_filtered, zeta00_filtered, label=r"$Z_{00}/(1; q^2)$")
plt.xlabel(r"$q^2$")
plt.ylim(top=1e2, bottom=-1e2)
plt.xticks(fontsize="large")
plt.yticks(fontsize="large")
plt.legend(fontsize="large")
plt.grid()
plt.show()
