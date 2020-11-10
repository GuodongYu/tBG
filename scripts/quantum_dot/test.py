from tBG.scripts.quantum_dot.point_group import C3v
from tBG.quantum_dot import QuantumDotQC
qc = QuantumDotQC()
qc.regular_polygon(3, 4.5)
chi_E = 93
chi_C3 = 0
chi_sigma_v = 5
a_s = C3v().rrep2irrep(chi_E,chi_C3,chi_sigma_v)
print(a_s)
