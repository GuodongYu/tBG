from tBG.periodic_structures import TBG30Approximant
from tBG.brillouin_zones import BZMoirePattern, BZHexagonal

st = TBG30Approximant()
st.make_structure(15)
bzm = BZMoirePattern(st.latt_vec_bott, st.latt_vec_top, st.latt_vec)
bz_bott = BZHexagonal(st.latt_vec_bott)
bz_top = BZHexagonal(st.latt_vec_top)
bz_bott.special_points()
bz_bott.special_points()
bzm.get_all_Qs()
bzm.all_kpts_after_1st_scattering()

from matplotlib import pyplot as plt
fig, ax = plt.subplots()
bzm.plot(ax, SBZ=False)
i=0
for K in bz_bott.special_points()['K']:
    ax.text(K[0], K[1], '$K_%s$' % i, color='blue')
    i = i+1
i=0
for K in bz_top.special_points()['K']:
    ax.text(K[0], K[1], '$\\tilde{K}_%s$' % i, color='red')
    i = i+1

ax.axis('equal')
plt.show()
