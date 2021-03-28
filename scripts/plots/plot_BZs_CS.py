import numpy as np
from tBG.crystal.brillouin_zones import BZMoirePattern, BZHexagonal
from tBG.crystal.structures import CommensuStruct_pq, TBG30Approximant
from tBG.scripts.get_twist_angle import twisted_angle
import tBG

from matplotlib import pyplot as plt
#plt.rcParams.update(tBG.params)
mns = [[6,7], [11,15], [5,8],[9,17],[4,9],[4,11]]
#mns = [[1,2], [6,7], [8,9], [11,12]]
mns = [[3,7], [2,5], [1,3], [3,5]]
mns = [[1,2], [2,3],[4,5],[4,11]]
mns = [[1,7]]
nrow = 1
ncol = int(len(mns)/nrow)
fig, axes = plt.subplots(nrow,ncol, figsize=[20, 20])
if nrow==1:
    if ncol==1:
        axes = [[axes]]
    else:
        axes = [axes]

def plot_bz(ax, m, n):
    cs = CommensuStruct_pq()
    cs.make_structure(m,n)
    bzs = BZMoirePattern(cs.layer_latt_vecs[0], cs.layer_latt_vecs[1], cs.latt_vec)
    bzs.plot(ax)

    K1s_to_top, K1s_to_bott = bzs.all_kpts_after_1st_scattering()
    #ax.scatter(K1s_to_top[:,0], K1s_to_top[:,1], color='red', s=10.0)
    #ax.scatter(K1s_to_bott[:,0], K1s_to_bott[:,1], color='blue', s=10.0)
    ax.scatter(0,0, color='black', s=40., marker='*', alpha=0.5)
    ax.set_title('(%s,%s) $%.2f^{\circ}$' % (m, n, cs.twist_angle))

for i in range(nrow):
    for j in range(ncol):
        k = i*ncol + j
        ax = axes[i][j]
        m,n = mns[k]
        plot_bz(ax, m, n)
plt.show()
