import numpy as np
from tBG.brillouin_zones import BZMoirePattern, BZHexagonal
from tBG.periodic_structures import CommensuStruct, TBG30Approximant
from tBG.scripts.get_twist_angle import twisted_angle


scale = 8
params = {'backend': 'ps',
          'axes.labelsize': 4*scale,
          'axes.linewidth': 0.3*scale,
          'font.size': 3*scale,
          'font.weight': 'normal',
          'legend.fontsize': 3*scale,
          'xtick.labelsize': 4*scale,
          'ytick.labelsize': 4*scale,
          'xtick.major.pad': 8,
          'ytick.major.pad': 8,
          'axes.labelpad': 8,
          'text.usetex': True,
          'text.latex.preamble':[r'\boldmath'],
          'figure.figsize': [12,12],
          'lines.markersize': 0.5*scale,
          'lines.linewidth':  0.3*scale,
          'font.family' : 'Times New Roman',
          'mathtext.fontset': 'stix'
          }


from matplotlib import pyplot as plt
plt.rcParams.update(params)
fig, ax = plt.subplots(1,1)

def plot_bz(ax):
    st = TBG30Approximant()
    st.make_structure(15)
    bzs = BZMoirePattern(st.latt_vec_bott, st.latt_vec_top, st.latt_vec)
    bzs.plot(ax)

    K1s_to_top, K1s_to_bott = bzs.all_kpts_after_1st_scattering()
    #ax.scatter(K1s_to_top[:,0], K1s_to_top[:,1], color='red', s=10.0)
    #ax.scatter(K1s_to_bott[:,0], K1s_to_bott[:,1], color='blue', s=10.0)

plot_bz(ax)
ax.set_xlim([-1.8, 1.8])
ax.set_ylim([-1.8, 1.8])
ax.axis('off')
plt.show()
#plt.savefig('bz.png', bbox_inches='tight',pad_inches=0, dpi=300)
