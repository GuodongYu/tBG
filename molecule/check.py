from tBG.molecule.structures import QuantumDotQC
import numpy as np
from tBG.molecule.round_disk_new import coords_to_strlist
from tBG.utils import rotate_on_vec

R=10
qc = QuantumDotQC()
qc.regular_polygon(12, R, OV_orientation=15)
qc.add_hopping_pz(max_dist=2.)

def get_side_ids(qc):
    coords_bott = qc.coords[:qc.layer_nsites[0]]
    x_max = np.max(coords_bott[:,0])

    ids_side0 = np.where(np.round(coords_bott[:,0],3)==np.round(x_max,3))[0]
    coords_side0 = qc.coords[ids_side0]

    coords_str = coords_to_strlist(qc.coords)
    ids = dict(zip(coords_str, range(len(coords_str))))
    def get_ids_sidei(i):
        theta_side = i*30
        coords_side = rotate_on_vec(theta_side, coords_side0)
        if i%2:
            coords_side[:,2] = qc.h
        coords_side_str = coords_to_strlist(coords_side)
        ids_side = [ids[i] for i in coords_side_str]
        return ids_side

    ids_side = np.array([get_ids_side(i) for i in range(12)])
    return ids_side

coords = qc.coords[:,0:2]
from matplotlib import pyplot as plt
fig, ax = plt.subplots()
qc.plot(fig, ax)
for i in ids_side:
    ax.scatter(coords[i][:,0], coords[i][:,1], color='red')
plt.show()
