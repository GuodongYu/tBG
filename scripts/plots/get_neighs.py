from tBG.molecule.round_disk import RoundDisk
from tBG.hopping import get_neighbor_shells_intralayer, filter_neig_list
import numpy as np

def get_rd_pmg_st():
    rd = RoundDisk()
    rd.make_structure(5)
    rd.remove_top_layer()
    rd.add_hopping_pz(max_dist=2.)
    pmg_st = rd.pymatgen_struct()
    return rd, pmg_st

rd, pmg_st = get_rd_pmg_st()
neigh_list = pmg_st.get_neighbor_list(10.0)
filtered_neigh_list = filter_neig_list(neigh_list)
dist_shell = get_neighbor_shells_intralayer(filtered_neigh_list, 8)[1]
rs = sorted(list(dist_shell.keys()))
a, b, c, d = neigh_list
d = np.round(d, 3)
id_from = 30
ids0 = np.where(a==id_from)[0]
xy0 = rd.coords[id_from][0:2]

for i in range(8):
    r = rs[i]
    ids1 =np.where(d==r)[0]
    ids = np.intersect1d(ids0, ids1)
    ids_to = b[ids]
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    rd.plot(fig, ax)
    rd.add_hopping_pz(max_dist=2.)
    rd.plot(fig, ax)
    for id_to in ids_to:
        xy1 = rd.coords[id_to][0:2]
        plt.plot([xy0[0], xy1[0]], [xy0[1], xy1[1]], lw=3.0)
        
    plt.savefig('NN%int.png' % i)
