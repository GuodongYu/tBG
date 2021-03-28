from tBG.molecule.round_disk_new import get_neigh_pairs_in_graphene_OBC_in_parellel
from itertools import combinations
import numpy as np
from scipy.linalg.lapack import zheev
import pickle

def get_NNN_with_mid(coords, latt_vec, idx_from=0):
    """
    coords [coords_a, coords_b] coordinates of sublatt a and b 
    latt_vec: lattice vectors
    """
    pairs_NN = get_neigh_pairs_in_graphene_OBC_in_parellel(1, coords, latt_vec, idx_from=idx_from, n_proc=1)[0]
    ids_a = np.unique(pairs_NN[:,0])
    ids_b = np.unique(pairs_NN[:,1])

    def NNN_with_mid(id_mid, sub_mid):
        pairs_chosen = pairs_NN[pairs_NN[:,sub_mid]==id_mid]
        sites_ends = np.sort(pairs_chosen[:,1-sub_mid])
        pairs_NNN_chosen = np.array(list(combinations(sites_ends, 2)))
        return np.column_stack([pairs_NNN_chosen[:,0],[id_mid]*len(pairs_NNN_chosen),pairs_NNN_chosen[:,1]])

    NNN_mid_a = np.concatenate([NNN_with_mid(id_mid, 0) for id_mid in ids_a], axis=0)
    NNN_mid_b = np.concatenate([NNN_with_mid(id_mid, 1) for id_mid in ids_b], axis=0)
    return np.append(NNN_mid_a, NNN_mid_b, axis=0) 
    
def get_vij(coords, latt_vec):
    """
    coords [coords_a, coords_b] coordinates of sublatt a and b 
    latt_vec: lattice vectors
    """
    NNN_with_mid = get_NNN_with_mid(coords, latt_vec)
    coords = np.concatenate(coords, axis=0)
    d1 = coords[NNN_with_mid[:,1]]-coords[NNN_with_mid[:,0]]
    d2 = coords[NNN_with_mid[:,2]]-coords[NNN_with_mid[:,1]]
    vij = np.sign(np.cross(d1, d2)[:,-1])
    pairs = np.column_stack([NNN_with_mid[:,0],NNN_with_mid[:,2]])
    return pairs, vij

def get_Hamiltonian_soc(qd_monolayer, t2=0.03*2.7):
    H0 = qd_monolayer.get_Hamiltonian()
    
    n_site = len(qd_monolayer.coords)
    id_a, id_b = qd_monolayer._layer_inds_sublatt()[0]
    coords_a = qd_monolayer.coords[id_a[0]:id_a[1]+1]
    coords_b = qd_monolayer.coords[id_b[0]:id_b[1]+1]
    pairs, vij = get_vij([coords_a,coords_b], qd_monolayer.layer_latt_vecs[0])

    def construct_Hsoc(sz):
        Hsoc = np.zeros([n_site,n_site], dtype=complex)
        def put_value(i):
            Hsoc[pairs[i][0], pairs[i][1]] = 1j*sz*t2*vij[i]
            Hsoc[pairs[i][1], pairs[i][0]] = -1j*sz*t2*vij[i]
        [put_value(i) for i in range(len(pairs))]
        return Hsoc

    Hsoc_up = H0 + construct_Hsoc(1) 
    Hsoc_dn = H0 + construct_Hsoc(-1)
    return Hsoc_up, Hsoc_dn

def diag_hamiltonian(H):
    vals = []
    vecs = []
    for i in [0, 1]:
        val, vec, info = zheev(H[i], 1)
        if info:
            raise ValueError('lapack in scipy zheev failed!')
        vals.append(val)
        vecs.append(vec)
    return vals, vecs

def get_eigen_vals_vecs(qd_monolayer, t2=0.03*2.7, fname='EIGEN'):
    H = get_Hamiltonian_soc(qd_monolayer, t2)
    vals, vecs = diag_hamiltonian(H)
    np.savez_compressed(fname, vals=vals, vecs=vecs)
    with open('struct.obj' , 'wb') as f:
        pickle.dump(qd_monolayer, f)


def size2R(size):
    b = 1/np.sqrt(3)
    R = size*1.5*b/np.cos(15*np.pi/180)
    return R

def main(size=6):
    R = size2R(size)
    from tBG.molecule.structures import QuantumDotQC
    qd = QuantumDotQC()
    qd.regular_polygon(12, R, OV_orientation=15)
    qd.remove_top_layer()
    qd.add_hopping_wannier(ts=[-2.7])
    get_eigen_vals_vecs(qd)

if __name__ == '__main__':
    main()

