import numpy as np
from scipy.linalg.lapack import zheev
from tBG.molecule.optical_conductivity import occup_0K

def hubbard_hamiltonian_MF(hopping, Es_onsite, ns_up, ns_dn, U):
    n_orb = len(ns_up)
    n_dim = 2*n_orb # 2 for spin
    H = np.zeros([n_dim, n_dim])
    pairs, ts = hopping
    ## hopping
    def put_value_hopping(i):
        ind0, ind1 = pairs[i]
        H[2*ind0:2*(ind0+1),2*ind1:2*(ind1+1)] = ts[i]*np.identity(2)
        H[2*ind1:2*(ind1+1),2*ind0:2*(ind0+1)] = np.transpose(ts[i]*np.identity(2))
    [put_value_hopping(i) for i in range(len(pairs))]
    ## onsite energy 
    if Es_onsite != 0.:
        np.fill_diagonal(H, np.column_stack([Es_onsite, Es_onsite]).reshape(1,-1))    

    ## hubbard term
    inds_up = np.array([2*i for i in range(n_orb)])
    inds_dn = np.array([2*i+1 for i in range(n_orb)])
    H[inds_up,inds_up] = H[inds_up,inds_up] + U*ns_dn
    H[inds_dn,inds_dn] = H[inds_dn,inds_dn] + U*ns_up
    C = U*np.dot(ns_up, ns_dn)
    H = H -C*np.identity(n_dim) 
    return H

def diag_hamiltonian(H_mat):
    vals, vecs, info = zheev(H_mat, 1)
    if info:
        raise ValueError('lapack in scipy zheev failed!')
    return vals, vecs

def get_spin_distribution(vals, vecs):
    n_orb = int(len(vals)/2)
    chgs = np.abs(vecs)**2
    occup = occup_0K(vals, spin=2)
    inds_up = np.array([2*i for i in range(n_orb)])
    inds_dn = np.array([2*i+1 for i in range(n_orb)])


    ## deal with levels totally occupied 
    inds_f1 = np.where(occup==1.0)[0] 
    n_up_f1 = np.sum(chgs[:,inds_f1][inds_up,:], axis=1)
    n_dn_f1 = np.sum(chgs[:,inds_f1][inds_dn,:], axis=1)
    
    ## deal with levels on fermi level
    inds_ef = np.intersect1d(np.where(occup<1.0)[0], np.where(occup>0.0)[0])
    if len(inds_ef):
        f_ef = occup[inds_ef[0]]
        n_up_ef = np.sum(chgs[:,inds_ef][inds_up,:], axis=1)*f_ef
        n_dn_ef = np.sum(chgs[:,inds_ef][inds_dn,:], axis=1)*f_ef
        return n_up_f1+n_up_ef, n_dn_f1+n_dn_ef
    else:
        return n_up_f1, n_dn_f1


def calc_spin_dist(hopping, Es_onsite, n_up_in, n_dn_in, U):
    H_mat = hubbard_hamiltonian_MF(hopping, Es_onsite, n_up_in, n_dn_in, U)
    vals, vecs = diag_hamiltonian(H_mat)
    n_up_out, n_dn_out = get_spin_distribution(vals, vecs)
    return n_up_out, n_dn_out

def init_spin_dist(n_orb):
    n_ele = n_orb
    n_up = np.ones(n_orb)*0.5 
    n_down = np.ones(n_orb)*0.5
    return n_up, n_down

def is_converge(n_up0, n_dn0, n_up1, n_dn1, prec=0.001):
    max_up = max(np.abs(n_up0-n_up1))
    max_dn = max(np.abs(n_dn0-n_dn1))
    diff_max = max(max_up, max_dn)
    print(diff_max)
    if diff_max<=prec:
        return True
    else:
        return False

def calc_spin_dist_iteration(qd, U, mix=0.5, prec=0.001):
    hopping = qd.hopping
    Es_onsite = qd.Es_onsite
    n_orb = len(qd.coords)
    print('n_e:%s' % n_orb)
    n_up0, n_dn0 = init_spin_dist(n_orb)
    n_up1, n_dn1 = calc_spin_dist(hopping, Es_onsite, n_up0, n_dn0, U)
        
    while True:
        n_up_in = mix*n_up0 + (1-mix)*n_up1
        n_dn_in = mix*n_dn0 + (1-mix)*n_dn1
        n_up_out, n_dn_out = calc_spin_dist(hopping, Es_onsite, n_up_in, n_dn_in, U)
        
        if is_converge(n_up_in, n_dn_in, n_up_out, n_dn_out, prec):
            break
        else:
            n_up0, n_dn0 = n_up_in, n_dn_in
            n_up1, n_dn1 = n_up_out, n_dn_out
    return n_up_out, n_dn_out

def plot_spin_dist(qd, n_up, n_dn, scale=8000, alpha=0.5):
    qd.add_hopping_wannier(ts=[-2.8])
    spin_dist = n_up - n_dn
    spin_abs = np.abs(spin_dist)
    print(spin_abs)
    cs = {1.:'blue', -1.:'red', 0.:'white'}
    colors = [cs[i] for i in np.sign(spin_dist)] 
    Xs = qd.coords[:,0]
    Ys = qd.coords[:,1]
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    qd.plot(fig, ax, site_size=0)
    ax.scatter(Xs, Ys, spin_abs*scale, c=colors, alpha=alpha)
    plt.show()

def main(U=3):
    from tBG.molecule.structures import QuantumDotQC
    qc = QuantumDotQC()
    qc.regular_polygon(12, 12, OV_orientation=15)
    qc.add_hopping_wannier(P=0)
    n_up, n_dn = calc_spin_dist_iteration(qc, U, mix=0.5, prec=1.e-4) 
    plot_spin_dist(qc, n_up, n_dn, scale=80)
        
if __name__ == "__main__":
    main(U=1)    

    
    
