import numpy as np
from scipy.linalg.lapack import zheev
from tBG.molecule.eigenvals_analyzer import occup_0K, occup_TK
import copy
import os
import pickle
import json
from monty.json import jsanitize

def hamiltonian_no_hubbard(hopping, Es_onsite, n_orb):
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
    return H

def hubbard_hamiltonian_MF(H_no_Hubbard, ns_up, ns_dn, U):
    H = copy.deepcopy(H_no_Hubbard)
    n_orb = len(ns_up)
    n_dim = 2*n_orb
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

def spin_distribution_from_eigenvecs(vals, vecs, T):
    n_orb = int(len(vals)/2)
    chgs = np.abs(vecs)**2
    occup = occup_TK(vals, T=T, spin=2)
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

def total_band_energy(vals):
    occup = occup_0K(vals, spin=2)
    inds = np.where(occup>0.)[0]
    return np.dot(vals[inds], occup[inds])


def calc_spin_distribution(H_no_hubbard, n_up_in, n_dn_in, U, T):
    n_orb = len(n_up_in)
    H_mat = hubbard_hamiltonian_MF(H_no_hubbard, n_up_in, n_dn_in, U)
    vals, vecs = diag_hamiltonian(H_mat)
    e_tot = total_band_energy(vals)
    n_up_out, n_dn_out = spin_distribution_from_eigenvecs(vals, vecs, T)
    return n_up_out, n_dn_out, e_tot

def init_spin_distribution(qd, spin_sides=[1,1,1,1,1,1,1,1,1,1,1,1]):
    site_inds = site_inds_group_as_sides(qd)
    n_orb = len(qd.coords)
    chg_sides = np.full(12, 1.)
    ns_up_sides = (np.array(spin_sides)+chg_sides)/2.
    ns_dn_sides = (chg_sides-np.array(spin_sides))/2.
    ns_up = np.zeros(n_orb)
    ns_dn = np.zeros(n_orb)
    def put_value_sidei(ith_side):
        inds = site_inds[ith_side]
        ns_up[inds] = ns_up_sides[ith_side]
        ns_dn[inds] = ns_dn_sides[ith_side]
    [put_value_sidei(i) for i in range(0,12)]
    return ns_up, ns_dn

def site_inds_group_as_sides(qd):
    angles_sides = [np.array([-15+i*30, 15+i*30])*(np.pi/180) for i in range(0, 12)]
    angles = np.angle(qd.coords[:,0]+1j*qd.coords[:,1])
    def angle_adjust(angle):
        angle = angle+2*np.pi if angle<=-15*np.pi/180 else angle
        return angle
    angles = np.array([angle_adjust(i) for i in angles])

    def site_inds_in_side(ith_side):
        angle_lim = angles_sides[ith_side]
        inds = np.intersect1d(np.where(angles>angle_lim[0])[0], np.where(angles<=angle_lim[1])[0])
        return inds
    site_inds_group = np.array([site_inds_in_side(i) for i in range(0, 12)])
    return site_inds_group


def calc_diff_spin(n_up0, n_dn0, n_up1, n_dn1):
    max_up = max(np.abs(n_up0-n_up1))
    max_dn = max(np.abs(n_dn0-n_dn1))
    diff_max = max(max_up, max_dn)
    return diff_max
    
def calc_spin_dist_iteration(qd, U, spin_sides=[1,1,1,1,1,1,1,1,1,1,1,1], mix=0.8, prec=1.e-4, T=5, fout='profile'):
    hopping = qd.hopping
    Es_onsite = qd.Es_onsite
    n_orb = len(qd.coords)
    H_no_hubbard = hamiltonian_no_hubbard(hopping, Es_onsite, n_orb)
    n_up0, n_dn0 = init_spin_distribution(qd, spin_sides)
    n_up1, n_dn1, e_tot1 = calc_spin_distribution(H_no_hubbard, n_up0, n_dn0, U, T)
    with open(fout, 'a') as f:
        f.write('Going into iteration...\n')
        f.write('%4s %15s %15s %18s\n' %  ('','Diff_spin', 'Diff_etot', 'Etot'))
        
    i = 1
    while True:
        n_up_in = mix*n_up0 + (1-mix)*n_up1
        n_dn_in = mix*n_dn0 + (1-mix)*n_dn1
        n_up_out, n_dn_out, e_tot_out = calc_spin_distribution(H_no_hubbard, n_up_in, n_dn_in, U, T)
            
        diff_spin = abs(max(np.abs(n_up_in-n_dn_in))- max(np.abs((n_up_out- n_dn_out))))
        diff_etot = e_tot_out - e_tot1
        with open(fout, 'a') as f:
            f.write('%4i %15f %15f  %18f\n' % (i, diff_spin, diff_etot, e_tot_out))
        if abs(diff_etot)<=prec:
            with open(fout, 'a') as f:
                f.write('Get converged!\n\n')
            break
        else:
            n_up0, n_dn0 = n_up_in, n_dn_in
            n_up1, n_dn1 = n_up_out, n_dn_out
            e_tot1 = e_tot_out
            i = i + 1
    return n_up_out, n_dn_out, e_tot_out

def plot_spin_dist(qd, n_up, n_dn, scale=8000, alpha=0.5):
    qd.add_hopping_wannier(ts=[-2.8])
    spin_dist = n_up - n_dn
    spin_abs = np.abs(spin_dist)
    print('max_spin_moment: %s' % max(spin_abs))
    cs = {1.:'blue', -1.:'red', 0.:'white'}
    colors = [cs[i] for i in np.sign(spin_dist)] 
    Xs = qd.coords[:,0]
    Ys = qd.coords[:,1]
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    qd.plot(fig, ax, site_size=0)
    ax.scatter(Xs, Ys, spin_abs*scale, c=colors, alpha=alpha, clip_on=False)
    plt.show()

def size2R(size):
    b = 1/np.sqrt(3)
    R = size*1.5*b/np.cos(15*np.pi/180)
    return R

def main(size, U, phase):
    fout = 'size_%s/profile_U%s' % (size, U)
    from tBG.molecule.structures import QuantumDotQC
    spin_sides = spin_sides_distribution(phase)
    R = size2R(size)
    qc = QuantumDotQC()
    qc.regular_polygon(12, R, OV_orientation=15)
    with open(fout, 'a') as f:
        f.write('num_electron: %s\n' % len(qc.coords))
        f.write('phase: %s\n' % phase)
    qc.add_hopping_wannier(P=0)
    n_up, n_dn, e_tot = calc_spin_dist_iteration(qc, U, spin_sides, mix=0.01, prec=1.e-4, T=5, fout=fout) 
    with open('size_%s/struct.obj' % size, 'wb') as f:
        pickle.dump(qc, f)
    out = {'n_up':n_up, 'n_dn':n_dn, 'e_tot':e_tot}
    with open('size_%s/U_%s_initPhase_%s.json' % (size, U, phase), 'w') as f:
        json.dump(jsanitize(out), f) 

    #plot_spin_dist(qc, n_up, n_dn, scale=3000)

def spin_sides_distribution(phase='NM'):
    if phase == 'NM':
        return np.zeros(12)
    elif phase == 'AF-AF':
        return [1,1,-1,-1,1,1,-1,-1,1,1,-1,-1]
    elif phase == 'FM-FM-FM':
        return [1,1,1,1,1,1,1,1,1,1,1,1]
    elif phase == 'FM-FM-AF':
        return [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1]
        
if __name__ == "__main__":
    Rs = [2]
    Us = [3]
    sizes = [2]
    phases = ['NM', 'AF-AF','FM-FM-FM','FM-FM-AF']
    for size in sizes:
        os.mkdir('size_%s' % size)
        for U in Us:
            for phase in phases:
                main(size, U, phase)    


    
    
