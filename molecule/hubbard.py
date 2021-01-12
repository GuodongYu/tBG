import numpy as np
from scipy.linalg.lapack import zheev
from tBG.molecule.eigenvals_analyzer import occup_0K, occup_TK
import copy
import os
import pickle
import json
from monty.json import jsanitize

def hubbard_hamiltonian_MF(H_no_Hubbard, ns_up, ns_dn, U):
    """
    construct hamiltonian for both spin up and down parts
    """ 
    n_orb = H_no_Hubbard.shape[0]
    ns = [ns_up, ns_dn]
    H = []
    for i in [0, 1]:
        Hi = copy.deepcopy(H_no_Hubbard)
        Hi = Hi + U*ns[1-i]*np.identity(n_orb)
        H.append(Hi)
    return H

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

def spin_distribution_from_eigenvecs(vals, vecs, T):
    n_orb = len(vals[0])
    chgs = [np.abs(vecs[i])**2 for i in [0,1]]
    vals_merge = np.concatenate(vals, axis=0)
    ids_arg = np.argsort(vals_merge) 
    occup = occup_TK(vals_merge[ids_arg], T=T, spin=2)
    ids_up = np.where(ids_arg<n_orb)[0]
    ids_dn = np.where(ids_arg>=n_orb)[0]
    ns_up = np.sum([occup[ids_up][i]*chgs[0][:,i] for i in range(n_orb)], axis=0)
    ns_dn = np.sum([occup[ids_dn][i]*chgs[1][:,i] for i in range(n_orb)], axis=0)
    return ns_up, ns_dn

def total_band_energy(vals, T, Ec):
    """
    Ec is the energy from the constant term U np.dot(ns_up, ns_dn)
    """
    vals = np.sort(np.concatenate(vals))
    occup = occup_TK(vals, T=T, spin=2)
    return np.dot(vals, occup)-Ec


def calc_spin_distribution(H_no_hubbard, n_up_in, n_dn_in, U, T):
    n_orb = len(n_up_in)
    H_mat = hubbard_hamiltonian_MF(H_no_hubbard, n_up_in, n_dn_in, U)
    vals, vecs = diag_hamiltonian(H_mat)
    n_up_out, n_dn_out = spin_distribution_from_eigenvecs(vals, vecs, T)
    Ec = U*np.dot(n_up_out, n_dn_out)
    e_tot = total_band_energy(vals, T, Ec)
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
    

def calc_spin_dist_iteration(qd, U, spin_sides=[1,1,1,1,1,1,1,1,1,1,1,1], mix0=0.8, mix1=0.1, prec=1.e-4, nmax=700, T=5, fout='profile'):
    if mix0+mix1>1.0:
        raise ValueError('mix0 + mix1 should be less than 1.0')
    ###
    with open(fout, 'a') as f:
        f.write('Going into iteration...\n')
        f.write('%4s %15s %18s\n' %  ('', 'Diff_etot', 'Etot'))
   
    H_no_hubbard = qd.get_Hamiltonian() 
    def init_prepare():    
        n_up_in0, n_dn_in0 = init_spin_distribution(qd, spin_sides)
        n_up_in1, n_dn_in1, e_tot_in1 = calc_spin_distribution(H_no_hubbard, n_up_in0, n_dn_in0, U, T)
        with open(fout, 'a') as f:
            f.write('%4i %15f  %18f\n' % (0, e_tot_in1, e_tot_in1))

        n_up_in2, n_dn_in2, e_tot_in2 = calc_spin_distribution(H_no_hubbard, n_up_in1, n_dn_in1, U, T)
        with open(fout, 'a') as f:
            f.write('%4i %15f  %18f\n' % (1, e_tot_in2-e_tot_in1, e_tot_in2))
        return n_up_in0, n_up_in1, n_up_in2, n_dn_in0, n_dn_in1, n_dn_in2, e_tot_in2

    n_up_in0, n_up_in1, n_up_in2, n_dn_in0, n_dn_in1, n_dn_in2, e_tot_in = init_prepare()
    
    i = 2
    while True:
        n_up_in = mix0*n_up_in0 + mix1*n_up_in1 + (1-mix0-mix1)*n_up_in2
        n_dn_in = mix0*n_dn_in0 + mix1*n_dn_in1 + (1-mix0-mix1)*n_dn_in2
        n_up_out, n_dn_out, e_tot_out = calc_spin_distribution(H_no_hubbard, n_up_in, n_dn_in, U, T)
        with open(fout, 'a') as f:
            f.write('%4i %15f  %18f\n' % (i, e_tot_out-e_tot_in, e_tot_out))
        if abs(e_tot_out-e_tot_in)<=prec:
            with open(fout, 'a') as f:
                f.write('Got converged!\n')
            converged = True
            break
        else:
            n_up_in0, n_up_in1, n_up_in2 = n_up_in1, n_up_in2, n_up_out
            n_dn_in0, n_dn_in1, n_dn_in2 = n_dn_in1, n_dn_in2, n_dn_out
            e_tot_in = e_tot_out
            i = i + 1
            if i>nmax:
                with open(fout, 'a') as f:
                    f.write('EXIT Warning: exceed the max step number %s!!\n\n' % nmax)
                converged = False
                break
    return n_up_out, n_dn_out, e_tot_out, converged

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

def main(size, U, phase, P=0):
    fout = 'size_%s/profile_U%s' % (size, U)
    from tBG.molecule.structures import QuantumDotQC
    spin_sides = spin_sides_distribution(phase)
    R = size2R(size)
    qc = QuantumDotQC()
    qc.regular_polygon(12, R, OV_orientation=15)
    with open(fout, 'a') as f:
        f.write('num_electron: %s\n' % len(qc.coords))
        f.write('phase: %s\n' % phase)
    qc.add_hopping_wannier(P=P)
    n_up, n_dn, e_tot, converged = calc_spin_dist_iteration(qc, U, spin_sides, mix0=0.001,mix1=0.001, prec=1.e-4, T=10, nmax=500, fout=fout) 
    with open('size_%s/struct.obj' % size, 'wb') as f:
        pickle.dump(qc, f)
    out = {'n_up':n_up, 'n_dn':n_dn, 'e_tot':e_tot, 'converged':converged}
    with open('size_%s/U_%s/initPhase_%s.json' % (size, U, phase), 'w') as f:
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
    P = 20
    Us = [2, 3]
    sizes = [9]
    phases = ['NM', 'AF-AF', 'FM-FM-FM', 'FM-FM-AF']
    for size in sizes:
        if not os.path.isdir('size_%s'%size):
            os.mkdir('size_%s' % size)
        for U in Us:
            if not os.path.isdir('size_%s/U_%s'%(size, U)):
                os.mkdir('size_%s/U_%s' % (size, U))
            for phase in phases:
                main(size, U, phase, P=P)    
