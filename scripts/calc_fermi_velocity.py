from tBG.brillouin_zones import BZHexagonal, BZMoirePattern
from tBG.periodic_structures import CommensuStruct, TBG30Approximant
from tBG.diagonalize import diag_onek_around
from tBG.scripts.get_twist_angle import twisted_angle
import numpy as np
from tBG.brillouin_zones import kpoints_line_mode_onek
from tBG.fortran.spec_func import get_ebs

def ebs_calc(vals, pmks, sigma, de):
    nk = len(vals)
    nb = len(vals[0])
    As = np.zeros([nk, nb])
    for i in range(nk):
        for j in range(nb):
            e = vals[i][j]
            Aij = get_ebs(e, vals, pmks, sigma, de)
            As[i][j] = Aij[i]
    return As

def diag_run(struct, dk=0.005, nstep=4, de=0.01, sigma=0.0001):
    """
    This function is used to calculate the band structure around one kpoint (corresponding to k_label) along the given direction 
    """
    bzs = BZMoirePattern(struct.latt_vec_bott, struct.latt_vec_top, struct.latt_vec)
    K = bzs.bz_bott.special_points()['K'][0]
    K_t = bzs.bz_top.special_points()['K'][0]
    K1_ts, K1s = bzs.all_kpts_after_1st_scattering()
    K1_t = K1_ts[0]
    K1 = K1s[0]
    #kpts = {'K':K, 'Kt':K_t, 'K1t':K1_t, 'K1':K1}
    kpts = {'K':K}
    dirts = {(1,0):'x', (0,1):'y', (1,1):'x+y',(1,-1):'x-y'}
    out = {}
    for kpt in kpts:
        k = kpts[kpt]
        for dirt in dirts:
            name = kpt+'_'+dirts[dirt]
            ks, ind = kpoints_line_mode_onek(k, dk=dk, direction=dirt, nstep=nstep)
            vals, vecs, pmks = struct.diag_kpts(ks, vec=1, pmk=1, elec_field=0.0)
            As = ebs_calc(vals, pmks, sigma, de)
            out[name] = {'kpts': ks, 'ind':ind, 'vals':vals, 'ebs':As}
    np.save('eig_for_vf.npy', [out])

def plot_ebs_band_comparison(eig_f='eig_for_vf.npy'):
    data = np.load(eig_f)[0]
    K = ['K_x', 'K_y', 'K_x+y', 'K_x-y']
    Kt = ['Kt_x', 'Kt_y', 'Kt_x+y', 'Kt_x-y']
    K1 = ['K1_x', 'K1_y', 'K1_x+y', 'K1_x-y']
    K1t = ['K1t_x', 'K1t_y', 'K1t_x+y', 'K1t_x-y']
    for k in [K, Kt, K1, K1t]:
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(2,2)
        count = 0
        for i in k:
            data_k_dirt = data[i]
            kpts = data_k_dirt['kpts']
            nk = len(kpts)
            vals = data_k_dirt['vals']
            nb = len(vals[0])
            ef = max(vals[:,int(nb/2)-1])
            vals = vals - ef
            ebs = data_k_dirt['ebs']
            for w in range(nb):
                axes[int(count/2)][count%2].plot(range(nk), vals[:,w], lw=0.1, color='black')
                axes[int(count/2)][count%2].scatter(range(nk), vals[:,w], s=ebs[:,w]*40, color='red')
                axes[int(count/2)][count%2].set_ylim(-0.5,0.5)
            count = count + 1
        plt.savefig('%s_band.pdf' % (k[0].split('_')[0]))
        plt.close() 

def calc_fermi_velocity_from_eig_f(eig_f='eig_for_vf.npy'):
    hbar = 6.582119514*10**(-16)
    data = np.load(eig_f)[0]
    K = ['K_x', 'K_y', 'K_x+y', 'K_x-y']
    Kt = ['Kt_x', 'Kt_y', 'Kt_x+y', 'Kt_x-y']
    K1 = ['K1_x', 'K1_y', 'K1_x+y', 'K1_x-y']
    K1t = ['K1t_x', 'K1t_y', 'K1t_x+y', 'K1t_x-y']
    vfs = {}
    #for k in [K, Kt, K1, K1t]:
    for k in [K]:
        vfs_K = []
        K_label = k[0].split('_')[0]
        vfs[K_label] = []
        for i in k:
            data_k_dirt = data[i]
            ind_K = data_k_dirt['ind']
            kpts = data_k_dirt['kpts']
            nk = len(kpts)
            dk = np.linalg.norm(kpts[1]-kpts[0])
            vals = data_k_dirt['vals']
            nb = len(vals[0])
            ind_vb = int(nb/2)-1
            ind_cb = ind_vb + 1
            vb0 = vals[:ind_K+1,ind_vb]
            vb1 = vals[ind_K:,ind_vb]
            cb0 = vals[:ind_K+1,ind_cb]
            cb1 = vals[ind_K:,ind_cb]
            ks = np.array(range(ind_K+1))*dk
            vf_i = []
            for band in [vb0, vb1, cb0, cb1]:
                vf = np.abs(np.polyfit(ks, band, 1)[0]/hbar * 10**(-10))
                vf_i.append(vf)
            vf_avg = np.sum(vf_i)/len(vf_i)
            vfs[K_label].append(vf_avg)
    return vfs

def vfs_calc_commensurate_struct(P=0):
    vfs = []
    vfs_min = []
    vfs_max = []
    ms = []
    ns = []
    thetas = []
    nsites = []
    mns = [[i,i+1] for i in range(1,20)]
    #mns1 = [[7,10],[5,8],[3,5],[4,7],[5,9],[8,15],[8,17],[5,11],[3,7]]
    #mns = np.concatenate([mns0, mns1])
    for i in mns:
        cs = CommensuStruct()
        cs.make_structure(i[0],i[1])
        cs.add_hopping_wannier(P=P)
        diag_run(cs, dk=0.005, nstep=4)
        vf = calc_fermi_velocity_from_eig_f()['K']
        vf_avg = np.sum(vf)/len(vf)
        vf_min = np.min(vf)
        vf_max = np.max(vf)
        ms.append(i[0])
        ns.append(i[1])
        nsites.append(cs.nsite)
        thetas.append(twisted_angle(i[0],i[1]))
        vfs.append(vf_avg)
        vfs_min.append(vf_min)
        vfs_max.append(vf_max)
    out = np.concatenate([[ms], [ns], [nsites], [thetas], [vfs], [vfs_min], [vfs_max]])
    np.savetxt('vfs_P%s.txt' % P, out.T, fmt=['%i', '%i', '%i', '%.2f', '%.4e', '%.4e', '%.4e'], \
                                      header='m n size theta vf_avg vf_min vf_max')
        
def vfs_calc_TBG30_approximant(n_bott=3, Ps=[0]):
    vfs = []
    vfs_min = []
    vfs_max = []
    for P in Ps:
        print('pressure:%s Gpa' % P)
        st = TBG30Approximant()
        st.make_structure(n_bott)
        st.add_hopping_wannier(P=P)
        diag_run(st, dk=0.005, nstep=4)
        vf = calc_fermi_velocity_from_eig_f()['K']
        vf_avg = np.sum(vf)/len(vf)
        vf_min = np.min(vf)
        vf_max = np.max(vf)
        vfs.append(vf_avg)
        vfs_min.append(vf_min)
        vfs_max.append(vf_max)
    out = np.concatenate([[Ps], [vfs], [vfs_min], [vfs_max]])
    np.savetxt('vfs.txt', out.T, fmt=['%.2f', '%.4e', '%.4e', '%.4e'], \
                                      header='P vf_avg vf_min vf_max')

def vfs_calc_commensurate_struct(m, n, Ps=[0]):
    vfs = []
    vfs_min = []
    vfs_max = []
    for P in Ps:
        print('pressure:%s Gpa' % P)
        cs = CommensuStruct()
        cs.make_structure(m,n)
        cs.add_hopping_wannier(P=P)
        diag_run(cs, dk=0.005, nstep=4)
        vf = calc_fermi_velocity_from_eig_f()['K']
        vf_avg = np.sum(vf)/len(vf)
        vf_min = np.min(vf)
        vf_max = np.max(vf)
        vfs.append(vf_avg)
        vfs_min.append(vf_min)
        vfs_max.append(vf_max)
    out = np.concatenate([[Ps], [vfs], [vfs_min], [vfs_max]])
    np.savetxt('vfs_%s_%s.txt' % (m,n), out.T, fmt=['%.2f', '%.4e', '%.4e', '%.4e'], \
                                      header='P vf_avg vf_min vf_max')

if __name__=='__main__':
    vfs_calc()
