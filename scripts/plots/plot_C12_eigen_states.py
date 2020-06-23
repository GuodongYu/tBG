from tBG.utils import cart2frac
import numpy as np
from tBG.brillouin_zones import BZHexagonal
from tBG.periodic_structures import TBG30Approximant

########## C12-fold symmetry analysis for structure #####
def coords_turn(struct):
    coords = struct.coords
    latt = struct.latt_vec[0:2][:,0:2]
    latt_2 = latt/2 
    coords_new = []
    for coord in coords:
        i, j = np.floor(cart2frac(coord[0:2], latt_2))
        if i == -1:
            i = 0
        if j == -1:
            j = 0
        if i == 2:
            i = 1
        if j == 2:
            j = 1
        shift = np.concatenate((-i*latt[0]-j*latt[1],[0.]), axis=0)
        coords_new.append(coord+shift)
    return np.array(coords_new)[:,0:2]

def _coords_group_by_R(coords, prec_r=0.12):
    rs = np.linalg.norm(coords, axis=1)
    inds = rs.argsort()
   
    inds_group = [[inds[0]]] 
    for ind in inds[1:]:
        r = rs[ind]
        r_avg = np.average(rs[inds_group[-1]])   
        if np.abs(r - r_avg)<=prec_r:
           inds_group[-1].append(ind)
        else:
            inds_group.append([ind])
    return inds_group
    
def _ind_group_by_Angle(angles, prec_ang=0.1):
    angles = np.array(angles)
    inds = range(len(angles))
    rots = np.array([i*30 for i in range(12)])
    inds_rm = []
    inds_left = [i for i in inds]
    grps = []
    while len(inds_left):
        for i in inds_left:
            ang_min = min(angles[inds_left])
            ang = angles[i]-ang_min
            if min(np.abs(rots-ang))<= prec_ang:
                inds_rm.append(i)
        inds_left = [i for i in inds_left if i not in inds_rm]
        grps.append(inds_rm)
        inds_rm = []
    return grps

def coords_group_by_C12(struct, prec_r=0.12, prec_ang=0.1):
    coords = coords_turn(struct)
    ind_shells = _coords_group_by_R(coords, prec_r)
    coords_img = coords[:,0] + 1j*coords[:,1]
    inds_c12 = []
    grps = []
    for ind_shell in ind_shells:
        grp = []
        coord_shell = coords_img[ind_shell]
        angs = np.angle(coord_shell)/np.pi*180
        inds = _ind_group_by_Angle(angs, prec_ang=prec_ang)
        grps = grps + [[ind_shell[inds[i][j]] for j in range(len(inds[i]))] for i in range(len(inds))]
    return coords, [np.array(i) for i in grps]

#################################################
def diag_run(n_bott=15, P=0, kpts=[[0.,0.]], prec_r=0.12, prec_ang=0.1, elec_field=0.0):
    kpts = np.array(kpts)
    st = TBG30Approximant()
    st.make_structure(n_bott)
    coords, inds_grp = coords_group_by_C12(st, prec_r, prec_ang)
    layer_nsites = st.layer_nsites
    st.add_hopping_wannier(P=P)
    vals, vecs, pmk = st.diag_kpts(kpts, vec=1, pmk=0, elec_field=elec_field)
    chgs = np.square(np.abs(vecs))
    stds = vec_std_calc_C12(vals, chgs, inds_grp)
    np.savez_compressed('EIGEN_C12', kpts=kpts, vals=vals, \
                   chgs=chgs, stds=stds, coords=coords, layer_nsites=layer_nsites)

def plot(eig_f='EIGEN_C12.npz', prec_std=3.e-2, ef=0., elim=[-2,2]):
    data = np.load(eig_f)
    coords = data['coords']
    layer_nsites = data['layer_nsites']
    ind_bott = layer_nsites[0]
    vals = data['vals'] 
    vals = vals - ef
    inds_e0 = np.where(vals>=elim[0])
    inds_e1 = np.where(vals<=elim[1])
    inds_std = np.where(data['stds']<=prec_std)
    inds_e0 = set([(inds_e0[0][i], inds_e0[1][i]) for i in range(len(inds_e0[0]))])
    inds_e1 = set([(inds_e1[0][i], inds_e1[1][i]) for i in range(len(inds_e1[0]))])
    inds_std = set([(inds_std[0][i], inds_std[1][i]) for i in range(len(inds_std[0]))])
    inds = inds_e0 & inds_e1 & inds_std
    for i in inds:
        ik = i[0]
        ib = i[1]
        from matplotlib import pyplot as plt
        chg = data['chgs'][ik][:,ib]
        e = vals[ik][ib]
        if e<=elim[0]:
            continue
        elif e>=elim[1]:
            continue
        plt.scatter(coords[:ind_bott,0], coords[:ind_bott,1], s = chg[:ind_bott]*20000, color='blue',linewidths=0,alpha=0.5)
        plt.scatter(coords[ind_bott:,0], coords[ind_bott:,1], s = chg[ind_bott:]*20000, color='red',linewidths=0,alpha=0.5)
        plt.title('ene=%.3f eV' % e)
        plt.axis('equal')
        plt.savefig('ik-%s_ib-%s.png' % (ik, ib))
        plt.close()
############################################################################

def vec_std_calc_C12(vals, chgs, inds_grp):
    nk = len(vals)
    nval = len(vals[0])
    stds = np.zeros([nk, nval])
    #stds_all = np.zeros([nk, nval])
    #maxs = np.zeros([nk, nval])
    for ik in range(nk):
        #stds_all[ik] = np.std(chgs[ik], axis=0)
        #maxs[ik] = np.max(chgs[ik], axis=0)
        for i in inds_grp:
            chgi = chgs[ik][i,:]
            stds[ik] = stds[ik] + np.std(chgi, axis=0)
    return stds

def main(n_bott=3, P=0, kpts=[[0., 0.]], ef=0., prec_std=3.e-2, elim=[-2,2], elec_field=0.0):
    diag_run(n_bott=n_bott, P=P, kpts=kpts, prec_r=0.12, prec_ang=0.1, elec_field=elec_field)
    plot(eig_f='EIGEN_C12.npz', prec_std=prec_std, ef=ef, elim=elim)

if __name__=='__main__':
    diag_run(n_bott=3, P=0, kpts=[[0.,0.]], prec_r=0.12, prec_ang=0.1)
    plot(eig_f='EIGEN_C12.npz', prec_std=3.e-2, ef=0.)
