from tBG.utils.approximant.plot_vec import coords_turn
from tBG.periodic_bilayer_graphene import rotate_on_vec
from tBG.periodic_bilayer_graphene import Structure
import numpy as np
import time

def coords_group_R(M, R_prec=0.12):
    s = Structure(M)
    coords = coords_turn(s)[:,:-1]
    rs = np.linalg.norm(coords, axis=1)
    inds = rs.argsort()
    
    inds_shell = [ [inds[0]] ]
    rs_shell = [ [rs[inds[0]]] ]
    for i in range(1, len(inds)):
        ind = inds[i]
        r = rs[ind]
        r_old = rs_shell[-1]
        if min(np.abs(np.array(r_old)-r))<=R_prec:
            rs_shell[-1].append(r)
            inds_shell[-1].append(ind)
        else:
            rs_shell.append([r])
            inds_shell.append([ind])
    return coords, np.array(inds_shell)
    
def ind_group_ByAngle(angles, ang_prec=0.01, rots=[i*30 for i in range(13)]):
    angles = np.array(angles)
    inds = range(len(angles))
    rots = np.array(rots)
    inds_rm = []
    inds_left = [i for i in inds]
    grps = []
    while len(inds_left):
        for i in inds_left:
            ang_min = min(angles[inds_left])
            ang = angles[i]-ang_min
            if min(np.abs(rots-ang))<=ang_prec:
                inds_rm.append(i)
        inds_left = [i for i in inds_left if i not in inds_rm]
        grps.append(inds_rm)
        inds_rm = []
    return grps

def coords_group_C12(M, R_prec=0.12, ang_prec=0.1):
    coords, ind_shells = coords_group_R(M, R_prec=R_prec)
    coords = coords[:,0] + 1j*coords[:,1]
    inds_c12 = []
    grps = []
    for ind_shell in ind_shells:
        grp = []
        coord_shell = coords[ind_shell]
        angs = np.angle(coord_shell)/np.pi*180
        inds = ind_group_ByAngle(angs, ang_prec=ang_prec)
        grps = grps + [[ind_shell[inds[i][j]] for j in range(len(inds[i]))] for i in range(len(inds))]
    return grps
        
def vec_std_calc_C12(eig_f, R_prec=0.12, ang_prec=0.1):
    data = np.load(eig_f)
    M = data['struct'][0].n_bottom
    inds_grp = coords_group_C12(M, R_prec=R_prec, ang_prec=ang_prec)
    nk = len(data['kpoints'])
    nval = len(data['vals'][0])
    stds = np.zeros([nk, nval])
    stds_all = np.zeros([nk, nval])
    chgs = data['chgs']
    maxs = np.zeros([nk, nval])
    for ik in range(nk):
        stds_all[ik] = np.std(chgs[ik], axis=0)
        maxs[ik] = np.max(chgs[ik], axis=0)
        for i in inds_grp:
            chgi = chgs[ik][i,:]
            stds[ik] = stds[ik] + np.std(chgi, axis=0)
    np.savez_compressed('std_c12', std=stds, std_all=stds_all, maxs=maxs)
