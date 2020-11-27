import numpy as np
from tBG.molecule.optical_conductivity import occup_0K
from tBG.molecule.symmetry_analyzer import classify_eigenstates_SymmeOp_and_irrep
from tBG.molecule.point_group import PointGroup
import pickle

def combine_all_vals(vals_irrep_symmeop):
    vals = np.array([vals_irrep_symmeop[i][j] for i in vals_irrep_symmeop for j in vals_irrep_symmeop[i]])
    vals = np.concatenate(vals)
    vals = np.sort(vals)
    print(vals)
    return vals

def get_vb_top_and_cb_bott(vals):
    occups = occup_0K(vals)
    ind_max_occp2 = max(np.where(occups==2)[0])
    ind_min_occp0 = max(np.where(occups==0)[0])
    if ind_max_occp2+1 == ind_min_occp0:
        # semiconductor
        val_mid = (vals[ind_max_occp2]+vals[ind_min_occp0])/2
        return val_mid, val_mid
    else:
        # metal
        ef = vals[ind_max_occp2+1]
        vbm = vals[ind_max_occp2]
        cbm = vals[ind_min_occp0]
        return (vbm+ef)/2, (cbm+ef)/2
    
def group_vb_cb(vals_cut, vb_top, cb_bott):
    vals_cut = np.array(vals_cut)
    inds_vb = np.where(vals_cut<vb_top)[0]
    inds_cb = np.where(vals_cut>cb_bott)[0]
    if len(inds_cb)+len(inds_vb)<len(vals_cut):
        inds_ef = np.arange(max(inds_vb)+1, min(inds_cb))
        inds_vb = np.append(inds_vb, inds_ef)
        inds_cb = np.append(inds_ef, inds_cb)
    return vals_cut[inds_vb], vals_cut[inds_cb]

def count_transitions(vbs, cbs):
    trans = [cbs-vb for vb in vbs]
    trans = np.concatenate(trans)
    trans = np.sort(trans[trans>1.e-8])
    return trans

def collect_transitions(struct_f='struct.obj', eigen_f='EIGEN.npz', vector='x'):
    if vector in ['x','y']:
        symmeop_label = 'sigma_x'
    elif vector in ['x+iy', 'x-iy']:
        symmeop_label = 'Cn1'
    vals_irrep_symmeop, _ = classify_eigenstates_SymmeOp_and_irrep(symmeop_label, struct_f, eigen_f)
    vals_expand = combine_all_vals(vals_irrep_symmeop)
    vb_top, cb_bott = get_vb_top_and_cb_bott(vals_expand) 
    with open(struct_f, 'rb') as f:
        qd = pickle.load(f)
    pg = PointGroup(qd.point_group)
    rule_irrep, rule_symmeop = pg.transition_rule_elec_dipole(vector)
    transitions = {}
    for angle0 in rule_symmeop:
        angle1 = rule_symmeop[angle0]
        for irrep0 in rule_irrep:
            for irrep1 in rule_irrep[irrep0]:
                try:
                    vals0 = vals_irrep_symmeop[angle0][irrep0]
                    vals1 = vals_irrep_symmeop[angle1][irrep1]
                except:
                    pass
                else:
                    vals0_vb, _ = group_vb_cb(vals0, vb_top, cb_bott)
                    _, vals1_cb = group_vb_cb(vals1, vb_top, cb_bott)
                    transitions['%s->%s&%s->%s'%(angle0, angle1, irrep0, irrep1)] = count_transitions(vals0_vb,vals1_cb)
    return transitions
t = 2.8
def plot_optical_cond_with_transition_rule_labeled(sigma_f='sigma.txt', struct_f='struct.obj', eigen_f='EIGEN.npz', vector='x'):
    sigma = np.loadtxt(sigma_f)
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(sigma[:,0], sigma[:,1], label='$\sigma_{xx}$', color='black')
    trans = collect_transitions(struct_f, eigen_f, vector)
    for tran in trans:
        xs = trans[tran]
        ys = [0]*len(xs)
        ax.scatter(xs, ys)
    plt.xlim(0, 3*t)
    plt.show()
    
