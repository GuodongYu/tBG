import pickle
import numpy as np
from tBG.molecule.point_group import PointGroup
from tBG.molecule.optical_conductivity import get_vbm_cbm, occup_0K
   
def _parse(struct_f, eigen_f):
    with open(struct_f, 'rb') as f:
        qd = pickle.load(f)
    eigen = np.load(eigen_f, allow_pickle=True)
    vals = eigen['vals']
    vecs = eigen['vecs']
    return qd, vals, vecs

class FullSpace:
    def __init__(self, qd):
        """
        the full space means all the pz orbitals for the tight-binding model
        """
        self.qd = qd    
        self.rep = self._decompose_as_irreps()

    def _decompose_as_irreps(self):
        """
        decompose the redicible representation to irreps
        """
        ops = self.get_mat_of_operators()
        chi = [np.trace(i[0]) for i in ops]
        pg = PointGroup(self.qd.point_group)
        return pg.rrep2irrep(chi)

    def get_mat_of_operators(self):
        """
        get the reducible representation matrices grouped by classes
        return: 
            for C3v, [[E], [C3_1,C3_2], [three sigma_v]]
            for C6v, [[E], [C6_1,C6_5], [C6_2,C6_4],[C6_3], [three sigma_v], [three sigma_d]]
            for D6h, [[E], [S12_1, S12_11], [C6_1, C6_5], [S12_3, S12_9], [C6_2, C6_4], [S12_5, S12_7], [C6_3], [six C2_1], [six sigma_d]]
        """
        qd = self.qd
        ops = qd.symmetry_operations()
        pg = qd.point_group
    
        E = np.identity(len(qd.coords))
        if pg=='C3v':
            C3_1_2 = [qd.symmetryop2matrix(i) for i in ops['Cns']]
            sigma_v = [qd.symmetryop2matrix(i) for i in ops['sigma_vs']]
            return [[E], C3_1_2, sigma_v]
        elif pg == 'C6v':
            C6_1_5 = [qd.symmetryop2matrix(ops['Cns'][i]) for i in [0,-1]]
            C6_2_4 = [qd.symmetryop2matrix(ops['Cns'][i]) for i in [1,-2]]
            C6_3 = [qd.symmetryop2matrix(ops['Cns'][2])]
            sigma_v = [qd.symmetryop2matrix(ops['sigma_vs'][i]) for i in [1, 3, 5]]
            sigma_d = [qd.symmetryop2matrix(ops['sigma_vs'][i]) for i in [0, 2, 4]]
            return [[E], C6_1_5, C6_2_4, C6_3, sigma_v, sigma_d]
        elif pg == 'D6d':
            S12_1_11 = [qd.symmetryop2matrix(ops['S2n_odds'][i]) for i in [0,-1]]
            S12_3_9 = [qd.symmetryop2matrix(ops['S2n_odds'][i]) for i in [1,-2]]
            S12_5_7 = [qd.symmetryop2matrix(ops['S2n_odds'][i]) for i in [2,-3]]
            C6_1_5 = [qd.symmetryop2matrix(ops['Cns'][i]) for i in [0,-1]]
            C6_2_4 = [qd.symmetryop2matrix(ops['Cns'][i]) for i in [1,-2]]
            C6_3 = [qd.symmetryop2matrix(ops['Cns'][2])]
            C2 = [qd.symmetryop2matrix(i) for i in ops['C2s_QC']]
            sigma_d = [qd.symmetryop2matrix(i) for i in ops['sigma_vs']]
            return [[E], S12_1_11, C6_1_5, S12_3_9, C6_2_4, S12_5_7, C6_3, C2, sigma_d] 

    def get_projection_operators(self):
        """
        get the projection operators for all irreps
        """
        ops = self.get_mats_of_operators()
        pg = PointGroup(self.qd.point_group)
        return pg.projection_operators(ops)



class SubSpace:
    def __init__(self, qd, vecs):
        self.qd = qd
        self.basis = vecs
        self.rep = self._decompose_as_irreps()

    def _decompose_as_irreps(self):
        _full_space = FullSpace(qd)
        ops = _full_space.get_mats_of_operators()
        ops_new = [[self.operater_mat(j) for j in i] for i in ops]
        chi = [np.trace(i[0]) for i in ops_new]
        pg = PointGroup(self.qd.point_group)
        return pg.rrep2irrep(chi)

    def operater_mat(self, op_mat_full):
        left_vec = np.transpose(self.basis).conj()
        right_vec = np.matmul(op_mat_full, self.basis)
        op_mat_sub = np.matmul(left_vec, right_vec)
        return op_mat_sub

    def get_irrep_basis(self):
        

        



def _vals_vecs_irrep_group(vals, vecs, irreps):
    irreps = np.array([i[0] for i in irreps])
    irreps_label = set(irreps)
    inds = {i:np.where(irreps==i)[0] for i in irreps_label}
    vals_irrep = {i:vals[inds[i]] for i in irreps_label}
    vecs_irrep = {i:vecs[:,inds[i]] for i in irreps_label}
    return vals_irrep, vecs_irrep
    
def classify_eigen_states_irrep(struct_f='struct.obj', eigen_f='EIGEN.npz'):
    """
    struct: the picked struct file
    eigen_f: the eigen file(npz) including vals and vecs for eigenvalues and eigenvectors
    
    Note: for some subspace with the same eigenvalue, the eigen_vectors has the possibility to mix the basis functions
          of different ireducible representations. For this case, this function will pick up the basis function of corresponding
          irreps with the projection operators acting on the eigenvectors.
    """

    qd, vals, vecs = _parse(struct_f, eigen_f)   
    pg = PointGroup(qd.point_group) 
    ndim = len(vals)
    proj_ops = get_projection_operators(qd)
    ops_class_group = get_representation_matrices(qd)
    irreps_label = proj_ops.keys()

    def pre_treatment():
        norms = {}
        for irrep in irreps_label:
            basis = np.matmul(proj_ops[irrep], vecs)
            norms[irrep] = np.linalg.norm(basis, axis=0)
        return norms
    norms = pre_treatment()

    def check_irrep_comp():
        irreps = [] # save all irrep conmpont for all eigen vectors
        for i in range(ndim):
            include_irrep = []
            for irrep in proj_ops:
                if norms[irrep][i]>1.e-8:
                    include_irrep.append(irrep)
            irreps.append(include_irrep)
        return irreps
    irrep_comps = np.array(check_irrep_comp())

    # pick up the indices with mixed irrep components
    irrep_nount = np.array([len(i) for i in irrep_comps]) 
    inds_mixed = np.where(irrep_nount>1)[0]
    if not len(inds_mixed):
        return _vals_vecs_irrep_group(vals, vecs, irrep_comps)

    def group_inds_with_mixed_irrep(prec=1.e-4):
        ind0 = inds_mixed[0]
        out = [[ind0]]
        val = vals[ind0]
        for ind in inds_mixed[1:]:
            val_i = vals[ind]
            if np.abs(val_i-val)<prec:
                out[-1].append(ind)
            else:
                out.append([ind])
            val = vals[ind]
        return out
    inds_mixed_group = group_inds_with_mixed_irrep()

    def get_new_rrep_representation_chi(basis_funcs):
        op_array = np.array([i[0] for i in ops_class_group])
        op_mat_new = np.array([np.matmul(np.transpose(basis_funcs).conj(), np.matmul(i,basis_funcs)) for i in op_array])
        chis = [np.trace(i) for i in op_mat_new]
        return chis

    def deal_with_mixed_irrep():
        for i in range(len(inds_mixed_group)):
            inds = inds_mixed_group[i]
            vecs_mixed = vecs[:,inds[0]:inds[-1]+1]
            chis = get_new_rrep_representation_chi(vecs_mixed)
            decompose_comp = pg.rrep2irrep(chis)
            print(decompose_comp)
            ind_val = 0
            for irrep in decompose_comp:
                n_time = decompose_comp[irrep]
                if not n_time:
                    continue
                if irrep[0] in ['A','B']:
                    n_basis = n_time
                elif irrep[0] == 'E':
                    n_basis = n_time*2
                vecs_picked = np.matmul(proj_ops[irrep], vecs_mixed[:,ind_val:ind_val+n_basis])

                vecs_orth, r = np.linalg.qr(vecs_picked)
                vecs[:,inds[ind_val]:inds[ind_val]+n_basis] = vecs_orth
                irrep_comps[inds[ind_val]:inds[ind_val]+n_basis] = [[irrep]]
                ind_val = ind_val + n_basis
    deal_with_mixed_irrep()
    return _vals_vecs_irrep_group(vals, vecs,irrep_comps)

        
    
def _classify_eigenstates_SymmeOp_and_irrep(SymmeOp_mat, SymmeOp_vals_1D_irrep, \
                                           struct_f='struct.obj', eigen_f='EIGEN.npz'):
    """
    classify the Hamiltonian eigenstates of the system with irrep and the eigen value of symmetry operation (SymmeOp_mat)
    Just for C3v, C6v and D6d poing groups (namely graphene quasicrystal dots)
    ** NOTE: symmetry operation is limited to sigma_x and Cn1 only **
    """
    def group_E_irrep(vals):
        out = [[0]]
        val = vals[0]
        for i in range(1, len(vals)):
            vali = vals[i]
            if np.abs(val-vali)<1.e-8:
                out[-1].append(i)
            else:
                out.append([i])
            val = vali
        return out
    vals_irrep, vecs_irrep = classify_eigen_states_irrep(struct_f='struct.obj', eigen_f='EIGEN.npz')

    vals_H_symm = {}
    vecs_H_symm = {}

    for irrep in vals_irrep:
        if irrep in SymmeOp_vals_1D_irrep:
            mu = SymmeOp_vals_1D_irrep[irrep]
            angle = int(round(np.angle(mu)*180/np.pi))
            if angle not in vals_H_symm:
                vals_H_symm[angle] = {}
            if angle not in vecs_H_symm:
                vecs_H_symm[angle] = {}
            vals_H_symm[angle][irrep] = vals_irrep[irrep]
            vecs_H_symm[angle][irrep] = vecs_irrep[irrep]
        elif irrep[0] == 'E':
            inds_group = group_E_irrep(vals_irrep[irrep])
            def put_E_value(inds):
                vals = vals_irrep[irrep][inds]
                vecs = vecs_irrep[irrep][:,inds]
                SymmeOp_new = np.matmul(np.transpose(vecs).conj(), np.matmul(SymmeOp_mat, vecs))
                vals_E, vecs_E = np.linalg.eig(SymmeOp_new)
                for i in range(len(vals_E)):
                    mu = vals_E[i]
                    angle = np.angle(mu)*180/np.pi
                    angle = int(round(angle))
                    if angle not in vals_H_symm:
                        vals_H_symm[angle] = {}
                    if angle not in vecs_H_symm:
                        vecs_H_symm[angle] = {}
                    vec = np.sum(vecs_E[:,i]*vecs, axis=1).reshape(-1,1)
                    try:
                        vecs_H_symm[angle][irrep] = np.append(vecs_H_symm[angle][irrep], vec, axis=1)
                        vals_H_symm[angle][irrep].append(vals[i])
                    except:
                        vecs_H_symm[angle][irrep] = vec
                        vals_H_symm[angle][irrep] = [vals[i]]
            [put_E_value(inds) for inds in inds_group]
    return vals_H_symm, vecs_H_symm

def classify_eigenstates_SymmeOp_and_irrep(SymmeOp_label='sigma_x', struct_f='struct.obj', eigen_f='EIGEN.npz'):
    """
    classify the Hamiltonian eigenstates and eigenvalues according to ireducible representation 
    and eigenvalues of symmetry operation matrix. 

    inputs:
        SymmeOp_label: sigma_x or Cn1 
        struct_f: the dumped struct file by pickle
        eigen_f: the eigen file saveing the eigenvalues and eigenvectors

    return:
        vals and vecs classified by irrep and symmetry operation 
        such as: vals = {rou1:{'A1':[H_eigenvalues],'A2':[H_eigenvalues],...}, rou2:{'B1':[H_eigenvalues],'B2':[H_eigenvalues], ....}}
            Note: rou1 and rou2 are not the direct eigen values of the symmetry operation, they are the angle in degree.
                  the real eigen value of the SymmeOp coresponding to rou is np.exp(i*rou*np.pi/180)
            vecs has the same structure as vals
    """
    with open(struct_f, 'rb') as f:
        qd = pickle.load(f)
    pg = qd.point_group
    if SymmeOp_label == 'sigma_x':
        SymmeOp_mat = get_representation_matrices(qd)[-1][0]
        if pg == 'C3v':
            SymmeOp_vals_1D_irrep = {'A1':1, 'A2':-1}
        elif pg in ['C6v','D6d']:
            SymmeOp_vals_1D_irrep = {'A1':1, 'A2':-1, 'B1':-1, 'B2':1}
    elif SymmeOp_label == 'Cn1':
        if pg == 'C3v':
            SymmeOp_mat = get_representation_matrices(qd)[1][0]
            SymmeOp_vals_1D_irrep = {'A1':1, 'A2':1}
        elif pg == 'C6v':
            SymmeOp_mat = get_representation_matrices(qd)[1][0]
            SymmeOp_vals_1D_irrep = {'A1':1, 'A2':1, 'B1':-1, 'B2':-1}
        elif pg == 'D6d':
            SymmeOp_mat = get_representation_matrices(qd)[2][0]
            SymmeOp_vals_1D_irrep = {'A1':1, 'A2':1, 'B1':1, 'B2':1}
    return _classify_eigenstates_SymmeOp_and_irrep(SymmeOp_mat, SymmeOp_vals_1D_irrep, \
                                           struct_f='struct.obj', eigen_f='EIGEN.npz')         
    
def plot_levels_with_SymmpOp_evalues(t=2.8, SymmeOp_label='sigma_x', ylim=None, struct_f='struct.obj', eigen_f='EIGEN.npz'):
    vals, vecs = classify_eigenstates_SymmeOp_and_irrep(SymmeOp_label=SymmeOp_label, struct_f=struct_f, eigen_f=eigen_f)
    eig = np.load(eigen_f)
    vals_expand = eig['vals']
    from tBG.molecule.optical_conductivity import get_vbm_cbm
    vbm, cbm = get_vbm_cbm(vals_expand)
    symmpOP_values = sorted(vals.keys())
    xs = [-0.15, -0.05]
    dx = xs[1]-xs[0]
    shift = 0.05
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    for rou in symmpOP_values:
        x0 = xs[1]+shift
        x1 = x0 + dx
        xs = [x0, x1]
        ax.text(np.sum(xs)/2,0, '$\\theta=%s^{\degree}$' % rou, color='red')
        i = 0
        for irrep in vals[rou]:
            for e in vals[rou][irrep]:
                e = e/t
                ax.plot(xs, [e, e], color='black')
                ax.text(x0+i*0.02, e, irrep, color='blue')
            i = i +1
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.axhline((vbm+cbm)/2, ls='dashed', color='blue')
    plt.show()

def calcu_Lz_averg(SymmeOp_label='Cn1', struct_f='struct.obj', eigen_f='EIGEN.npz'):
    with open(struct_f, 'rb') as f:
        qd = pickle.load(f)
    Lz_mat = qd.get_Lz_mat()
    vals, vecs = classify_eigenstates_SymmeOp_and_irrep(SymmeOp_label=SymmeOp_label, struct_f='struct.obj', eigen_f='EIGEN.npz')
    lzs = {}
    for rou in vals:
        lzs[rou] = {}
        for irrep in vals[rou]:
            vecs_rou_irrep = vecs[rou][irrep]
            vecs_new = np.matmul(Lz_mat, vecs_rou_irrep)
            lzs_rou_irrep = [np.dot(vecs_rou_irrep[:,i].conj(), vecs_new[:,i])  for i in range(len(vecs_new[0]))]
            lzs[rou][irrep] = np.array(lzs_rou_irrep).real
    return vals, lzs, vecs

def calc_transition_matrix(vector='x', struct_f='struct.obj', eigen_f='EIGEN.npz'):
    hbar_eVs = 6.582119514 *10**(-16)
    with open(struct_f, 'rb') as f:
        qc = pickle.load(f)
    Jx, Jy = qc.get_current_mat()
    if vector == 'x':
        symmeop_label = 'sigma_x'
        J = Jx
    elif vector == 'y':
        symmeop_label = 'sigma_x'
        J = Jy
    elif vector == 'x+iy':
        symmeop_label = 'Cn1'
        J = Jx+1j*Jy
    elif vector == 'x-iy':
        symmeop_label = 'Cn1'
        J = Jx-1j*Jy
    vals, vecs = classify_eigenstates_SymmeOp_and_irrep(symmeop_label, struct_f, eigen_f)

    def tran_mat(angle0, irrep0, angle1, irrep1):
        vecs0 = vecs[angle0][irrep0]
        vecs1 = vecs[angle1][irrep1]
        vecs_left = np.transpose(vecs0).conj()
        vecs_right = np.matmul(J, vecs1)
        J_mat = np.matmul(vecs_left, vecs_right)*hbar_eVs
        J_square =  J_mat.conj()*J_mat
        return J_square.real
    
    angles = sorted(vals.keys())
    labels = [sorted(list(vals[i].keys())) for i in angles]
    n_levls_sub = np.array([[len(vals[angles[i]][j]) for j in labels[i]] for i in range(len(labels))])
    n_levls = np.array([np.sum(i) for i in n_levls_sub])
    ndim = np.sum(n_levls)
    J_square = np.zeros([ndim, ndim])
    for i in range(len(angles)):
        angle0 = angles[i]
        for irrep0 in labels[i]:
            ind_irrep0 = labels[i].index(irrep0)
            for j in range(i, len(angles)):
                angle1 = angles[j]
                for irrep1 in labels[j]:
                    ind_irrep1 = labels[j].index(irrep1)
 
                    ind0_start = np.sum(n_levls[:i]) + np.sum(n_levls_sub[i][:ind_irrep0])
                    ind0_end = ind0_start + n_levls_sub[i][ind_irrep0]
                    slice0 = slice(int(ind0_start), int(ind0_end))
                    ind1_start = np.sum(n_levls[:j]) + np.sum(n_levls_sub[j][:ind_irrep1])
                    ind1_end = ind1_start + n_levls_sub[j][ind_irrep1]
                    slice1 = slice(int(ind1_start), int(ind1_end))
                    J_square[slice0, slice1] = tran_mat(angle0, irrep0, angle1, irrep1)
                    J_square[slice1, slice0] = tran_mat(angle1, irrep1, angle0, irrep0)
    return J_square, n_levls_sub, labels, angles
    
                    
def plot_transition_matrix(vector='x', struct_f='struct.obj', eigen_f='EIGEN.npz'):
    J_square, n_levels_sub, labels, angles = calc_transition_matrix(vector, struct_f, eigen_f) 
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=[10,10])
    ndim = len(J_square)
    x = np.arange(-0.5, ndim, 1) 
    y = np.arange(-0.5, ndim, 1)
    
    ax.pcolormesh(x, y, J_square, cmap='gist_heat_r')
    n_levels = np.array([np.sum(i) for i in n_levels_sub])
    angles_line = [np.sum(n_levels[:i]) for i in range(1,len(n_levels))]

    n_levels_sub = np.concatenate(n_levels_sub)
    irreps_line = [np.sum(n_levels_sub[:i]) for i in range(1,len(n_levels_sub))]

    
    ### ireep label 
    irreps_labels = [(np.sum(n_levels_sub[:i])+np.sum(n_levels_sub[:i+1]))/2-0.5 for i in range(0,len(n_levels_sub))]
    ax.set_xticks(irreps_labels)
    y_labels = [[str(angles[i])+'$\degree$ '+j for j in labels[i]] for i in range(len(labels))]
    x_labels = [[str(angles[i])+'$\degree$\n '+j for j in labels[i]] for i in range(len(labels))]
    ax.set_xticklabels(np.concatenate(x_labels))
    ax.set_yticks(irreps_labels)
    ax.set_yticklabels(np.concatenate(y_labels))

    for i in angles_line:
        ax.axvline(i-0.5, ls='solid', color='black')
        ax.axhline(i-0.5, ls='solid', color='black')
    
    for i in irreps_line:
        ax.axvline(i-0.5, ls='dashed', color='black')
        ax.axhline(i-0.5, ls='dashed', color='black')
    
    ax.set_xlim([-0.5, ndim+0.5])
    ax.set_ylim([-0.5, ndim+0.5])
    ax.axis('equal')
    plt.savefig('transition_matrix.pdf')


def _plot_one_vec(fig, ax, qd, vec, title, scale=8000, lw=0.1, alpha=0.8):
    qd.plot(fig, ax, site_size=0, lw=lw)      
    occp = np.round(np.abs(vec)**2, 10) *scale
    ### bottom layer ##
    ind0 = 0
    ind1 = qd.layer_nsites[0]
    coords_i = qd.coords[ind0:ind1]
    ax.scatter(coords_i[:,0], coords_i[:,1], occp[ind0:ind1],color='black', alpha=alpha, linewidths=0, edgecolors=None)
    ### top layer ##
    ind0 = qd.layer_nsites[0]
    ind1 = qd.layer_nsites[0] + qd.layer_nsites[1]
    coords_i = qd.coords[ind0:ind1]
    ax.scatter(coords_i[:,0], coords_i[:,1], occp[ind0:ind1], color='red', alpha=alpha, linewidths=0, edgecolors=None)
    ax.set_title('%s$\degree$ %s' % (title[0], title[1]))
    ax.axis('equal')

def _group_subplots(n, plt):
    ncol = int(np.ceil(np.sqrt(n)))
    if ncol*(ncol-1)>=n:
        nrow = ncol-1
    else:
        nrow = ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*5, nrow*5))

    if n>1:
        axes_ = axes.reshape(-1,)
    else:
        axes_ = [axes]
    return fig, axes_, nrow*ncol
            
def plot_eigenvectors(symmeop_label, elim=None, struct_f='struct.obj',eigen_f='EIGEN.npz',alpha=0.8, lw=0.05):
    """
    symmeop_label: the symmetry operation depending on which the eigen states are grouped
    nlevel_cut: for each subgroup the number of levels above and below the fermi levels are under consideration  
    """
    qd, vals_expand, _ = _parse(struct_f, eigen_f)
    qd.add_hopping_wannier(ts=[-2.8])
    vbm, cbm = get_vbm_cbm(vals_expand)
    ef = (vbm+cbm)/2.
    elim = np.array(elim)+ef
    vals, vecs = classify_eigenstates_SymmeOp_and_irrep(symmeop_label, struct_f, eigen_f)
    def add_one_picture(ax, vec, val):
        qd.plot(fig, ax, site_size=0, lw=lw)      
        occp = np.abs(vec)**2 *4000
        ### bottom layer ##
        ind0 = 0
        ind1 = qd.layer_nsites[0]
        coords_i = qd.coords[ind0:ind1]
        ax.scatter(coords_i[:,0], coords_i[:,1], occp[ind0:ind1],color='black', alpha=alpha, linewidths=0, edgecolors=None)
        ### top layer ##
        ind0 = qd.layer_nsites[0]
        ind1 = qd.layer_nsites[0] + qd.layer_nsites[1]
        coords_i = qd.coords[ind0:ind1]
        ax.scatter(coords_i[:,0], coords_i[:,1], occp[ind0:ind1], color='red', alpha=alpha, linewidths=0, edgecolors=None)
        ax.set_title('$E-E_f$: %.2f' % (val-ef))
        ax.axis('equal')
    for angle in vecs:
        for irrep in vecs[angle]:
            print(angle, irrep)
            vecs_shot = np.array(vecs[angle][irrep])
            vals_shot = np.array(vals[angle][irrep])
            inds = np.intersect1d(np.where(vals_shot>=elim[0])[0], np.where(vals_shot<=elim[1])[0])
            vals_shot_cut = vals_shot[inds]
            vecs_shot_cut = vecs_shot[:,inds]
            n = len(vals_shot_cut)
            nrow = int(np.ceil(np.sqrt(n)))
            if not n:
                print('  No levels chosen for angle %s irrep %s' % (angle, irrep))
                continue 
                
            from matplotlib import pyplot as plt
            if nrow*(nrow-1)>=n:
                ncol = nrow-1
            else:
                ncol = nrow
            fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*5, nrow*5))

            if n>1:
                axes_ = axes.reshape(-1,)
            else:
                axes_ = [axes]
            [add_one_picture(axes_[i], vecs_shot_cut[:,i], vals_shot_cut[i]) for i in range(0,n)]
            for i in range(n, ncol*nrow):
                axes_[i].axis('off')
            plt.savefig('angle%s_irrep%s.png' % (angle, irrep))
            plt.close()

def plot_eigenvals(eigen_f='EIGEN.npz'):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    eig = np.load(eigen_f)
    vals = eig['vals']
    vbm, cbm = get_vbm_cbm(vals)
    ax.scatter(range(len(vals)), vals, color='black')
    ax.axhline((vbm+cbm)/2, ls='dashed')
    ax.set_ylabel('Eenrgy (eV)')
    plt.savefig('eigen_vals.pdf')

def plot_HOMO_LUMO(symmeop_label='Cn1', struct_f='struct.obj', eigen_f='EIGEN.npz', scale=8000):
    qd, vals, _ = _parse(struct_f, eigen_f)
    qd.add_hopping_wannier(ts=[-2.8])
    
    ## get vb_top and cb_bott ##
    occups = occup_0K(vals)
    inds_ef = np.intersect1d(np.where(occups>0)[0], np.where(occups<2)[0])
    ind_vbm = max(np.where(occups==2)[0])
    ind_cbm = min(np.where(occups==0)[0])
    vbm = vals[ind_vbm]
    cbm = vals[ind_cbm]
    is_metal = False
    if len(inds_ef):
        ef = (vals[inds_ef[0]]+vals[inds_ef[-1]])/2
        is_metal = True

    def get_inds_vbm_cbm_zero(angle, irrep, vals_shot):
        vb_relat = np.round(vals_shot-vbm, 7)
        cb_relat = np.round(vals_shot-cbm, 7)
        inds_vbm = np.where(vb_relat==0.)[0]
        inds_cbm = np.where(cb_relat==0.)[0]
        inds_ef = []
        if is_metal:
            ef_relat = np.round(vals_shot-ef, 7)
            inds_ef = np.where(ef_relat==0.)[0]
        return [(angle, irrep, i) for i in inds_vbm], \
               [(angle, irrep, i) for i in inds_cbm], \
               [(angle, irrep, i) for i in inds_ef]
        
    vals, vecs = classify_eigenstates_SymmeOp_and_irrep(symmeop_label, struct_f, eigen_f)
    homos = []
    lumos = []
    zeros = []
    
    for angle in vals:
        for irrep in vals[angle]:
            vals_shot = vals[angle][irrep]
            inds_vbm, inds_cbm, inds_ef = get_inds_vbm_cbm_zero(angle, irrep, vals_shot)
            homos.append(inds_vbm)
            lumos.append(inds_cbm)
            zeros.append(inds_ef)
    homos = np.concatenate([i for i in homos if i])
    lumos = np.concatenate([i for i in lumos if i])
    zeros = [i for i in zeros if i]
    if len(zeros):
        zeros = np.concatenate(zeros)
    band_edges = {'HOMO':homos, 'LUMO':lumos, 'zero_energy':zeros}
    for label in band_edges:
        if not len(band_edges[label]):
            continue
        vecs_pickup = [vecs[int(i[0])][i[1]][:,int(i[2])] for i in band_edges[label]]
        vecs_pickup = np.column_stack(vecs_pickup)
        titles = [(i[0], i[1]) for i in band_edges[label]] 
        n = len(band_edges[label])
        nrow = int(np.ceil(np.sqrt(n)))
        if not n:
            continue 
        from matplotlib import pyplot as plt
        fig, axes, nt = _group_subplots(n, plt)
        
        [_plot_one_vec(fig, axes[i], qd, vecs_pickup[:,i], titles[i], scale=scale, lw=0.1, alpha=0.8) for i in range(0,n)]

        for i in range(n, nt):
            axes_[i].axis('off')
        plt.savefig('%s.png' % label)
        plt.close()

        
            
            
                 
    
    



def check_classified_vecs_SymmeOp(SymmeOp_label='Cn1', struct_f='struct.obj', eigen_f='EIGEN.npz'):
    vals, vecs = classify_eigenstates_with_SymmeOp_values(SymmeOp_label=SymmeOp_label, struct_f=struct_f, eigen_f=eigen_f)
    with open(struct_f, 'rb') as f:
        qd = pickle.load(f)
    H = qd.get_Hamiltonian()
    if SymmeOp_label == 'Cn1':
        OP = get_representation_matrices(qd)[1][0]
    elif SymmeOp_label == 'sigma_x':
        OP = get_representation_matrices(qd)[-1][0]
    print('%s check' % SymmeOp_label)
    for angle in vecs:
        print('\n\n\nangle=%s degree' % angle)
        mu = np.exp(1j*angle*np.pi/180)
        for irrep in vecs[angle]:
            print('\n', irrep)
            vecs_shot = vecs[angle][irrep]
            vecs_new = np.matmul(OP, vecs_shot)
            for i in range(len(vecs_shot[0])):
                vecs_new[:,i] = vecs_new[:,i]/mu
                print(np.round(vecs_new[:,i] - vecs_shot[:,i], 4), mu)
                print('orth?', np.dot(vecs_shot[:,i].conj(), vecs_shot[:,i]))

def check_classified_vecs_irrep(struct_f='struct.obj', eigen_f='EIGEN.npz'):
    vals, vecs = classify_eigen_states_irrep(struct_f=struct_f, eigen_f=eigen_f)
    with open(struct_f, 'rb') as f:
        qd = pickle.load(f)
    H = qd.get_Hamiltonian()

    for irrep in vecs:
        print(irrep)
        vecs_shot = vecs[irrep]
        vals_shot = vals[irrep]
        vecs_new = np.matmul(H, vecs_shot)
        for i in range(len(vals_shot)):
            vecs_new[:,i] = vecs_new[:,i]/vals_shot[i]
            print(np.round(vecs_new[:,i] - vecs_shot[:,i], 4).real, vals_shot[i])
        print('\n\n')

def plot_HOMO_LUMO_new(symmeop_label='Cn1', struct_f='struct.obj', eigen_f='EIGEN.npz', scale=8000):
    qd, vals, vecs = _parse(struct_f, eigen_f)
    qd.add_hopping_wannier(ts=[-2.8])
    inds_vbm, inds_cbm, inds_ef = get_inds_band_edge('vals')
    vecs_cbm = vecs[:,inds_cbm]
    vecs_vbm = vecs[:,inds_vbm]
    vecs_ef = vecs[:,inds_ef]
    
    

