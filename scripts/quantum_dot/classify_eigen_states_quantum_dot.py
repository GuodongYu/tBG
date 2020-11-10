import pickle
import numpy as np
from tBG.scripts.quantum_dot.point_group import PointGroup
   
def _parse(struct_f, eigen_f):
    with open(struct_f, 'rb') as f:
        qd = pickle.load(f)
    eigen = np.load(eigen_f, allow_pickle=True)
    vals = eigen['vals']
    vecs = eigen['vecs']
    return qd, vals, vecs

def get_representation_matrices(qd):
    """
    get the reducible representation matrices grouped by classes
    return: 
        for C3v, [[E], [C3_1,C3_2], [three sigma_v]]
        for C6v, [[E], [C6_1,C6_5], [C6_2,C6_4],[C6_3], [three sigma_v], [three sigma_d]]
        for D6h, [[E], [S12_1, S12_11], [C6_1, C6_5], [S12_3, S12_9], [C6_2, C6_4], [S12_5, S12_7], [C6_3], [six C2_1], [six sigma_d]]
    """
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
        C2 = [qd.symmetryop2matrix(i) for i in ops['C2s_Qs']]
        sigma_d = [qd.symmetryop2matrix(i) for i in ops['sigma_vs']]
        return [[E], S12_1_11, C6_1_5, S12_3_9, C6_2_4, S12_5_7, C6_3, C2, sigma_d] 

def get_projection_operators(qd):
    """
    get the projection operators for all irreps
    """
    ops = get_representation_matrices(qd)
    #ops_expand = np.concatenate(ops, axis=0)
    pg = PointGroup(qd.point_group)
    return pg.projection_operators(ops)

def decompose_as_irreps(qd):
    """
    decompose the redicible representation to irreps
    """
    ops = get_representation_matrices(qd)
    chi = [np.trace(i[0]) for i in ops]
    pg = PointGroup(qd.point_group)
    return pg.rrep2irrep(chi)

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
    
def _classify_eigenstates_with_SymmeOp_values(SymmeOp_mat, SymmeOp_vals_1D_irrep, \
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

def classify_eigenstates_with_SymmeOp_values(SymmeOp_label='sigma_x', struct_f='struct.obj', eigen_f='EIGEN.npz'):
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
    return _classify_eigenstates_with_SymmeOp_values(SymmeOp_mat, SymmeOp_vals_1D_irrep, \
                                           struct_f='struct.obj', eigen_f='EIGEN.npz')         
    
def plot_levels_with_SymmpOp_evalues(ax, t=2.8, SymmeOp_label='sigma_x', ylim=None, struct_f='struct.obj', eigen_f='EIGEN.npz'):
    vals, vecs = classify_eigenstates_with_SymmeOp_values(SymmeOp_label=SymmeOp_label, struct_f=struct_f, eigen_f=eigen_f)
    symmpOP_values = vals.keys()
    xs = [-0.15, -0.05]
    dx = xs[1]-xs[0]
    shift = 0.05

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

def calcu_Lz_averg(SymmeOp_label='Cn1', struct_f='struct.obj', eigen_f='EIGEN.npz'):
    with open(struct_f, 'rb') as f:
        qd = pickle.load(f)
    Lz_mat = qd.get_Lz_mat()
    vals, vecs = classify_eigenstates_with_SymmeOp_values(SymmeOp_label=SymmeOp_label, struct_f='struct.obj', eigen_f='EIGEN.npz')
    lzs = {}
    for rou in vals:
        lzs[rou] = {}
        for irrep in vals[rou]:
            vecs_rou_irrep = vecs[rou][irrep]
            vecs_new = np.matmul(Lz_mat, vecs_rou_irrep)
            lzs_rou_irrep = [np.dot(vecs_rou_irrep[:,i].conj(), vecs_new[:,i])  for i in range(len(vecs_new[0]))]
            lzs[rou][irrep] = np.array(lzs_rou_irrep)
    return vals, lzs, vecs

def check_classified_vecs_SymmeOp(SymmeOp_label='Cn1', struct_f='struct.obj', eigen_f='EIGEN.npz'):
    vals, vecs = classify_eigenstates_with_SymmeOp_values(SymmeOp_label=SymmeOp_label, struct_f=struct_f, eigen_f=eigen_f)
    with open(struct_f, 'rb') as f:
        qd = pickle.load(f)
    H = qd.get_Hamiltonian()
    if SymmeOp_label == 'Cn1':
        OP = get_representation_matrices(qd)[1][0]
    elif SymmeOp_label == 'sigma_x':
        OP = get_representation_matrices(qd)[-1][0]

    #print('Hamiltonian check')
    #for angle in vecs:
    #    print(angle)
    #    for irrep in vecs[angle]:
    #        print(' ', irrep)
    #        vecs_shot = vecs[angle][irrep]
    #        vals_shot = vals[angle][irrep]
    #        vecs_new = np.matmul(H, vecs_shot)
    #        for i in range(len(vals_shot)):
    #            vecs_new[:,i] = vecs_new[:,i]/vals_shot[i]
    #            print(np.round(vecs_new[:,i] - vecs_shot[:,i], 4), vals[angle][irrep][i])

    #print('\n\n\n\n\n')

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
                
        
                

def prb_tri_lz():
    from tBG.quantum_dot import QuantumDotQC
    
    qc = QuantumDotQC()
    qc.regular_polygon(3, 5, overlap='atom1')
    qc.remove_top_layer()
    qc.add_hopping_pz(max_dist=2.0, g0=2.8)
    x = qc.pymatgen_struct()
    x.to('poscar','POSCAR')
    with open('struct.obj','wb') as f:
        pickle.dump(qc, f)
    qc.diag_Hamiltonian()


def prb_tri():
    from tBG.quantum_dot import QuantumDotQC
    
    qc = QuantumDotQC()
    qc.regular_polygon(3, 2, overlap='atom1')
    qc.remove_top_layer()
    qc.add_hopping_pz(max_dist=2.0, g0=2.8)
    x = qc.pymatgen_struct()
    x.to('poscar','POSCAR')
    with open('struct.obj','wb') as f:
        pickle.dump(qc, f)
    qc.diag_Hamiltonian()

def main():
    from tBG.quantum_dot import QuantumDotQC
    
    qc = QuantumDotQC()
    qc.regular_polygon(6, 4, overlap='hole')
    qc.remove_top_layer()
    #qc.add_hopping_pz(max_dist=2.0, g0=2.8)
    #qc.add_hopping_pz()
    x = qc.pymatgen_struct()
    x.to('poscar','POSCAR')
    qc.add_hopping_wannier(P=0)
    with open('struct.obj','wb') as f:
        pickle.dump(qc, f)
    qc.diag_Hamiltonian()


if __name__ == "__main__":
    prb_tri_lz()
