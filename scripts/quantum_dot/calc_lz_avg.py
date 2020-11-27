import numpy as np
from tBG.molecule.symmetry_analyzer import _parse, get_representation_matrices

def subspaces_as_energy(vals):
    inds = [[0]]
    val = vals[0]
    for i in range(1, len(vals)):
        vali = vals[i]
        if abs(vali-val)<=1.e-8:
            inds[-1].append(i)
        else:
            inds.append([i])
            val = vali
    return inds

def classify_eigen_states_SymmeOp(SymmeOp_mat,struct_f='struct.obj', eigen_f='EIGEN.npz'):
    _, vals, vecs = _parse(struct_f=struct_f, eigen_f=eigen_f)
    subspaces = subspaces_as_energy(vals)
    angles = []
    for inds_sub in subspaces:
        ndim = len(inds_sub)
        vecs_sub = vecs[:,inds_sub]
        left_vec = np.transpose(vecs_sub).conj()
        right_vec = np.matmul(SymmeOp_mat, vecs_sub)
        SymmeOp_mat_sub = np.matmul(left_vec, right_vec)
        vals_Op, vecs_Op = np.linalg.eig(SymmeOp_mat_sub)
        for i in range(len(vals_Op)):
            mu = vals_Op[i]
            angle = np.angle(mu)*180/np.pi
            angle = int(round(angle))
            angles.append(angle)
            vec = np.sum(vecs_Op[:,i]*vecs_sub, axis=1).reshape(-1,1)
            vecs[:,inds_sub[i]] = vec.reshape(1,-1)
    return angles, vals, vecs

def calc_Lz_avg(vecs, Lz_mat):
    right_vecs = np.matmul(Lz_mat, vecs)
    left_vecs = np.transpose(vecs).conj()
    lz_avgs = np.diag(np.matmul(left_vecs, right_vecs))
    return lz_avgs


def main():
    qd, vals, vecs = _parse('struct.obj','EIGEN.npz')
    C31 = get_representation_matrices(qd)[1][1]
    Lz_mat = qd.get_Lz_mat()
    angles, vals, vecs = classify_eigen_states_SymmeOp(C31)
    lzs = calc_Lz_avg(vecs, Lz_mat)
    for i in range(len(vals)):
        print(angles[i], lzs[i].real, vals[i], np.linalg.norm(vecs[:,i]))


if __name__ == '__main__':
    main() 
