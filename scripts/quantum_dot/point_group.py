import numpy as np
import pickle

class SharedMethods:

    def rrep2irrep(self, chis):
        """
        input:
            chi: the list of characters of all classes, it must be in order of classs E, [C3_1, C3_2], [three sigma_v]
        retrun: 
            the coefficient of each irrep 
        """
        n_class = len(chis)
        chis = np.array(chis)
        a_s = {}
        for irrep in self.irreps:
            a = 1/self.g* np.sum([chis[i]*np.conj(self.irreps[irrep][i])*len(self.classes[i]) for i in range(n_class)])    
            if np.abs(a.imag) > 1.e-8 or np.abs(round(a.real)-a.real) > 1.e-8:
                raise ValueError('This is not a good group representation!')
            a_s[irrep] = int(round(a.real,0))
        return a_s

    def projection_operators(self, ops):
        """
        ops: mats group as class
        """
        ops_class = np.array([np.sum(i, axis=0) for i in ops])
        n_class = len(ops)
        proj_ops = {}
        for irrep in self.irreps:
            if irrep[0] in ['A','B']:
                l = 1
            elif irrep[0] == 'E':
                l = 2
            proj_ops[irrep] = l/self.g * np.sum([self.irreps[irrep][i]*ops_class[i] for i in range(n_class)], axis=0)
        return proj_ops
    

class C3v(SharedMethods):
    def __init__(self):
        self.group_elems = ['E', 'C3_1', 'C3_2', 'sigma_v_1', 'sigma_v_2', 'sigma_v_3']
        self.g = len(self.group_elems)
        self.classes = [['E'], ['C3_1','C3_2'], ['sigma_v_1', 'sigma_v_2', 'sigma_v_3']]
        self.irreps = {'A1': np.array([1, 1, 1]), 
                       'A2': np.array([1, 1, -1]), 
                       'E': np.array([2, -1, 0])}

    @property
    def product_table(self):
        A1 = {'A1':['A1'], 'A2':['A2'], 'E':['E']}
        A2 = {'A1':['A2'], 'A2':['A1'], 'E':['E']}
        E = {'A1':['E'], 'A2':['E'], 'E':['A1','A2','E']}
        return {'A1':A1, 'A2':A2, 'E':E}


class C6v(SharedMethods):
    def __init__(self):
        self.group_elems = ['E', 'C6_1', 'C6_5', 'C6_2', 'C6_4', 'C6_3', \
                            'sigma_v_1', 'sigma_v_2', 'sigma_v_3',\
                            'sigma_d_1', 'sigma_d_2', 'sigma_d_3']
        self.g = len(self.group_elems)
        self.classes = [['E'], ['C6_1','C6_5'], ['C6_2','C6_4'], ['C6_3'], \
                        ['sigma_v_1', 'sigma_v_2', 'sigma_v_3'],\
                        ['sigma_d_1', 'sigma_d_2', 'sigma_d_3']]
        self.irreps = {'A1': np.array([1,  1,  1,  1,  1,  1]), 
                       'A2': np.array([1,  1,  1,  1, -1, -1]), 
                       'B1': np.array([1, -1,  1, -1,  1, -1]),
                       'B2': np.array([1, -1,  1, -1, -1,  1]),
                       'E1': np.array([2,  1, -1, -2,  0,  0]),
                       'E2': np.array([2, -1, -1,  2,  0,  0])}

    @property
    def product_table(self):
        A1 = {'A1':['A1'], 'A2':['A2'], 'B1':['B1'], 'B2':['B2'], 'E1':['E1'], 'E2':['E2']}
        A2 = {'A1':['A2'], 'A2':['A1'], 'B1':['B2'], 'B2':['B1'], 'E1':['E1'], 'E2':['E2']}
        B1 = {'A1':['B1'], 'A2':['B2'], 'B1':['A1'], 'B2':['A2'], 'E1':['E2'], 'E2':['E1']}
        B2 = {'A1':['B2'], 'A2':['B1'], 'B1':['A2'], 'B2':['A1'], 'E1':['E2'], 'E2':['E1']}
        E1 = {'A1':['E1'], 'A2':['E1'], 'B1':['E2'], 'B2':['E2'], 'E1':['A1', 'A2', 'E2'], 'E2':['B1', 'B2', 'E1']}
        E2 = {'A1':['E2'], 'A2':['E2'], 'B1':['E1'], 'B2':['E1'], 'E1':['B1','B2', 'E1'], 'E2':['A1', 'A2', 'E2']}
        return {'A1':A1, 'A2':A2, 'B1':B1, 'B2':B2, 'E1':E1, 'E2':E2}


class D6d(SharedMethods):
    def __init__(self):
        self.group_elems = ['E', 'S12_1', 'S12_11', 'C6_1', 'C6_5', 'S12_3', 'S12_9',\
                            'C6_2', 'C6_4', 'S12_5', 'S12_7', 'C6_3', \
                            'C2_1_1', 'C2_1_2', 'C2_1_3', 'C2_1_4', 'C2_1_5', 'C2_1_6', \
                            'sigma_d_1', 'sigma_d_2', 'sigma_d_3', 'sigma_d_4', 'sigma_d_5', 'sigma_d_6']
        self.g = len(self.group_elems)
        self.classes = [['E'], ['S12_1', 'S12_11'], ['C6_1', 'C6_5'], ['S12_3', 'S12_9'],\
                        ['C6_2', 'C6_4'], ['S12_5', 'S12_7'], ['C6_3'], \
                        ['C2_1_1', 'C2_1_2','C2_1_3', 'C2_1_4', 'C2_1_5', 'C2_1_6'],\
                        ['sigma_d_1', 'sigma_d_2', 'sigma_d_3', 'sigma_d_4', 'sigma_d_5', 'sigma_d_6']]
        self.irreps = {'A1': np.array([1,    1,        1,  1,  1,     1,        1,  1,  1]), 
                       'A2': np.array([1,    1,        1,  1,  1,     1,        1, -1, -1]), 
                       'B1': np.array([1,   -1,        1, -1,  1,    -1,        1,  1, -1]),
                       'B2': np.array([1,   -1,        1, -1,  1,    -1,        1, -1,  1]),
                       'E1': np.array([2, np.sqrt(3),  1,  0, -1, -np.sqrt(3), -2,  0,  0]),
                       'E2': np.array([2,    1,       -1, -2, -1,     1,        2,  0,  0]),
                       'E3': np.array([2,    0,       -2,  0,  2,     0,       -2,  0,  0]),
                       'E4': np.array([2,   -1,       -1,  2, -1,    -1,        2,  0,  0]),
                       'E5': np.array([2,-np.sqrt(3),  1,  0, -1,  np.sqrt(3), -2,  0,  0])}

    @property
    def product_table(self):
        A1 = {'A1':['A1'], 'A2':['A2'], 'B1':['B1'], 'B2':['B2'], 'E1':['E1'], 'E2':['E2'], 'E3':['E3'], 'E4':['E4'], 'E5':['E5']}
        A2 = {'A1':['A2'], 'A2':['A1'], 'B1':['B2'], 'B2':['B1'], 'E1':['E1'], 'E2':['E2'], 'E3':['E3'], 'E4':['E4'], 'E5':['E5']}
        B1 = {'A1':['B1'], 'A2':['B2'], 'B1':['A1'], 'B2':['A2'], 'E1':['E5'], 'E2':['E4'], 'E3':['E3'], 'E4':['E2'], 'E5':['E1']}
        B2 = {'A1':['B2'], 'A2':['B1'], 'B1':['A2'], 'B2':['A1'], 'E1':['E5'], 'E2':['E4'], 'E3':['E3'], 'E4':['E2'], 'E5':['E1']}
        E1 = {'A1':['E1'], 'A2':['E1'], 'B1':['E5'], 'B2':['E5'], 'E1':['A1', 'A2', 'E2'], 'E2':['E1','E3'], \
              'E3':['E2','E4'], 'E4':['E3','E5'],'E5':['B1','B2','E4']}
        E2 = {'A1':['E2'], 'A2':['E2'], 'B1':['E4'], 'B2':['E4'], 'E1':['E1','E3'], 'E2':['A1', 'A2', 'E4'], \
              'E3':['E1','E5'],'E4':['B1','B2','E3'], 'E5':['E3','E5']}
        E3 = {'A1':['E3'], 'A2':['E3'], 'B1':['E3'], 'B2':['E3'], 'E1':['E2','E4'], 'E2':['E1','E5'], \
              'E3':['A1','A2','B1','B2'], 'E4':['E1','E5'],'E5':['E2','E4']}
        E4 = {'A1':['E4'], 'A2':['E4'], 'B1':['E2'], 'B2':['E2'], 'E1':['E3','E5'], 'E2':['B1','B2','B5'],\
              'E3':['E1','E5'], 'E4':['A1','A2','E4'],'E5':['E1','E3']}
        E5 = {'A1':['E5'], 'A2':['E5'], 'B1':['E1'], 'B2':['E1'], 'E1':['B1','B2', 'E4'], 'E2':['E3','E5'],\
              'E3':['E2','E4'], 'E4':['E1','E3'],'E5':['A1','A2','E2']}
        return {'A1':A1, 'A2':A2, 'B1':B1, 'B2':B2, 'E1':E1, 'E2':E2, 'E3':E3, 'E4':E4, 'E5':E5}

def PointGroup(label):
    if label == 'C3v':
        return C3v()
    elif label == 'C6v':
        return C6v()
    elif label == 'D6d':
        return D6d()


def test_projects():
    from tBG.quantum_dot import QuantumDotQC

    qc = QuantumDotQC()
    qc.regular_polygon(3, 4.5)
    qc.remove_top_layer()
    ops = qc.symmetry_operations()
    E = np.identity(len(qc.coords))
    C3 = qc.symmetryop2matrix(ops['Cns'][0])
    C3_2 = qc.symmetryop2matrix(ops['Cns'][1])
    sigma_v_1 = qc.symmetryop2matrix(ops['sigma_vs'][0])
    sigma_v_2 = qc.symmetryop2matrix(ops['sigma_vs'][1])
    sigma_v_3 = qc.symmetryop2matrix(ops['sigma_vs'][2])

    c3v = C3v()
    projs = c3v.projection_operators(E, C3, C3_2, sigma_v_1, sigma_v_2, sigma_v_3)
    return projs

def check_states(eigen_f='EIGEN.npz'):
    projs = test_projects()
    A1 = projs['A1']
    A2 = projs['A2']
    E = projs['E']
    eigen = np.load('EIGEN.npz', allow_pickle=True)
    vals = eigen['vals']
    vecs = eigen['vecs']
    for i in range(len(vecs[0])):
        vec = vecs[:,i]
        norm_A1 = np.linalg.norm(np.matmul(A1, vec))
        norm_A2 = np.linalg.norm(np.matmul(A2, vec))
        norm_E = np.linalg.norm(np.matmul(E, vec))
        print(vals[i])
        if norm_A1 > 0.005:
            print('  A1')
        if norm_A2 > 0.005:
            print('  A2')
        if norm_E > 0.005:
            print('  E')
        print('\n')
        #print('val:%s' % vals[i], 'A1:%s'%np.round(norm_A1,3), 'A2:%s'%np.round(norm_A2,3), 'E:%s'%np.round(norm_E,3))

def check_states_orth(eigen_f='EIGEN.npz', struct_obj='struct.obj'):
    with open(struct_obj, 'rb') as f:
        qc = pickle.load(f)
    H = qc.get_Hamiltonian()

    projs = test_projects()
    A1 = projs['A1']
    A2 = projs['A2']
    E = projs['E']
    eigen = np.load('EIGEN.npz', allow_pickle=True)
    vals = eigen['vals']
    vecs = eigen['vecs']
    v0 = vecs[:, 6]
    val = vals[6]
    v1 = vecs[:, 7]
    v2 = vecs[:, 8]
    A2_v0 = np.matmul(A2, v0)
    A2_v0 = 1/np.linalg.norm(A2_v0)*A2_v0
    E_v1 = np.matmul(E, v1)
    E_v1 = 1/np.linalg.norm(E_v1)*E_v1
    
    E_v2 = np.matmul(E, v2)
    E_v2 = 1/np.linalg.norm(E_v2)*E_v2
    dv0 = np.matmul(H, A2_v0)/val - A2_v0
    dv1 = np.matmul(H, E_v1)/val - E_v1
    dv2 = np.matmul(H, E_v2)/val - E_v2
    print(dv0,'\n')
    print(dv1,'\n')
    print(dv2,'\n')

if __name__ == '__main__':
    #check_states()
    check_states_orth()
