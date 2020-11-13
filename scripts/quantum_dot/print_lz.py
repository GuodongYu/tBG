import numpy as np
from symmetry_analyzer import check_classified_vecs_irrep, check_classified_vecs_SymmeOp, \
        classify_eigenstates_with_SymmeOp_values, calcu_Lz_averg, plot_levels_with_SymmpOp_evalues
from optical_conductivity import occup_0K, get_vbm_cbm

eig = np.load('EIGEN.npz', allow_pickle=True)
vbm, cbm = get_vbm_cbm(eig['vals'])
val, lz, vec = calcu_Lz_averg()
print('vbm:',vbm, 'cbm:',cbm)
print('120 degree: E irrep')
for i in range(len(lz[120]['E'])):
    print('%i lz:%.4f val:%.4f' % (i, lz[120]['E'].real[i], val[120]['E'][i]))

print('\nvbm:',vbm, 'cbm:',cbm)
print('-120 degree: E irrep')
for i in range(len(lz[-120]['E'])):
    print('%i lz:%.4f val:%.4f' % (i, lz[-120]['E'].real[i], val[-120]['E'][i]))

print('\nvbm:',vbm, 'cbm:',cbm)
print('0 degree: A1 irrep')
for i in range(len(lz[0]['A1'])):
    print('%i lz:%.4f val:%.4f' % (i, lz[0]['A1'].real[i], val[0]['A1'][i]))

print('\nvbm:',vbm, 'cbm:',cbm)
print('0 degree: A2 irrep')
for i in range(len(lz[0]['A2'])):
    print('%i lz:%.4f val:%.4f' % (i, lz[0]['A2'].real[i], val[0]['A2'][i]))

