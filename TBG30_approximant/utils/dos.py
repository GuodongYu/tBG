import numpy as np
import json
from monty.json import jsanitize
from tBG.TBG30_approximant.structure import Structure
from tBG.TBG30_approximant.diag import DiagSample

def diag_sample(M, append=None,hop_dist=5.0, elec_field=0.0, k_mesh_size=[20, 20], k_mesh_method='Gamma', \
                symmetry=True, vec_calc=False, chg_calc=True):
    """
    diag system to get the eigen values or vectors
    """
    s = Structure()
    s.make_structure(M)
    if append:
        s.append_layers(append)
    s.add_hopping(max_dist=hop_dist)
    diagsmp = DiagSample(s, elec_field=elec_field)
    diagsmp.add_kpoints_BZ(k_mesh_size, k_mesh_method)
    diagsmp.diag_run(symmetry=symmetry, vec_calc=vec_calc, chg_calc=chg_calc)

def dos_calc(eig_file='EIGEN.npz', save_to='DOS', sigma=0.01, nedos=10000):
    """
    To caculate the dos from eigenvalues saved in eig_file using Gaussian function
    Args:
        eig_file: the binary file saving the eigenvalues and eigenvectors output by numpy
        sigma: the width of Gaussion function to smearing the DOS
        nedos: the number of points for sampling the energy range
        save_to: the calculated dos will be saved into save_to file
    """
    from tBG.fortran.get_dos import get_dos
    data = np.load(eig_file)
    weights = data['weights']
    eigs = data['vals']
    energies, dos = get_dos(weights, eigs, sigma, nedos)
    nk = len(eigs)
    norb = len(eigs[0])

    def get_ne(inds):
        summ = 0
        for ik in range(nk):
            summ = summ + 2*weights[ik]*inds.count(ik)
        ne = summ/sum(weights)
        return ne

    def get_Ef(prec=0.0001):
        ind_VBM = int(np.ceil(norb/2)) - 1
        ind_CBM = ind_VBM + 1
        VBM = max(eigs[:,ind_VBM])
        CBM = min(eigs[:,ind_CBM])
        if VBM<=CBM:
            return (VBM+CBM)/2
        else:
            VBM, CBM = CBM, VBM
            inds = np.where(eigs<=CBM)[0].tolist()
            ne = get_ne(inds)
            while abs(VBM-CBM)> prec:
                e = (VBM+CBM)/2
                inds = np.where(eigs<=e)[0].tolist()
                ne = get_ne(inds)
                print('ne:%s, norb:%s, VBM:%s, CBM:%s' % (ne, norb, VBM, CBM))
                if ne > norb:
                    CBM = e
                else:
                    VBM = e
            return VBM
    Ef = get_Ef() 
    DOS = np.array([energies, dos]).T
    np.savetxt(save_to, DOS, header='Ef %s' % Ef)

def dos_plot(dos_f='DOS'):
    """
    plot dos from the data in dos_f file
    """
    from matplotlib import pyplot as plt
    with open(dos_f) as f:
        Ef = float(f.readline().split(' ')[-1])
    dos = np.loadtxt(dos_f)
    plt.plot(dos[:,0]-Ef, dos[:, 1])
    plt.xlabel('$\mathrm{Energy-E_f}$ (eV)')
    plt.ylabel('Densit of states')
    plt.savefig('dos.pdf')


if __name__=='__main__':
    diag_sample(15)
    dos_calc()
    dos_plot()
