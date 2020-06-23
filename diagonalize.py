import numpy as np
import copy
from tBG.brillouin_zones import BZHexagonal, BZMoirePattern, kpoints_line_mode_onek


############### diag run #####################
def diag_k_mesh(struct, k_mesh=[20,20], method='Gamma', symm=True, elec_field=0.0, fname='EIGEN_val_mesh'):
    """
    This function is used to calculate DOS
    """
    bz = BZHexagonal(struct.latt_vec)
    kpts, weigs = bz.kpoints_mesh_mode(k_mesh, method, symm)
    vals, vecs, pmks = struct.diag_kpts(kpts, vec=0, pmk=0, elec_field=elec_field)
    np.savez_compressed(fname, kpoints=kpts, vals=vals, weights=weigs)

def diag_k_path(struct, k_path=['G','M','K','G'], dk=0.01, elec_field=0.0, fname='EIGEN_val_path'):
    """
    This function is used to calculate band structure along k_path
    """
    bz = BZHexagonal(struct.latt_vec)
    kpts, inds = bz.kpoints_line_mode(k_path, dk)
    vals, vecs, pmks = struct.diag_kpts(kpts, vec=0, pmk=0, elec_field=elec_field)
    k_info = {'labels':k_path, 'inds':inds}
    np.savez_compressed(fname, kpoints=kpts, vals=vals, k_info=[k_info])

def diag_k_path_rotate(struct, k_path=['G','M','K','G'], dk=0.01, elec_field=0.0, fname='EIGEN_val_path', rotate=30):
    """
    This function is used to calculate band structure along rotated k_path
    """
    from tBG.utils import rotate_on_vec
    bz = BZHexagonal(struct.latt_vec)
    kpts, inds = bz.kpoints_line_mode(k_path, dk)
    kpts = rotate_on_vec(rotate, kpts)
    vals, vecs, pmks = struct.diag_kpts(kpts, vec=0, pmk=0, elec_field=elec_field)
    k_info = {'labels':k_path, 'inds':inds}
    np.savez_compressed(fname, kpoints=kpts, vals=vals, k_info=[k_info])

def diag_k_path_between_Diracs(struct, dk=0.01, elec_field=0.0, fname='EIGEN_val_path_between_Diracs'):
    bzs = BZMoirePattern(struct.latt_vec_bott, struct.latt_vec_top, struct.latt_vec)
    kpts, inds, labels = bzs.kpath_between_Diracs(dk)
    vals, vecs, pmks = struct.diag_kpts(kpts, vec=0, pmk=0, elec_field=elec_field)
    k_info = {'labels':labels, 'inds':inds}
    np.savez_compressed(fname, kpoints=kpts, vals=vals, k_info=[k_info])

def diag_onek_around(struct, k_label='K', dk=0.002, elec_field=0.0, fname='EIGEN_val_path_onek', direction=[1,0], nstep=40):
    """
    This function is used to calculate the band structure around one kpoint (corresponding to k_label) along the given direction 
    """
    bz = BZHexagonal(struct.latt_vec)
    kpts, ind = bz.kpoints_line_mode_onek(k_label, dk, direction, nstep=nstep)
    vals, vecs, pmks = struct.diag_kpts(kpts, vec=0, pmk=0, elec_field=elec_field)
    k_info = {'labels':[k_label], 'inds':[ind]}
    np.savez_compressed(fname, kpoints=kpts, vals=vals, k_info=[k_info])

def diag_onek_around_ebs(struct, kpt=None, k_label=None, dk=0.002, elec_field=0.0, fname='EIGEN_val_path_onek', direction=[1,0], nstep=40):
    """
    This function is used to calculate the unfolded band structure around one kpoint (corresponding to k_label) along the given direction 
    """
    bz = BZHexagonal(struct.latt_vec)
    if k_label is not None:
        kpts, ind = bz.kpoints_line_mode_onek(k_label, dk, direction, nstep=nstep)
    elif kpt is not None:
        kpts, ind = kpoints_line_mode_onek(kpt, dk=dk, direction=direction, nstep=nstep)
    vals, vecs, pmks = struct.diag_kpts(kpts, vec=0, pmk=1, elec_field=elec_field)
    k_info = {'labels':[k_label], 'inds':[ind]}
    np.savez_compressed(fname, kpoints=kpts, vals=vals, k_info=[k_info], pmks=pmks)

def diag_k_path_MKG_ebs(struct, dk=0.01, elec_field=0.0, fname='EIGEN_val_pmk_path_MKG'):
    bzs = BZMoirePattern(struct.latt_vec_bott, struct.latt_vec_top, struct.latt_vec)
    kpts, inds, labels = bzs.kpath_M_K_G_bottom(dk)
    vals, vecs, pmks = struct.diag_kpts(kpts, vec=1, pmk=1, elec_field=elec_field)
    #k_info = {'labels':['$K$','$\widetilde{K}_1$','$\Gamma$'], 'inds':inds}
    k_info = {'labels':labels, 'inds':inds}
    np.savez_compressed(fname, kpoints=kpts, vals=vals, pmks=pmks, k_info=[k_info])

def diag_k_path_between_Diracs_ebs(struct, dk=0.01, elec_field=0.0, fname='EIGEN_val_pmk_path_between_Diracs'):
    bzs = BZMoirePattern(struct.latt_vec_bott, struct.latt_vec_top, struct.latt_vec)
    kpts, inds, labels = bzs.kpath_between_Diracs(dk)
    vals, vecs, pmks = struct.diag_kpts(kpts, vec=1, pmk=1, elec_field=elec_field)
    #k_info = {'labels':['$K$','$\widetilde{K}_1$','$\Gamma$'], 'inds':inds}
    k_info = {'labels':labels, 'inds':inds}
    np.savez_compressed(fname, kpoints=kpts, vals=vals, pmks=pmks, k_info=[k_info])

def diag_k_mesh_ebs(struct, k_mesh, elec_field=0.0, fname='EIGEN_val_pmk_mesh'):
    bzs = BZMoirePattern(struct.latt_vec_bott, struct.latt_vec_top, struct.latt_vec)
    kpts = bzs.kpoints_mesh_mode(k_mesh)
    vals, vecs, pmks = struct.diag_kpts(kpts, vec=1, pmk=1, elec_field=elec_field)
    np.savez_compressed(fname, kpoints=kpts, vals=vals, pmks=pmks)

########### post-process after diag ###########################

def chem_pot_calc(eigs, weights, d_ne=0, prec=0.0001):
    """
    args:
        eigs: 2D array saving eigenvalues with shape (nk, norb), nk: kpoint number, norb: band number
        weights: 1D array saving the weights of all kpoints
        d_ne: the change of total electron number (ne_tot), positive and negative for electron and hole doping respectively.
              namely ne_tot = norb + d_ne
              d_ne=0 means no doping, namely ne = norb
    return:
        the absolute chemical potential after adding d_ne electrons, 
    NOTE:
        ** the returned value is absolute not relative to Fermi level (Ef)
        ** if chemical potential relative to Ef is needed, please calculate Ef, namely chemical potential at d_ne=0, firstly

    comment:
        if the chemical potential is needed at a specfic doping concentration (c)
        d_ne = c*area, area is the area of the 2D system.
        if you need to know the area of your system firstly and get d_ne from c and area.
    """
    nk = len(eigs)
    norb = len(eigs[0])
    ne_tot = norb + d_ne

    def get_ne(inds):
        summ = 0
        for ik in range(nk):
            summ = summ + 2*weights[ik]*inds.count(ik)
        ne = summ/sum(weights)
        return ne
    ind_VBM = int(np.ceil(ne_tot/2)) - 1
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
            print('ne:%s, norb+d_ne:%s, VBM:%s, CBM:%s' % (ne, norb+d_ne, VBM, CBM))
            if ne > ne_tot:
                CBM = e
            else:
                VBM = e
        return VBM

def ne_added_calc(eigs, weights, chem_pot, prec=0.0001):
    """
    calculate the number of added electrons when the chemical potential change chem_pot from charge neutral point

    NOTE: Charge neutral point: ne=norb
   
    args:
        eigs: eigen value array with shape (nk, norb), nk: kpoint number, norb: band (orbital) number 
        weithgs: a 1D array saving the weights of all kpoints
        chem_pot: the chemical potential relative to charge neutral point
    return:
        the added electron number, positive and negative values mean electron and hole doping
    """
    Ef = chem_pot_calc(eigs, weights, d_ne=0, prec=prec)
    chem_pot = chem_pot + Ef
    norb = len(eigs[0])
    nk = len(eigs)
    def get_ne(inds):
        summ = 0
        for ik in range(nk):
            summ = summ + 2*weights[ik]*inds.count(ik)
        ne = summ/sum(weights)
        return ne
    inds = np.where(eigs<=chem_pot)[0].tolist()
    return get_ne(inds)-norb


def dos_calc(eig_file='EIGEN_mesh.npz', save_to='DOS', sigma=0.01, nedos=10000, elim=None):
    """
    To caculate the dos from eigenvalues saved in eig_file using Gaussian smearing function
    Args:
        eig_file: the binary file saving the eigenvalues and eigenvectors output by numpy
        sigma: the width of Gaussion function to smearing the DOS
        nedos: the number of points for sampling the energy range
        save_to: the calculated dos will be saved into save_to file
    """
    from tBG.fortran.get_dos import get_dos, get_dos_fix_elim
    data = np.load(eig_file)
    weights = data['weights']
    try:
        eigs = data['vals']
    except:
        eigs = data['energies']
    if elim is not None:
        energies, dos = get_dos_fix_elim(elim[0], elim[1], weights, eigs, sigma, nedos)
    else:
        energies, dos = get_dos(weights, eigs, sigma, nedos)
    Ef = chem_pot_calc(eigs, weights, d_ne=0, prec=0.0001) 
    DOS = np.array([energies, dos]).T
    np.savetxt(save_to, DOS, header='Ef %s' % Ef)

###### plot ###############################################
def plot_dos(ax=None, eig_f='EIGEN_val_mesh.npz', sigma=0.01, nedos=10000, xlim=None,ylim=None, show=False, xy_change=False):
    dos_calc(eig_file=eig_f, save_to='DOS', sigma=sigma, nedos=nedos)
    plot = False
    if ax is None:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1,1)
        import tBG
        plt.rcParams.update(tBG.params)
        plot = True
    with open('DOS') as f:
        Ef = float(f.readline().split(' ')[-1])
    dos = np.loadtxt('DOS')
    if xy_change:
        ax.plot(dos[:,1], dos[:,0]-Ef, color='black')
    else:
        ax.plot(dos[:,0]-Ef, dos[:, 1], color='black')
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
        
    #ax.set_ylim([0, max(dos[:,1])*1.01])
    if plot:
        ax.set_xlabel('$Energy-E_f$ (eV)')
        ax.set_ylabel('Densit of states')
        if show:
            plt.show()
        else:
            plt.savefig('dos.pdf')
        plt.close()

def plot_band(eig_f='EIGEN_val_path.npz', fname='band.pdf', Ef=None, title='', title_size=40,ylim=None, show=False):
    eig = np.load(eig_f)
    kpts =eig['kpoints']
    dk = np.linalg.norm(kpts[1]-kpts[0])

    k_info = eig['k_info'][0]
    def labels_new():
        labels = []
        for i in k_info['labels']:
            if i=='G':
                labels.append('$\mathbf{\Gamma}$')
            else:
                labels.append('$%s$' % i)
        return labels
    labels = labels_new()

    norb = len(eig['vals'][0])
    if Ef is None:
        Ef = max(eig['vals'][:,int(norb/2)-1])
    enes = eig['vals']-Ef
    

    nk = len(kpts)
    ks = np.linspace(0,nk*dk, nk)

    from matplotlib import pyplot as plt
    from mpl_toolkits import axisartist
    import tBG
    plt.rcParams.update(tBG.params)
    fig = plt.figure(1)
    ax = axisartist.Subplot(fig, 111)
    fig.add_subplot(ax)
    for i in range(norb):
        ax.plot(ks, enes[:,i], color='black')
    for i in k_info['inds']:
        ax.axvline(ks[i], linestyle='dashed', color='black', linewidth=0.5)
    ax.axhline(0.0, linestyle='dashed', color='black',linewidth=0.5)
    ax.set_ylabel('$\mathbf{Energy-E_f}$ (eV)')
    ax.set_xticks([ks[i] for i in k_info['inds']])
    ax.set_xticklabels(labels)
    #ax.set_yticks([-0.6, -0.3, 0.0, 0.3, 0.6])
    #ax.set_title('$\mathbf{\\rightarrow x}$')
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlim(min(ks), max(ks))
    fig.suptitle(title, x=0.5, y=1.01, fontsize=title_size)
    plt.tight_layout(w_pad=0.01, h_pad=0.01)
    if show:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches='tight',pad_inches=0.2)
    plt.close()

def _ebs_path_calc(elim, vals, PMK, de=0.01):
    """
    Args:
        elim: the energy range
        de: the step width in energy sampling
        kpt_eig_pmk_f: the npz file saving kpoints, eigvals and pmk
    """
    from tBG.fortran.spec_func import get_ebs
    sigma = de/10

    es = [elim[0]+i*de for i in range(int(np.ceil((elim[1]-elim[0])/de)))]
    As = []
    for e in es:
        A = get_ebs(e, vals, PMK, sigma, de)
        As.append(A)
    return np.array(As)

def plot_ebs_path(ax=None, eig_f='EIGEN_val_vec_pmk_path.npz', fname='ebs.pdf', Ef=None, title='', title_size=40, ylim=[-3,3], show=False):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    data = np.load(eig_f, allow_pickle=True)
    kpts = data['kpoints']
    norb = len(data['vals'][0])
    if Ef is None:
        Ef = (max(data['vals'][:,int(norb/2)-1])+min(data['vals'][:,int(norb/2)]))/2
    vals = data['vals'] - Ef
    PMK = data['pmks']
    As = _ebs_path_calc(ylim, vals, PMK, de=0.01)
    k_info = data['k_info'][0]
    labels = k_info['labels']
    inds = k_info['inds']
    from matplotlib.colors import Normalize
    xmin = 0
    xmax = As.shape[1]
    ymin = ylim[0]
    ymax = ylim[1]
    colour_min = np.min(As)
    colour_max = np.max(As)
    return_ax = True
    if ax is None:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1,1)
        return_ax = False
    im = ax.imshow(As,origin='lower',aspect='auto',extent=[xmin,xmax,ymin,ymax],interpolation='bilinear',\
                cmap = 'RdBu',norm=Normalize(vmin=0,vmax=0.25),alpha=0.9)
    ax.set_ylim(ylim)
    ax.set_ylabel('Energy-E$\\rm{_f}$ (eV)')
    ax.set_xticks(inds)
    ax.set_xticklabels(labels)
    if return_ax:
        return
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='3%',pad=0.2)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.tight_layout(pad=0.1, h_pad=0.5, w_pad=None, rect=None)
    if show:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches='tight',pad_inches=0)
    plt.close()

def plot_ebs_mesh(e, de=0.01, sigma=None, Ef=0., eig_f='EIGEN_val_pmk_mesh.npz', fname='ebs_mesh.png', show=False):
    from tBG.fortran.spec_func import get_ebs
    def get_Nkx_Nky(kpoints):
        nx = 0
        k0y = kpoints[0][1]
        while True:
            kiy = kpoints[nx][1]
            if np.abs(kiy-k0y)<=1.e-4:
                nx = nx + 1
            else:
                break
        ny = len(kpoints)/nx
        return nx, int(ny)
    if sigma is None:
        sigma = de/100
    data = np.load(eig_f)
    kpts = data['kpoints']
    nx, ny = get_Nkx_Nky(kpts)
    kxmin, kymin = kpts[0]
    kxmax, kymax = kpts[-1]
    PMK = data['pmks']
    vals = data['vals'] - Ef
    As = get_ebs(e, vals, PMK, sigma, de)
    As = As.reshape(ny, nx)
    from matplotlib.colors import Normalize
    colour_min = np.min(As)
    colour_max = np.max(As)
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(1,1)
    im = ax.imshow(As,origin='lower',aspect='auto',extent=[kxmin,kxmax,kymin,kymax],interpolation='bilinear',\
                cmap = 'RdBu',norm=Normalize(vmin=0.0,vmax=0.25),alpha=0.9)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%',pad=0.2)
    ax.set_xlabel('$k_x $')
    ax.set_ylabel('$k_y $')
    ax.axis('equal')
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.tight_layout(pad=0.1, h_pad=0.5, w_pad=None, rect=None)
    if show:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches='tight',pad_inches=0)
    plt.close()

