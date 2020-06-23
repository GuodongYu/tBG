import numpy as np
from scipy.linalg.lapack import zheev
import time
import matplotlib as mpl
from tBG.fortran.spec_func import get_pk, get_ebs
from tBG.TBG30_approximant.diag import Hamilt
from tBG.TBG30_approximant.structure import  BZs

class BandUnfold(Hamilt, BZs):
    def __init__(self, struct, elec_field=0.0):
        Hamilt.__init__(self, struct, elec_field=elec_field)
        BZs.__init__(self, struct)
        self.struct = struct
        self.coords = struct.coords
        self.natom = np.sum(struct.layer_nsites)
        self.norb = self.natom
        self.disorder = {'Anderson':False, 'off_diag':False}

    def add_Ansderon_disorder(self, W):
        nsite = np.sum(self.struct.layer_nsites)
        self.disorder['Anderson'] = np.random.uniform(-W,W,nsite)

    def add_off_diag_disorder(self, theta):
        pass

    def Hk_disorder(self, k):
        Hk = self._Hk(k)
        if self.disorder['Anderson'] is not False:
            Hk = Hk + np.diag(self.disorder['Anderson'])
        return Hk
                 
    def folded_band_struct(self, dk=0.01):
        save_to = 'FOLDED_BAND'
        f = open(save_to, 'w')
        f.write('#kx'+'\t'+'ky'+'\t'+'vals'+'\n')
        ks = self.kpoints_line_mode(dk=dk)
        for k in ks:
            Hk = self.Hk_disorder(k)
            vals, vecs, info = zheev(Hk, 0)
            if info:
                raise ValueError('zheev false') 
            f.write(str(k[0])+'\t'+str(k[1])+'\t'+'\t'.join(map(str,vals))+'\n')
        f.close()

    def PMK_calc(self, ks_cart, k_info=None, fname='KPT_EIG_PMK'):
        """
        calculate all the eigen states adn values at all Ks of the SC
        corresponding to the given ks_cart, which means K = k + G
        G is a reciprocal lattice vector and k + G in the BZ of SC
        """
        PMK = []
        eigvals = []
        i = 0
        for k in ks_cart:
            t0 = time.time()
            Hk = self.Hk_disorder(k)
            val, vec, info = zheev(Hk, 1)
            if info:
                raise ValueError('zheev failed!')
            eigvals.append(val)
            Pk = get_pk(k, np.array(self.struct.layer_nsites)/2, [1,1], 2, 2, vec, self.coords, self.struct.species)
            
            PMK.append(Pk)
            t1 = time.time()
            print('ik=%s/%s %s s' % (i+1,len(ks_cart), (t1-t0)))
            i += 1
        np.savez_compressed(fname, kpoints=ks_cart, k_info=[k_info], eigvals=eigvals, pmk=PMK)

    def kpoints_line_onek(self, k_D='K_bot', dk = 0.01, direction=[1,0], nstep=40):
        """
        Args:
            k_nodes: the kpoint nodes for the line mode 
            dk: the kpoints distance for cutting the kpoint lines 
            PMK_run: if the PMK will be run
            kws: if k_nodes only has one point, the kpoints around this point will be calculate 
                 kws['direction'] and kws['nstep'] should be given.
                 kws['direction'] = [1,0] means the line along kx direction and kws['nstep'] kpoints 
                                   will be included at two sides.
        Return:
            the dense kpoints along all lines given by ks
        """
        ks_symm = self._spectial_kpoints()
        k_D_cart = ks_symm[k_D]
        k_step = dk*np.array(direction)/np.linalg.norm(direction)
        kpts = []
        for i in range(-nstep, nstep+1):
            k = k_D_cart + i * k_step
            kpts.append(k)
        return np.array(kpts), {'labels':[k_D], 'inds':[nstep], 'dks':[dk], 'type':'EBS_onek'}
                
    def kpoints_line_mode(self, k_path=['M_bot_orig','K_bot','M_top','KR_bot', 'D_bot','G','D_top',\
                                        'KR_top','M_bot','K_top','M_top_end'], dk = 0.01):
        """
        Args:
            k_nodes: the kpoint nodes for the line mode 
            dk: the kpoints distance for cutting the kpoint lines 
            PMK_run: if the PMK will be run
            kws: if k_nodes only has one point, the kpoints around this point will be calculate 
                 kws['direction'] and kws['nstep'] should be given.
                 kws['direction'] = [1,0] means the line along kx direction and kws['nstep'] kpoints 
                                   will be included at two sides.
        Return:
            the dense kpoints along all lines given by ks
        """
        ks_symm = self._spectial_kpoints()
        ks_node = [ks_symm[i] for i in k_path]
        n_range = len(ks_node) - 1

        kpts = [] # the kpoints after sampling
        dks = [] # the real dk in each path
        inds = [] # the index of each in k_nodes
        for i in range(n_range):
            inds.append(len(kpts))
            length = np.linalg.norm(ks_node[i+1]-ks_node[i])
            nbin = int(round(length/dk))
            if nbin < 1:
                nbin = 1
            dki = length/nbin
            dks.append(dki)
            dki_vec = dki*(ks_node[i+1]-ks_node[i])/length
            for j in range(nbin):
                k = ks_node[i] + j*dki_vec
                kpts.append(k)
        kpts.append(ks_node[-1])
        inds.append(len(kpts)-1)
        return np.array(kpts), {'labels':k_path, 'inds':inds, 'dks':dks, 'type':'EBS_path'}

    def kpoints_mesh_mode(self, dk=0.005, loc='K_bot', size=10):
        sp_ks = self._spectial_kpoints()
        K = sp_ks[loc] 
        ks = []
        for i in range(-size, size+1):
            for j in range(-size, size+1):
                k = K +np.array([i*dk, j*dk])
                ks.append(k)
        ks = np.array(ks)
        #if PMK_run:
        #    return self.PMK_calc(ks, label='ARPES_%s' % loc)
        return ks, None

    def get_VBM_CBM(self, check='K_top'):
        K = self._spectial_kpoints()[check]
        Hk = self._Hk(K)
        val0, vec0, info = zheev(Hk, 0)
        ind_mu = int(np.ceil(self.norb/2))-1
        return {'VBM':val0[ind_mu], 'CBM':val0[ind_mu+1]}

    def get_bandgap_at_M(self, check='M_top'):
        K = self._spectial_kpoints()[check]
        Hk = self._Hk(K)
        val, vec, info = zheev(Hk, 0)
        if info:
            raise ValueError('zheev failed!')
        ind_mu = int(np.ceil(self.norb/2))-1
        return val[ind_mu]-val[ind_mu-1]

    def fermi_velocity(self, where='K_top', dk=0.01, nk=5):
        """
        Calculate the Fermi velocity from folded band structur
        where: one of 'K_top', 'K_bot', 'KR_top', 'KR_bot'
        """
        def f2e(x):
            y = '%.2e' % x
            x, y = y.split('e')
            y = int(float(y))
            return '%sx10^{%s}' % (x, y)

        hbar = 6.58211951e-16 # eVs
        Ang2m = 1.e-10 
        pnts = self._spectial_kpoints()

        ### sample the kpoints around Dirac point and diag H(K)
        K = pnts[where]
        kxs = []
        kys = []
        kx_s = []
        ky_s = []
        ind_mu = int(np.ceil(self.norb/2))-1
        for i in range(1,nk+1):
            kxs.append(K+np.array([dk*i, 0]))
            kys.append(K+np.array([0, dk*i]))
            kx_s.append(K+np.array([-dk*i, 0]))
            ky_s.append(K+np.array([0, -dk*i]))
        Hk0 = self._Hk(K)
        val0, vec0, info = zheev(Hk0, 0)
        from matplotlib import pyplot as plt
        fig, axs = plt.subplots(2,2)
        axs = [axs[0][0], axs[0][1], axs[1][0], axs[1][1]]
        tits = ['kx','ky','-kx','-ky']
        ####################################################
        ## get_VB and get_CB can get the correct VB and CB energy 
        ## on the Dirac cones
        def get_VB(k_dist, val, slope, VBM):
            dist = k_dist
            slope_diff = abs(abs(val[ind_mu]-val0[ind_mu])/dist -slope)
            e0 = val[ind_mu]
            ind = ind_mu - 1
            while True:
                e = val[ind]
                slope_k = abs((e-VBM)/dist)
                slope_diff_new = abs(slope_k-slope)
                if slope_diff_new > slope_diff:
                    return e0
                slope_diff = slope_diff_new
                e0 = e
                ind = ind-1
        def get_CB(k_dist, val, slope, CBM):
            dist = k_dist
            slope_diff = abs(abs(val[ind_mu+1]-val0[ind_mu+1])/dist -slope)
            e0 = val[ind_mu+1]
            ind = ind_mu+2
            while True:
                e = val[ind]
                slope_k = abs((e-CBM)/dist)
                slope_diff_new = abs(slope_k-slope)
                if slope_diff_new>slope_diff:
                    return e0
                slope_diff = slope_diff_new
                e0 = e
                ind = ind+1
        ###############################################
        i = 0
        vfs = []   
        for ks in [kxs, kys, kx_s, ky_s]:
            print('Running along  %s direction...' % tits[i])
            VBM = val0[ind_mu]
            CBM = val0[ind_mu+1]
            ### collect all bands and VBM CBM
            vals = []
            vals.append(val0)
            ks_dist = []
            ks_dist.append(0.0)
            for k in ks:
                Hk = self._Hk(k)
                val, vec, info = zheev(Hk, 0)
                if val[ind_mu] > VBM:
                    VBM = val[ind_mu]
                if val[ind_mu+1] < CBM:
                    CBM = val[ind_mu+1]
                vals.append(val)
                k_dist = np.linalg.norm(k-K)
                ks_dist.append(k_dist)
            vals = np.array(vals)
                
            ### pick up four Dirac cones 
            VB = [vals[0][ind_mu], vals[1][ind_mu]]
            CB = [vals[0][ind_mu+1], vals[1][ind_mu+1]]
            slope_VB = abs((vals[1][ind_mu]-VBM)/dk)
            slope_CB = abs((vals[1][ind_mu+1]-CBM)/dk)
            for ik in range(2,len(ks_dist)):
                k_dist = ks_dist[ik]
                val = vals[ik]
                VB.append(get_VB(k_dist, val, slope_VB, VBM))
                CB.append(get_CB(k_dist, val, slope_CB, CBM))
            axs[i].plot(ks_dist, VB, color='red', linestyle='dashed', linewidth=2.)
            axs[i].plot(ks_dist, CB, color='red', linestyle='dashed', linewidth=2.)
            slope_avg_VB = np.polyfit(ks_dist, [i-VBM for i in VB], 1)[0]
            slope_avg_CB = np.polyfit(ks_dist, [i-CBM for i in CB], 1)[0]
            vf_VB = abs(slope_avg_VB/hbar * Ang2m)
            vf_CB = abs(slope_avg_CB/hbar * Ang2m)
            vf_avg = (vf_VB+vf_CB)/2
            vfs.append(vf_avg)
            axs[i].text(0.3, 0.4, '$v^{VB}_f= %s m/s$' % f2e(vf_VB),transform=axs[i].transAxes)
            axs[i].text(0.3, 0.55, '$v^{CB}_f= %s m/s$' % f2e(vf_CB),transform=axs[i].transAxes)
            axs[i].text(0.01, 0.8, '$v^{avg}_f= %s m/s$' % f2e(vf_avg),transform=axs[i].transAxes)
            axs[i].text(0.1, 0.1, '%s' % tits[i], transform=axs[i].transAxes)

            ### plot band
            for j in range(self.norb):
                axs[i].plot(ks_dist, vals[:,j], marker='o', color='black', linewidth=1.)
            if VBM > CBM:
                print('Warning: VBM > CBM. Not a semiconductor!!!')
            axs[i].set_ylim((VBM-0.3, VBM+0.3))
            axs[i].axhline(VBM, linestyle='dashed', linewidth=1.0)
            axs[i].set_ylabel('Energy (eV)')
            i = i + 1
        vf = np.average(vfs)
        x,y =where.split('_')
        fig.suptitle('$%s_{%s} \t v_f=%s m/s$' % (x,y, f2e(vf)))
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        plt.savefig('Ek_%s.pdf' % where)
        plt.clf()

def unfolded_band_struct(elim, de=0.01, kpt_eig_pmk_f=None, fname='EBS'):
    """
    Args:
        elim: the energy range
        de: the step width in energy sampling
        kpt_eig_pmk_f: the npz file saving kpoints, eigvals and pmk
    """
    data = np.load(kpt_eig_pmk_f)
    ks = data['kpoints']
    eigvals = data['eigvals']
    PMK = data['pmk']
    k_info = data['k_info'][0]

    sigma = de/10

    es = [elim[0]+i*de for i in range(int(np.ceil((elim[1]-elim[0])/de)))]
    As = []
    for e in es:
        A = get_ebs(e, eigvals, PMK, sigma, de)
        As.append(A)
    np.savez_compressed(fname, energies=es, kpoints=ks, k_info=[k_info], EBS=As)

def get_ARPES_map(e, ks, eigvals, PMK, label, de=0.01):
    if type(eigvals) is str:
        ks = np.loadtxt(ks)
        eigvals = np.loadtxt(eigvals)
        PMK = np.loadtxt(PMK)
    nk = len(ks)
    sigma = de/100
    nb = len(eigvals[0])
    g = open('ARPES_%s' % label, 'w')
    g.write('#nk=' +'\t' + str(nk)+' energy:'+'\t'+str(e)+'n')
    g.write('#kx'+'\t'+'ky'+'\t'+'deltaN'+'\n')
    As = get_ebs(e, eigvals, PMK, sigma, de)
    for i in range(len(ks)):
        g.write(str(ks[i][0])+'\t'+str(ks[i][1])+'\t'+str(As[i])+'\n')
    g.close()

def plot_ARPES_map(ax=None, ARPES_f='', scale=1., interpolation='nearest'):
    data = np.loadtxt(ARPES_f)
    kxs = data[:,0]
    kys = data[:,1]
    nx = len(set(kxs))
    ny = len(set(kys))
    dx = kxs[ny]-kxs[0]
    dy = kys[1]-kys[0]
    #As = np.float16(np.abs(data[:,-1]))*scale
    As = data[:,-1]*scale
    X = np.reshape(kxs, (nx, ny))
    Y = np.reshape(kys, (nx, ny))
    C = np.reshape(As, (nx, ny))
    if not ax:
        from matplotlib import pyplot as plt
        fig, (ax) = plt.subplots(nrows=1)
        ret = False
    else:
        ret = True
    from matplotlib.ticker import MaxNLocator
    from matplotlib.colors import BoundaryNorm
    cmap = mpl.pyplot.get_cmap('inferno')
    levels = MaxNLocator(nbins=100).tick_values(0, 2)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    if interpolation:
        im = ax.imshow(C.T, cmap=cmap, interpolation = interpolation, norm=norm, extent=(0, 1, 0, 1))
    else:
        im = ax.pcolormesh(X, Y, C, cmap=cmap, norm=norm)
        ax.set_xlim(min(kxs), max(kxs))
        ax.set_ylim(min(kys), max(kys))
    #fig.colorbar(im, ax)
    #if scale!=1.0:
    #    ax.text(0.5, 0.5, '$\\times%s$' % scale, color='white', fontsize=40, \
    #        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    #ax.margins(0.0)
    ax.tick_params(axis="y",direction="in", pad=-22, labelsize=0.)
    ax.tick_params(axis="x",direction="in", pad=-15, labelsize=0.)
    ax.axis('equal')
    ax.axis('off')
    ax.set_adjustable('box-forced')
    if ret:
        return ax
    plt.savefig('%s.png' % ARPES_f, bbox_inches='tight', pad_inches=0)
    plt.clf()
       
def plot_folded_band(ylim=None, plt=None, ret=False):
    if not ret:
        from matplotlib import pyplot as plt
    fold_band_f = 'FOLDED_BAND'
    data = np.loadtxt(fold_band_f)
    kx = data[:,0]
    ky = data[:,1]
    nk = len(kx)
    dk = np.linalg.norm([kx[1]-kx[0],ky[1]-ky[0]])
    vals = data[:,2:]
    nb = len(vals[0]) 
    ks = [dk*i for i in range(nk)]
    for i in range(nb):
        plt.plot(ks, vals[:,i],c='black',linewidth=0.1)
    if ylim:
        plt.ylim(ylim)
    if ret:
        return plt
    plt.xlabel('wavevector')
    plt.ylabel('energy(eV)')
    plt.savefig('folded_band_struct.pdf')
    plt.clf()

def band_comparison(ylim=None):
    from matplotlib import pyplot as plt
    plt = plot_unfolded_band(plt, ret=True)
    plt = plot_folded_band(ylim, plt, ret=True)
    plt.savefig('band_comparison.pdf')
    plt.clf()

