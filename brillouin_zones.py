import numpy as np
from tBG.utils import *


def dist(p1, p2):
    """
    distance between p1 and p2
    """
    vec = np.array(p1)-np.array(p2)
    return np.linalg.norm(vec)

def square(p1,p2,p3):
    """
    area square spanned by p1, p1 and p3
    """
    s = p1[0]*p2[1] - p2[0]*p1[1] + p2[0]*p3[1] - p3[0]*p2[1] + p3[0]*p1[1] - p1[0]*p3[1]
    return 0.5*abs(s)

def _kpt_sampling_between_2nodes(k1, k2, dk=0.001, include_k2=True):
    k1 = np.array(k1)
    k2 = np.array(k2)
    norm = np.linalg.norm(k1-k2)
    dirt_vec = (k2-k1)/norm
    nsegment = int(norm/dk)
    dk = norm/nsegment * dirt_vec
    kpts = [k1+dk*i for i in range(nsegment+1)]
    if include_k2:
        return np.array(kpts)
    else:
        return np.array(kpts[:-1])

def kpoints_line_mode(kpts, dk=0.001):
    """
    args:
        kpts: the cart-coordinates of k-nodes
        dk: the sampling step 
    return:
        kpoints: all kpoints after sampling
        inds: the inds of the k-nodes in kpoints after sampling
    """
    kpts_sampling = [] # the kpoints after sampling
    inds = [] # the indice of the k-nodes
    n_node = len(kpts)
    for i in range(n_node-1):
        if i == n_node-2:
            include_k2 = True
        else:
            include_k2 = False
        kpts_i = _kpt_sampling_between_2nodes(kpts[i], kpts[i+1], dk=dk, include_k2=include_k2)
        kpts_sampling.append(kpts_i)
    npt_segment = np.array([len(i) for i in kpts_sampling])
    kpts_out = np.concatenate(kpts_sampling)
    inds = [np.sum(npt_segment[0:i]) for i in range(len(npt_segment))]
    inds.append(len(kpts_out)-1)
    return kpts_out, inds

def kpoints_line_mode_onek(kpt, dk=0.01, direction=[1,0], nstep=40):
    """
    Args:
        kpt: the cart-coordinate of the kpoint 
        dk: the sampling step
        direction: the direction for sampling
        nstep: the step number for sampling on two sides
    Return:
        the dense kpoints along all lines given by ks
    """
    kpt = np.array(kpt)
    k_step = dk*np.array(direction)/np.linalg.norm(direction)
    kpts = []
    for i in range(-nstep, nstep+1):
        k = kpt + i * k_step
        kpts.append(k)
    ind = nstep
    return np.array(kpts), ind

def kpoints_mesh_mode(cornors, k_mesh, shift=False):
    """
    args:
        cornors: [left_bott, right_bott, left_top, right_top] cart_coords of four cornors
        k_mesh: the mesh size
        shift: whether shift the mesh
    """
    ### get unit cell vectors
    left_bott, right_bott, left_top, right_top = np.array(cornors)
    latt_vec = np.array([right_bott-left_bott,left_top-left_bott])
    orig = left_bott
    ##################
    kpts_frac = []
    N0, N1 = k_mesh
    for n1 in range(N1):
        for n0 in range(N0):
            kpts_frac.append([n0/N0, n1/N1])
    kpts_frac = np.array(kpts_frac)
    if shift:
        kpts_frac = kpts_frac + np.array([0.5/N0, 0.5/N1])
    kpts_cart = np.array([frac2cart(i, latt_vec) for i in kpts_frac])
    return kpts_cart+orig

def kpoints_mesh_mode_onek(kpt, dk=0.01, nstep=20):
    dx = np.array([1,0])*dk
    dy = np.array([0,1])*dk
    kpts = []
    for i in range(-nstep, nstep+1):
        for j in range(-nstep, nstep+1):
            kpt_ij = kpt + i*dy + j*dx
            kpts.append(kpt_ij)
    kpts = np.array(kpts)
    return kpts

class BZMoirePattern:
    def __init__(self, latt_vec_bott, latt_vec_top, latt_vec_moire):
        self.bz_bott = BZHexagonal(latt_vec_bott)
        self.bz_top = BZHexagonal(latt_vec_top)
        self.bz_moire = BZHexagonal(latt_vec_moire)

    def kpoint_after_1st_scattering(self, kpt, in_layer):
        """
        return the end kpoint after strongest scattering for kpt in layer 'in_layer'
        kpt should be in the BZ of 'in_layer' layer
        """
        kpt = np.array(kpt)
        if in_layer=='bottom':
            rec_latt_vec_0 = self.bz_bott.recip_latt_vec
            rec_latt_vec_1 = self.bz_top.recip_latt_vec
            bz_0 = self.bz_bott
            bz_1 = self.bz_top
        elif in_layer=='top':
            rec_latt_vec_0 = self.bz_top.recip_latt_vec
            rec_latt_vec_1 = self.bz_bott.recip_latt_vec
            bz_0 = self.bz_top
            bz_1 = self.bz_bott
        if not bz_0.in_BZ(kpt):
            raise ValueError('kpt is not in BZ of %s layer' % in_layer)
        Gs_0 = np.array([rec_latt_vec_0[0], rec_latt_vec_0[1], np.sum(rec_latt_vec_0, axis=0)])
        Gs_0 = np.concatenate([Gs_0, -Gs_0, [[0., 0.]]])
        Gs_1 = np.array([rec_latt_vec_1[0], rec_latt_vec_1[1], np.sum(rec_latt_vec_1, axis=0)])
        Gs_1 = np.concatenate([Gs_1, -Gs_1, [[0., 0.]]])
        norms = np.round(np.linalg.norm(kpt+Gs_0, axis=1), 5)
        inds = np.where(norms==norms.min())[0]
        kpt_ends = []
        kpt_plus_Gs = []
        G0s = []
        G1s = []
        for ind in inds:
            G0 = Gs_0[ind]
            kpt_plus_G = kpt + G0
            kpt_end = bz_1.to_BZ(kpt_plus_G)
            G1 = kpt_plus_G - kpt_end
            G0s.append(G0)
            kpt_plus_Gs.append(kpt_plus_G)
            kpt_ends.append(kpt_end)
            G1s.append(G1)
        return kpt_ends

    def all_kpts_after_1st_scattering(self):
        Ks_bott = self.bz_bott.special_points()['K']
        Ks_top = self.bz_top.special_points()['K']
        K1s_to_top = []
        for K in Ks_bott:
            K1s = self.kpoint_after_1st_scattering(K, 'bottom')
            K1s_to_top.append(K1s)
        K1s_to_top = np.float32(np.concatenate(K1s_to_top))

        K1s_to_bott = []
        for K in Ks_top:
            K1s = self.kpoint_after_1st_scattering(K, 'top')
            K1s_to_bott.append(K1s)
        K1s_to_bott = np.float32(np.concatenate(K1s_to_bott))
        return np.unique(K1s_to_top, axis=0), np.unique(K1s_to_bott, axis=0)


    def kpath_K_K1tld_Gamma(self, dk=0.01):
        """
        The special path 
        """
        Ks_bott = self.bz_bott.special_points()['K']
        K = Ks_bott[3]
        K1_tld = self.kpoint_after_1st_scattering(K, 'bottom',plot=False)[0]
        kpts, inds = kpoints_line_mode([Ks_bott[2], K1_tld, [0,0]], dk)
        labels = ['$K$','$\widetilde{K}_1$', '$\Gamma$']
        return kpts, inds, labels

    def kpath_M_K_G_bottom(self, dk=0.01):
        K = self.bz_bott.special_points()['K'][0]
        M = self.bz_bott.special_points()['M'][0]
        G = [0.0, 0.0]
        kpts, inds = kpoints_line_mode([M, K, G], dk)
        labels = ['$M$','$K$','$\Gamma$']
        return kpts, inds, labels

    def kpath_between_Diracs(self, dk=0.01):
        K = self.bz_bott.special_points()['K'][2]
        K_t = self.bz_top.special_points()['K'][2]
        K_mid = 0.5*(K+K_t)
        K1s_t, K1s = self.all_kpts_after_1st_scattering()
        K1_ind = np.argmin(np.linalg.norm(K1s-K_mid, axis=1))
        K1 = K1s[K1_ind]
        K1_t_ind = np.argmin(np.linalg.norm(K1s_t-K_mid, axis=1))
        K1_t = K1s_t[K1_t_ind]
        kpts, inds = kpoints_line_mode([K, K1, K_t, K, K1_t, K_t], dk)
        labels = ['$K$', '$K_1$', '$\widetilde{K}$', '$K$', '$\widetilde{K}_1$', '$\widetilde{K}$']
        return kpts, inds, labels

    def kpoints_mesh_mode(self, k_mesh, one_forth=True):
        Ks_0 = self.bz_bott.special_points()['K']
        Ks_1 = self.bz_top.special_points()['K']
        Ks_all = np.concatenate([Ks_0, Ks_1])
        Kx_min, Ky_min = np.min(Ks_all, axis=0)
        Kx_max, Ky_max = np.max(Ks_all, axis=0)
        Kx_min = 0.
        Ky_min = 0.
        cornors = [[Kx_min,Ky_min], [Kx_max,Ky_min], [Kx_min,Ky_max], [Kx_max,Ky_max]]
        return kpoints_mesh_mode(cornors, k_mesh, shift=False)

    def get_all_Qs(self):
        def get_abc(p0, p1):
            """
            ax + by + c = 0
            """
            if p0[0]==p1[0]:
                a = 1
                b = 0
                c = -p0[0]
            else:
                a = -(p1[1]-p0[1])/(p1[0]-p0[0])
                b = 1
                c = -(a*p0[0]+p0[1])
            return a, b, c 

        def get_cross(p00, p01, p10, p11):
            a0, b0, c0 = get_abc(p00, p01)
            a1, b1, c1 = get_abc(p10, p11)
            if b0==0 or b1==0:
                y = -(c1/a1-c0/a0)/(b1/a1-b0/a0)
                x = -(b0/a0*y+c0/a0)
            else:
                x = -(c1/b1-c0/b0)/(a1/b1-a0/b0)
                y = -(a1/b1*x+c1/b1)
            return [x, y]

        Ks_bott = self.bz_bott.special_points()['K']
        Ks_top = self.bz_top.special_points()['K']
        Qs = []
        for i in range(6):
            K0_top = Ks_top[i%6]
            K1_top = Ks_top[(i+1)%6]

            K0_bot = Ks_bott[i%6]
            K1_bot = Ks_bott[(i+1)%6]
            K2_bot = Ks_bott[(i+2)%6]
            Qs.append(get_cross(K0_top, K1_top, K0_bot, K1_bot))
            Qs.append(get_cross(K0_top, K1_top, K1_bot, K2_bot))
        return np.array(Qs)

    def plot(self, ax, SBZ=True):
        size = 0
        #Qs = self.get_all_Qs()
        #ax.scatter(Qs[:,0], Qs[:,1], color='purple')
        #for i in range(12):
        #    ax.text(Qs[i][0], Qs[i][1], '$Q_{%s}$' % i, color='purple', fontsize=size)
        Ks_0 = self.bz_bott.special_points()['K']
        Ms_0 = self.bz_bott.special_points()['M']
        for i in range(len(Ks_0)):
            k = Ks_0[i]
            m = Ms_0[i]
            ax.text(k[0],k[1],'$K_{bot}^{%s}$' % i, color='blue', fontsize=size)
            ax.text(m[0],m[1],'$M_{bot}^{%s}$' % i, color='blue', fontsize=size)
        Ks_1 = self.bz_top.special_points()['K']
        Ms_1 = self.bz_top.special_points()['M']
        for i in range(len(Ks_1)):
            k = Ks_1[i]
            m = Ms_1[i]
            ax.text(k[0],k[1],'$K_{top}^{%s}$' % i, color='red', fontsize=size)
            ax.text(m[0],m[1],'$M_{top}^{%s}$' % i, color='red', fontsize=size)
        Ks_0 = np.append(Ks_0, [Ks_0[0]], axis=0)
        Ks_1 = np.append(Ks_1, [Ks_1[0]], axis=0)
        ax.plot(Ks_0[:,0], Ks_0[:,1], color='blue')
        ax.plot(Ks_1[:,0], Ks_1[:,1], color='red')
        if SBZ:
            Ks_all = np.concatenate([Ks_0, Ks_1])
            Kx_min, Ky_min = np.min(Ks_all, axis=0) 
            Kx_max, Ky_max = np.max(Ks_all, axis=0) 
            ### K points of moire BZ ###
            frac1 = [1/3., 2/3.]
            frac2 = [2/3., 1/3.]
            G11, G12 = self.bz_moire.recip_latt_vec[0]
            G21, G22 = self.bz_moire.recip_latt_vec[1]
            mat = np.array([[G11, G21],[G12, G22]])
            for frac in [frac1, frac2]:
                ms = []
                ns = []
                C1 = np.dot(frac, [G11, G21])
                C2 = np.dot(frac, [G12, G22])
                for A in [Kx_min, Ky_max]:
                    for B in [Ky_min, Ky_max]:
                        D1 = A - C1
                        D2 = B - C2
                        m,n = np.linalg.solve(mat, [D1, D2])
                        ms.append(int(m))
                        ns.append(int(n))
                m0 = min(ms)-1
                m1 = max(ms)+1
                n0 = min(ns)-1
                n1 = max(ns)+1
                fracs = [[i+frac[0],j+frac[1]] for i in range(m0, m1+1) for j in range(n0, n1+1)]
                carts = frac2cart(fracs, self.bz_moire.recip_latt_vec)
                cut = 0.5
                ind0 = np.where(carts[:,0]>=Kx_min-cut)
                ind1 = np.where(carts[:,0]<=Kx_max+cut)
                ind2 = np.where(carts[:,1]>=Ky_min-cut)
                ind3 = np.where(carts[:,1]<=Ky_max+cut)
                ax.set_xlim(Kx_min-0.25, Kx_max+0.25)
                ax.set_ylim(Ky_min-0.25, Ky_max+0.25)
                from functools import reduce
                inds = reduce(np.intersect1d, (ind0, ind1, ind2, ind3))
                carts = carts[inds]
                ax.scatter(carts[:,0], carts[:,1], color='black')
        ax.axis('equal')


class BZHexagonal:
    def __init__(self, latt_vec):
        self.latt_vec = latt_vec[0:2][:,0:2]
        self.recip_latt_vec = np.linalg.inv(self.latt_vec).transpose()*2*np.pi

    def special_points(self):
        Ms = [[0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [-0.5, 0.0], [-0.5, -0.5], [0.0, -0.5]]
        Ks = [[2/3, 1/3], [1/3, 2/3], [-1/3, 1/3], [-2/3, -1/3], [-1/3, -2/3], [1/3, -1/3]]
        G = [[0.0,0.0]]
        return {'M':frac2cart(Ms, self.recip_latt_vec), \
                'K':frac2cart(Ks, self.recip_latt_vec),
                'G':np.array(G)}

    def in_BZ(self, kpt, prec=1.e-4):
        """
        check whether kpt is in BZ
        """
        Ks = self.special_points()['K']
        square_BZ = np.abs(np.cross(self.recip_latt_vec[0], self.recip_latt_vec[1]))
        square_sum = 0.
        for i in range(5):
            K1, K2 = Ks[i], Ks[i+1]
            square_sum = square_sum + square(kpt, K1, K2)
        square_sum = square_sum + square(kpt, Ks[0], Ks[5])
        if square_sum <= square_BZ + prec:
            return True
        else:
            return False
    
    def to_BZ(self, kpt):
        """
        unfold the point kpt to BZ
        """
        def to_BZ_1D(pt):
            if pt>=-0.5 and pt<0.5:
                return pt
            elif pt>=0.5:
                return pt-1.0
            elif pt<-0.5:
                return pt+1.0

        def fold(kpt, G_vec):
            G_vec_norm = np.linalg.norm(G_vec)
            k_along_G = np.dot(kpt, G_vec)*G_vec/G_vec_norm**2
            k_vert_G = kpt - k_along_G
            frac = np.sign(np.dot(kpt, G_vec))*np.linalg.norm(k_along_G)/G_vec_norm
            if frac>=-0.5 and frac<0.5:
                return k_vert_G+ frac*G_vec, False
            frac_fold = to_BZ_1D(frac - round(frac,0))
            return k_vert_G+ frac_fold*G_vec, True

        if self.in_BZ(kpt):
            return kpt
        else:
            G1, G2 = self.recip_latt_vec
            G3 = G1 + G2

            run1 = True
            run2 = True
            run3 = True
            while run1 or run2 or run3:
                kpt1, run1 = fold(kpt, G1)
                kpt2, run2 = fold(kpt1, G2)
                kpt3, run3 = fold(kpt2, G3)
                kpt = kpt3
            return kpt
            
    def _get_kpoints_IBZ(self, kpts_cart, prec=1.e-6):
        kpts_ibz_cart = []
        weights = []
        
        ks_symm = self.special_points()
        G_cart = ks_symm['G'][0]
        M_cart = ks_symm['M'][1]
        K_cart = ks_symm['K'][1]
        d_GM = dist(G_cart, M_cart)
        d_GK = dist(G_cart, K_cart) 
        d_MK = dist(M_cart, K_cart) 
        s_GMK = square(G_cart, M_cart, K_cart)
        for i in range(len(kpts_cart)):
            k_cart = kpts_cart[i]
            d_kG = dist(k_cart, G_cart)
            d_kM = dist(k_cart, M_cart)
            d_kK = dist(k_cart, K_cart)
            s_kGM = square(k_cart, G_cart, M_cart)
            s_kGK = square(k_cart, G_cart, K_cart)
            s_kMK = square(k_cart, M_cart, K_cart)
            if d_kG < prec: # Gamma point
                kpts_ibz_cart.append(k_cart)
                weights.append(1)
            elif d_kM < prec: # M point
                kpts_ibz_cart.append(k_cart)
                weights.append(3)
            elif d_kK < prec: # K point
                kpts_ibz_cart.append(k_cart)
                weights.append(2)
            elif abs(d_kG +d_kK - d_GK)<prec: # Gamma-K
                kpts_ibz_cart.append(k_cart)
                weights.append(6)
            elif abs(d_kG +d_kM - d_GM)<prec: # Gamma-M
                kpts_ibz_cart.append(k_cart)
                weights.append(6)
            elif abs(d_kM +d_kK - d_MK) < prec: # M-K
                kpts_ibz_cart.append(k_cart)
                weights.append(6)
            elif abs(s_kGM + s_kGK + s_kMK - s_GMK) < prec: # in GMK 
                kpts_ibz_cart.append(k_cart)
                weights.append(12)
        return kpts_ibz_cart, weights

    def select_reciprocal_lattice_vectors(self, G_cutoff):
        """
        select the reciprocal lattice vectors inside the circle with G_cutoff radius
        """
        G_length = np.linalg.norm(self.recip_latt_vec[0])
        m = int(G_cutoff/(G_length*np.sqrt(3)/2))+1
        Gs = np.array([frac2cart([i,j], self.recip_latt_vec) for i in range(-m, m+1) for j in range(-m, m+1)])
        Gs_norm = np.linalg.norm(Gs, axis=1)
        inds = np.where(Gs_norm<=G_cutoff)[0]
        return Gs[inds]
        
    def kpoints_mesh_mode(self, k_mesh, method='Gamma', symm=True):
        left_bott = [0,0]
        right_bott, left_top = self.recip_latt_vec
        right_top = right_bott + left_top
        cornors = [left_bott, right_bott, left_top, right_top]
        if method == 'Gamma':
            shift = False
        elif method == 'MP':
            shift = True
        kpts_cart = kpoints_mesh_mode(cornors, k_mesh, shift)
        weights = [1]*len(kpts_cart)
        if symm:
            kpts_cart, weights =  self._get_kpoints_IBZ(kpts_cart)
        return np.array(kpts_cart), np.array(weights)             

    def kpoints_line_mode_onek(self, k_label='K', dk=0.001, direction=[1,0], nstep=40):
        """
        Args:
            k_label: the kpoint nodes for the line mode 
            dk: the kpoints distance for cutting the kpoint lines 
            direction: the direction for k 
            nstep: the step number k point going
        Return:
            the dense kpoints along all lines given by ks
        """
        ks_symm = self.special_points()
        k_cart = ks_symm[k_label][0]
        return kpoints_line_mode_onek(k_cart, dk, direction, nstep)

    def kpoints_line_mode(self, k_path=['G','M','K','G'], dk=0.01):
        ks_symm = self.special_points()
        ks_node = [ks_symm[i][0] for i in k_path]
        return kpoints_line_mode(ks_node, dk)

    
    
    def _plot(self, ax, color='black'):
        ks_symm = self.special_points()
        ax.plot(np.append(ks_symm['K'][:,0], [ks_symm['K'][0][0]]), \
                 np.append(ks_symm['K'][:,1], [ks_symm['K'][0][1]]), color=color)
    
    def _plot_BZ(self, k_path=['G','M','K','G'], dots=True):
        from matplotlib import pyplot as plt
        ks_symm = self.special_points()
        plt.plot(np.append(ks_symm['K'][:,0], [ks_symm['K'][0][0]]), \
                 np.append(ks_symm['K'][:,1], [ks_symm['K'][0][1]]), color='black')
        if True:
            plt.plot([0,self.recip_latt_vec[0][0]],[0,self.recip_latt_vec[0][1]])
            plt.plot([0,self.recip_latt_vec[1][0]],[0,self.recip_latt_vec[1][1]])
        if dots:
            for i in range(6):
                plt.scatter(ks_symm['K'][i][0], ks_symm['K'][i][1], marker='o', color='black')
                plt.text(ks_symm['K'][i][0], ks_symm['K'][i][1], str(i))
                plt.scatter(ks_symm['M'][i][0], ks_symm['M'][i][1], marker='s', color='black')
                plt.text(ks_symm['M'][i][0], ks_symm['M'][i][1], str(i))
        if k_path:
            ks_node = np.array([ks_symm[i][0] for i in k_path])
            plt.plot(ks_node[:,0],ks_node[:,1])
        plt.axis('equal')
        plt.show()
        #plt.savefig('BZ.pdf', bbox_inches='tight', pad_inches=0)
        #plt.clf()
