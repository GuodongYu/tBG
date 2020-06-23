import numpy as np
import numpy.linalg as npla
from scipy.linalg.lapack import zheev
import math
import json
import time
import sys
import tipsi
import copy
from monty.json import jsanitize
import struct
from scipy.interpolate import interp2d
import matplotlib as mpl

def frac2cart(frac, latt_vec):
    return np.dot(np.transpose(np.array(latt_vec)), frac)

class Hamilt(object):
    """
    setup the Hamiltian matrix given struct with hoppings
    """
    def __init__(self, struct, elec_field=0.0):
        """
        Unit:
            length:  angstrom
            elec_field: eV/angstrom
        struct: the Structure object
        elec_field: the electric field along z axis 
        """
        self.struct = struct
        self.coords = struct.coords
        self.natom = len(self.coords)
        self.norb = self.natom
        self.latt_comp = struct.latt_compatible
        self.H0 = self._get_H0(elec_field=elec_field)

    def _get_H0(self, elec_field=0.0):
        H0 = np.zeros((self.norb, self.norb), dtype=complex)
        layer_inds = self.struct._layer_inds()
        if elec_field:
            for i in range(len(layer_inds)):
                z = self.struct.layer_zcoords[i]
                for j in range(layer_inds[i][0], layer_inds[i][1]+1):
                    H0[j,j] = z*elec_field
        hopping = self.struct.hopping['0_0']
        for pair in hopping:
            i,j = [int(w) for w in pair.split('_')]
            H0[i,j] = hopping[pair]
            H0[j,i] = hopping[pair]
        return H0

    def _Hk(self, k):
        """
        k: kpoint in cartesian coord
        """
        a0 = self.latt_comp[0]
        a1 = self.latt_comp[1]
        Hk = copy.deepcopy(self.H0)
        for neig_cell in ['0_1','1_0','1_1','-1_1']:
            m,n = [int(i) for i in neig_cell.split('_')]
            R = m*a0 + n*a1
            hopping = self.struct.hopping[neig_cell]
            for pair in hopping:
                i,j = [int(i) for i in pair.split('_')]
                R_dot_k = np.dot(k, R)
                Hk[i,j] = Hk[i,j] + np.exp(1j*R_dot_k) * hopping[pair]
                Hk[j,i] = Hk[j,i] + np.exp(-1j*R_dot_k) * hopping[pair]
        return Hk

class DiagSample(Hamilt):
    def __init__(self, struct, elec_field=0.0):
        Hamilt.__init__(self, struct, elec_field=elec_field)
        self.reciprocal_vecs = np.linalg.inv(self.latt_comp).transpose()*2*np.pi
        self.Gamma = [0., 0.]
        self.M = [0.5, 0.]
        self.K = [2./3, 1./3]    
        self._symm_ks = dict(zip(['G','K','M'], [frac2cart(i,self.reciprocal_vecs) \
                                             for i in [self.Gamma, self.K, self.M]]))   

    def add_kpoints_BZ(self, k_mesh, method):
        self.k_mesh = k_mesh
        self.k_mesh_method = method
        kpts = []
        N0, N1 = k_mesh
        for n0 in range(N0):
            for n1 in range(N1):
                kpts.append([n0/N0, n1/N1])
        kpts = np.array(kpts)
        if method == 'Gamma':
            kpoints = kpts
        elif method == 'MP':
            kpoints = kpts + np.array([0.5/N0, 0.5/N1])
        self.kpoints_frac = np.array(kpoints)
        self.kpoints_cart = np.array([frac2cart(i, self.reciprocal_vecs) for i in kpoints])
        self._get_kpoints_IBZ()
             
    def _get_kpoints_IBZ(self, prec=1.e-6):
        #kpts = [i for i in self.kpoints if i[0]<=self.K[0] and i[1]<=self.K[1]]
        kpts_frac = self.kpoints_frac
        kpts_cart = self.kpoints_cart
        kpts_ibz_cart = []
        kpts_ibz_frac = []
        weights = []
        G_cart = frac2cart(self.Gamma, self.reciprocal_vecs)
        K_cart = frac2cart(self.K, self.reciprocal_vecs)
        M_cart = frac2cart(self.M, self.reciprocal_vecs)
        def dist(p1, p2):
            vec = np.array(p1)-np.array(p2)
            return np.linalg.norm(vec)
        def square(p1,p2,p3):
            s = p1[0]*p2[1] - p2[0]*p1[1] + p2[0]*p3[1] - p3[0]*p2[1] + p3[0]*p1[1] - p1[0]*p3[1]
            return 0.5*abs(s)
        d_GM = dist(G_cart, M_cart)
        d_GK = dist(G_cart, K_cart) 
        d_MK = dist(M_cart, K_cart) 
        s_GMK = square(G_cart, M_cart, K_cart)
        for i in range(len(kpts_frac)):
            k_cart = kpts_cart[i]
            k_frac = kpts_frac[i]
            d_kG = dist(k_cart, G_cart)
            d_kM = dist(k_cart, M_cart)
            d_kK = dist(k_cart, K_cart)
            s_kGM = square(k_cart, G_cart, M_cart)
            s_kGK = square(k_cart, G_cart, K_cart)
            s_kMK = square(k_cart, M_cart, K_cart)
            if d_kG < prec: # Gamma point
                kpts_ibz_cart.append(k_cart)
                kpts_ibz_frac.append(k_frac)
                weights.append(1)
            elif d_kM < prec: # M point
                kpts_ibz_cart.append(k_cart)
                kpts_ibz_frac.append(k_frac)
                weights.append(3)
            elif d_kK < prec: # K point
                kpts_ibz_cart.append(k_cart)
                kpts_ibz_frac.append(k_frac)
                weights.append(2)
            elif abs(d_kG +d_kK - d_GK)<prec: # Gamma-K
                kpts_ibz_cart.append(k_cart)
                kpts_ibz_frac.append(k_frac)
                weights.append(6)
            elif abs(d_kG +d_kM - d_GM)<prec: # Gamma-M
                kpts_ibz_cart.append(k_cart)
                kpts_ibz_frac.append(k_frac)
                weights.append(6)
            elif abs(d_kM +d_kK - d_MK) < prec: # M-K
                kpts_ibz_cart.append(k_cart)
                kpts_ibz_frac.append(k_frac)
                weights.append(6)
            elif abs(s_kGM + s_kGK + s_kMK - s_GMK) < prec: # in GMK 
                kpts_ibz_cart.append(k_cart)
                kpts_ibz_frac.append(k_frac)
                weights.append(12)
        self.kpoints_IBZ = {'kpoints_cart':kpts_ibz_cart, 'weights':weights, 'kpoints_frac':kpts_ibz_frac}

    def diag_run(self, kpts=None, symmetry=True, vec_calc=False, chg_calc=True):
        """
        kpts: the kpoints in cart coord, if None, added kpoints will be choosen
        symmetry: whether take symmetry into account
        vec_calc: True or False, whether calculate eigen vectors
        chg_calc: True or False, whether calculate charges 
        """
        if kpts is None:
            kpts = self.kpoints_cart
            weights = [1]*len(kpts)
            if symmetry:
                kpts = self.kpoints_IBZ['kpoints_cart']
                weights = self.kpoints_IBZ['weights']
        else:
            weights = [1]*len(kpts)
        if vec_calc or chg_calc:
            v = 1
        else:
            v = 0
        vals = []
        vecs = []
        chgs = []
        ik=0
        for k in kpts:
            t0 = time.time()
            Hk = self._Hk(k)
            val, vec, info = zheev(Hk, v)
            if info:
                raise ValueError('zheev error')
            vals.append(val)
            if chg_calc:
                chg =np.zeros((len(val),len(val)))
                for i in range(len(val)):
                    chg[:,i] = np.square(np.absolute(vec[:,i]))
                chgs.append(chg)
            if vec_calc:
                vecs.append(vec)
            t1 = time.time()
            print('%s/%s %.3fs' % (ik+1, len(kpts), (t1-t0)))
            ik = ik + 1
        np.savez_compressed('EIGEN', struct=[self.struct], \
                             kpoints=kpts, weights=weights, vals=vals, vecs=vecs, chgs=chgs)


