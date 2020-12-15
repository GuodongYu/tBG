import numpy as np
from pymatgen.core.structure import Structure as pmg_struct
from scipy.linalg.lapack import zheev
import copy
from tBG.utils import frac2cart

def get_append_dict(stack):
    """
    get the append dictionary of a multilayer stack
    as input parameter of function self.append_layers() 
    """
    stack = stack.replace('Atld', 'C')
    stack = stack.replace('At', 'C')
    stack = stack.replace('Btld', 'D')
    stack = stack.replace('Bt', 'D')
    for i in range(len(stack)):
        if stack[i]=='A' and stack[i+1]=='C':
            ind = i
            break
    ids = np.array(range(len(stack))) - ind
    append = {}
    for i in range(len(stack)):
        if i in [ind, ind+1]:
            continue
        layer = stack[i]
        if layer == 'C':
            layer_ = 'Atld'
        elif layer == 'D':
            layer_ = 'Btld'
        else:
            layer_ = layer
        try:
            append[layer_].append(ids[i])
        except:
            append[layer_] = [ids[i]]
    return append

class _LayeredStructMethods:
    def pymatgen_struct(self):
        try:
            return pmg_struct(self.latt_vec, ['C']*len(self.coords), self.coords, coords_are_cartesian=True)
        except:
            coords = copy.deepcopy(self.coords)
            nsite  = len(coords)
            x_min, y_min, z_min = np.min(coords, axis=0)
            x_max, y_max, z_max = np.max(coords, axis=0)
            coords[:,0] = coords[:,0] - x_min + 10
            coords[:,1] = coords[:,1] - y_min + 10
            coords[:,2] = coords[:,2] - z_min + 10
            latt_vec = np.array([[x_max-x_min+20, 0, 0],[0, y_max-y_min+20, 0],[0, 0, z_max-z_min+20]])
            species = ['C']*len(self.coords)
            return  pmg_struct(latt_vec, species, coords, coords_are_cartesian=True)
    def pymatgen_molecule(self):
        from pymatgen.core.structure import Molecule
        species = ['C']*len(self.coords)
        return Molecule(species, self.coords)

    def get_layer_vecs_to_NNs(self):
        """
        get the vecs to the nearest neighbors
        the vecs are used to add interlayer hopping between wannier functions
        see paper PRB 93 235153 (2016)
        """
        nlayer = len(self.layer_nsites)
        vecs_site0 = [np.sum(self.layer_latt_vecs[i], axis=0)/3. for i in range(nlayer)]
        return np.array([[i, -i] for i in vecs_site0])

    def species(self):
        specs = []
        for layer in self.layer_nsites_sublatt:
            n_s0, n_s1 = layer
            specs.append([1]*n_s0)
            specs.append([2]*n_s1)
        return np.concatenate(specs)

    def _layer_inds(self):
        layer_inds = []
        for i in range(len(self.layer_nsites)):
            ind0 = sum(self.layer_nsites[:i])
            ind1 = ind0 + self.layer_nsites[i]-1
            layer_inds.append([ind0,ind1])
        return layer_inds

    def _layer_inds_sublatt(self):
        layer_inds = self._layer_inds()
        layer_nsite_sublatt = self.layer_nsites_sublatt
        out = [[[],[]] for _ in range(len(layer_inds))]
        for i in range(len(layer_inds)):
            out[i][0] = [layer_inds[i][0],layer_inds[i][0]+layer_nsite_sublatt[i][0]-1]
            out[i][1] = [layer_inds[i][0]+layer_nsite_sublatt[i][0],layer_inds[i][1]]
        return out

    def append_layers(self, layers):
        """
        Add new layers (A, B, Atld or Btld) into the system.
        Note: A and Atld layers already exist at 0 and 1*h 

        layers: a dictory for the appended layers { layer_type: positions list, }
        such as {'A': [-1, -3], 'B':[-2, -4], 'Atld':[2, 4], 'Btld':[3,5]}
                -1*h and -3*h  add A layer
                -2*h and -4*h  add B layer
                 2*h and 4*h add A_tilde layer
                 3*h and 5*h add B_tilde layer
        """

        def _coords_xy(layer):
            """
            layer: 'A', 'B', 'Atld' or 'Btld'
            """
            if layer in ['A','B']:
                nsite_bott = self.layer_nsites[0]
                latt_vec_bott = self.latt_vec_bott[0:2,0:2]
                xy = copy.deepcopy(self.coords[0:nsite_bott][:,0:2])
                return xy if layer=='A' else xy + 1/3*(latt_vec_bott[0]+latt_vec_bott[1])
            elif layer in ['Atld','Btld']:
                nsite_bott = self.layer_nsites[0]
                nsite_top = self.layer_nsites[1]
                latt_vec_top = self.latt_vec_top[0:2,0:2]
                xy = copy.deepcopy(self.coords[nsite_bott:nsite_bott+nsite_top][:,0:2])
                return xy if layer == 'Atld' else xy + 1/3.*(latt_vec_top[0]+latt_vec_top[1])

        def nsites_sublatt(layer):
            if layer in ['A','B']:
                return self.layer_nsites_sublatt[0]
            elif layer in ['Atld','Btld']:
                return self.layer_nsites_sublatt[1]

        def latt_vec(layer):
            if layer in ['A','B']:
                return self.layer_latt_vecs[0]
            elif layer in ['Atld','Btld']:
                return self.layer_latt_vecs[1]

        for layer in layers:
            coord_xy = _coords_xy(layer)
            nsite = len(coord_xy)
            for i in layers[layer]:
                z = self.h*i
                coord = np.concatenate((coord_xy, [[z]]*nsite), axis=1)
                self.coords = np.concatenate((self.coords, coord), axis=0)
                self.layer_nsites.append(nsite)
                self.layer_nsites_sublatt.append(nsites_sublatt(layer))
                self.layer_latt_vecs = np.append(self.layer_latt_vecs, [latt_vec(layer)], axis=0)
                self.layer_zcoords.append(z)
        self.nsite = len(self.coords)

    def remove_top_layer(self):
        """
        after removing the top layer, it changes to be graphene
        """
        ids = self._layer_inds()[0]
        self.coords = self.coords[0:ids[1]+1]
        self.layer_nsites = [self.layer_nsites[0]]
        self.layer_zcoords = [self.layer_zcoords[0]]
        self.layer_types = [self.layer_types[0]]

    def remove_bottom_layer(self):
        """
        after removing the top layer, it changes to be graphene
        """
        ids = self._layer_inds()[1]
        self.coords = self.coords[ids[0]:ids[1]+1]
        self.layer_nsites = [self.layer_nsites[1]]
        self.layer_zcoords = [self.layer_zcoords[1]]
        self.layer_types = [self.layer_types[1]]

    def adjust_interlayer_dists(self, interlayer_dists={'AA':3.61, 'AB':3.38, 'AAtld':3.46}):
        """
        adjust the interlayer distances according to the inputing parameter: interlayer_dists
        """
        hs = interlayer_dists
        hs['BA'] = hs['AB']
        hs['AtldA'] = hs['AAtld']
        hs['AtldAtld'] = hs['AA']
        hs['BtldBtld'] = hs['AA']
        hs['AtldBtld'] = hs['AB']
        hs['BtldAtld'] = hs['AB']
        hs['BB'] = hs['AA']
        ids = self._layer_inds()
        ids_sort = np.argsort(self.layer_zcoords)
        zs = [0.]
        for i in range(1,len(ids_sort)):
            ind = ids_sort[i]
            ind0 = ids_sort[i-1]
            stack = self.layer_types[ind0]+self.layer_types[ind]
            zs.append(zs[-1]+hs[stack])
        for i in range(len(zs)):
            ind = ids_sort[i]
            self.layer_zcoords[ind]=zs[i]
            self.coords[:,-1][ids[ind][0]:ids[ind][1]+1] = zs[i]

class _MoirePatternMethods(_LayeredStructMethods):

    def add_hopping_pz(self, Rcut_intra=5.0, Rcut_inter=5.0, g0=3.12, g1=0.48, rc=6.14, lc=0.265, q_dist_scale=2.218):
        from tBG.hopping import calc_hopping_pz_PBC
        pmg_st = self.pymatgen_struct()
        a0 = self.a/np.sqrt(3)
        h0 = self.h
        self.hoppings = calc_hopping_pz_PBC(pmg_st, Rcut_intra=Rcut_intra, Rcut_inter=Rcut_inter,\
                      g0=g0, a0=a0, g1=g1, h0=h0, rc=rc, lc=lc, q_dist_scale=q_dist_scale,layer_inds=self._layer_inds()) 

    def add_hopping_wannier(self, max_dist=5.0, P=0, ts=[-2.8922, 0.2425, -0.2656, 0.0235, 0.0524, -0.0209, -0.0148, -0.0211]):
        if len(self.layer_nsites)>2:
            raise ValueError('Current version can not be used for nlayer>2')
        from tBG.hopping import calc_hopping_wannier_PBC, calc_hopping_wannier_PBC_new
        pmg_st = self.pymatgen_struct()
        layer_inds = self._layer_inds()
        layer_inds_sublatt = self._layer_inds_sublatt()
        layer_vec_to_NN = self.get_layer_vecs_to_NNs()
        latt_cont_max = max(np.linalg.norm(self.latt_vec_bott, axis=1)[0:2])
        self.hoppings = calc_hopping_wannier_PBC_new(pmg_st, layer_inds, layer_inds_sublatt, layer_vec_to_NN, \
                                                   max_dist=max_dist, P=P, ts=ts, a=self.a)
    def hoppings_2to3(self):
        return [{(j[0],j[1],0,j[2]):self.hoppings[i][j] for j in self.hoppings[i]} for i in range(self.nsite)]

    def hamilt_cell_diff(self, k, elec_field=0.0):
        Hk = np.zeros((self.nsite, self.nsite),dtype=complex)
        if elec_field:
            np.fill_diagonal(Hk,self.coords[:,-1]*elec_field)
        latt_vec = self.latt_vec[0:2][:,0:2]
        for i in range(self.nsite):
            for m,n,j in self.hoppings[i]:
                R = m*latt_vec[0]+n*latt_vec[1]
                t = self.hoppings[i][(m,n,j)]
                phase = np.exp(1j*np.dot(k, R))
                Hk[i,j] = Hk[i,j] + t*phase
                Hk[j,i] = Hk[j,i] + t*np.conj(phase)
        return Hk

    def diag_kpts(self, kpts, vec=0, pmk=0, elec_field=0.):
        """
        kpts: the coordinates of kpoints
        vec: whether to calculate the eigen vectors
        pmk: whether to calculate PMK for effective band structure
        elec_field: the electric field perpendicular to graphane plane
        fname: the file saveing results
        """
        if pmk:
            from tBG.fortran.spec_func import get_pk
        val_out = []
        vec_out = []
        pmk_out = []
        i = 1
        if vec or pmk:
            vec_calc = 1
        else:
            vec_calc = 0
        for k in kpts:
            print('%s/%s k' % (i, len(kpts)))
            Hk = self.hamilt_cell_diff(k, elec_field)
            vals, vecs, info = zheev(Hk, vec_calc)
            if info:
                raise ValueError('zheev failed')
            if pmk:
                Pk = get_pk(k, np.array(self.layer_nsites)/2, [1,1], 2, 2, vecs, self.coords, self.species())
                pmk_out.append(Pk)
            val_out.append(vals)
            if vec:
                vec_out.append(vecs)
            i = i + 1
        return np.array(val_out), np.array(vec_out), np.array(pmk_out)

class CommensuStruct(_MoirePatternMethods):
    """
    PRB 86 125414 (2012)
    """
    def __init__(self, a=2.46, h=3.35, rotate_cent='atom'):
        self.a = a
        self.h = h
        self.rotate_cent = rotate_cent

    def _pmg_sublatts_prim(self, m, n):
        def rotate_mat_mn(m,n):
            cos = (n**2+m**2+4*m*n)/(2*(n**2+m**2+m*n))
            sin = np.sqrt(3)*(m**2-n**2)/(2*(n**2+m**2+m*n))
            return np.array([[cos, -sin, 0],\
                             [sin,  cos, 0],
                             [0,     0,  1]])
        #latt_vec_bott = self.a*np.array([[1., 0., 0.],[0.5, np.sqrt(3)/2, 0.], [0, 0, 100/self.a]])
        latt_vec_bott = self.a*np.array([[np.sqrt(3)/2, -1/2., 0.],[np.sqrt(3)/2, 1/2., 0.], [0, 0, 100/self.a]])
        latt_vec_top = np.matmul(rotate_mat_mn(m,n), latt_vec_bott.T).T
        if self.rotate_cent=='atom':
            sites_bott = np.array([[0., 0., 0.],[1/3., 1/3., 0.]])
            sites_top = np.array([[0., 0., self.h/100],[1/3., 1/3., self.h/100]])
        elif self.rotate_cent=='hole':
            sites_bott = np.array([[1/3., 1/3., 0.],[2/3., 2/3., 0.]])
            sites_top = np.array([[1/3., 1/3., self.h/100],[2/3., 2/3., self.h/100]])
        latt_bott_0 =  pmg_struct(latt_vec_bott, ['C'], [sites_bott[0]])
        latt_bott_1 =  pmg_struct(latt_vec_bott, ['C'], [sites_bott[1]])
        latt_top_0 = pmg_struct(latt_vec_top, ['C'], [sites_top[0]])
        latt_top_1 = pmg_struct(latt_vec_top, ['C'], [sites_top[1]])

        self.latt_vec_bott = latt_vec_bott
        self.latt_vec_top = latt_vec_top
        return latt_bott_0, latt_bott_1, latt_top_0, latt_top_1

    def make_structure(self, m, n):
        latt_bott_0, latt_bott_1, latt_top_0, latt_top_1  = self._pmg_sublatts_prim(m, n)
        latt_bott_0.make_supercell([[m+n,-n,0],[n, m, 0],[0,0,1]])
        latt_bott_1.make_supercell([[m+n,-n,0],[n, m, 0],[0,0,1]])
        latt_top_0.make_supercell([[m+n, -m, 0],[m, n,0],[0,0,1]])
        latt_top_1.make_supercell([[m+n, -m, 0],[m, n,0],[0,0,1]])
        self.layer_nsites = [latt_bott_0.num_sites+latt_bott_1.num_sites, latt_top_0.num_sites+latt_top_1.num_sites]
        self.layer_nsites_sublatt = [[latt_bott_0.num_sites, latt_bott_1.num_sites],[latt_top_0.num_sites,latt_top_1.num_sites]]
        self.latt_vec = latt_bott_0.lattice.matrix
        self.coords = np.concatenate([latt_bott_0.cart_coords, latt_bott_1.cart_coords, \
                                      latt_top_0.cart_coords, latt_top_1.cart_coords])
        self.layer_latt_vecs = np.array([self.latt_vec_bott[0:2,0:2], self.latt_vec_top[0:2,0:2]])
        self.nsite = len(self.coords)
        self.twist_angle = np.arccos(0.5*(m**2+4*m*n+n**2)/(n**2+m*n+m**2))*180/np.pi

        
class TBG30Approximant(_MoirePatternMethods):

    def __init__(self, a=2.46, h=3.349, rotate_cent='hole'):
        self.a = a
        self.h = h
        self.rotate_cent = rotate_cent

    def _pmg_sublatts_prim(self, a_top):
        self.latt_vec_bott = self.a*np.array([[np.sqrt(3)/2, -1/2., 0.],
                                              [np.sqrt(3)/2, 1/2, 0],
                                              [0, 0, 100/self.a]])
        self.latt_vec_top = a_top*np.array([[1, 0, 0],
                                            [1/2, np.sqrt(3)/2, 0],
                                            [0, 0, 100/a_top]])
        if self.rotate_cent == 'hole':
            sites_bott = np.array([[1/3., 1/3., 0],[2/3., 2/3.,0]])
            sites_top = np.array([[1/3., 1/3., self.h/100],[2/3., 2/3., self.h/100]])
        elif self.rotate_cent == 'atom':
            sites_bott = np.array([[0., 0., 0],[1/3., 1/3.,0]])
            sites_top = np.array([[0., 0., self.h/100],[1/3., 1/3., self.h/100]])

        latt_bott_0 = pmg_struct(self.latt_vec_bott, ['C'], [sites_bott[0]]) 
        latt_bott_1 = pmg_struct(self.latt_vec_bott, ['C'], [sites_bott[1]])
        latt_top_0 = pmg_struct(self.latt_vec_top, ['C'], [sites_top[0]]) 
        latt_top_1 = pmg_struct(self.latt_vec_top, ['C'], [sites_top[1]])
        return latt_bott_0, latt_bott_1, latt_top_0, latt_top_1

    def make_structure(self, n_bottom):
        b = self.a/np.sqrt(3.)
        n_top = int(round(np.sqrt(3)*n_bottom)) # N value
        a_top = 3*b*n_bottom / n_top
        import math
        if math.gcd(n_bottom, n_top) != 1:
            raise ValueError('n_bottom and n_top share common dividor!')
        if not n_top%3:
            raise ValueError('n_top is times of 3!')
        latt_bott_0, latt_bott_1, latt_top_0, latt_top_1 = self._pmg_sublatts_prim(a_top)

        latt_bott_0.make_supercell([[n_bottom,n_bottom,0],[-n_bottom, 2*n_bottom, 0],[0,0,1]])
        latt_bott_1.make_supercell([[n_bottom,n_bottom,0],[-n_bottom, 2*n_bottom, 0],[0,0,1]])
        latt_top_0.make_supercell([[n_top, 0, 0],[0, n_top,0],[0,0,1]])
        latt_top_1.make_supercell([[n_top, 0, 0],[0, n_top,0],[0,0,1]])
        self.layer_nsites = [latt_bott_0.num_sites+latt_bott_1.num_sites, latt_top_0.num_sites+latt_top_1.num_sites]
        self.layer_nsites_sublatt = [[latt_bott_0.num_sites, latt_bott_1.num_sites],[latt_top_0.num_sites,latt_top_1.num_sites]]
        self.latt_vec = latt_bott_0.lattice.matrix
        self.coords = np.concatenate([latt_bott_0.cart_coords, latt_bott_1.cart_coords, \
                                      latt_top_0.cart_coords, latt_top_1.cart_coords])
        self.layer_latt_vecs = np.array([self.latt_vec_bott[0:2,0:2], self.latt_vec_top[0:2,0:2]])
        self.nsite = len(self.coords)

class Graphene(_MoirePatternMethods):
    def __init__(self, a=2.456, h=3.349):
        self.a = a
        self.h = h # for graphene multilayer structures
        self.latt_vec = a*np.array([[1, 0, 0],[np.cos(np.pi/3), np.sin(np.pi/3), 0], [0, 0, 100/a]])
        self.latt_vec_bott = a*np.array([[1, 0],[np.cos(np.pi/3), np.sin(np.pi/3)]])
        coords_frac = np.array([[1/3, 1/3],[2/3, 2/3]])
        coords_cart = frac2cart(coords_frac, self.latt_vec[0:2,0:2])
        self.coords = np.append(coords_cart, [[0],[0]], axis=1)
        self.layer_nsites = [2]
        self.layer_nsites_sublatt = [[1,1]]
        self.layer_latt_vecs=np.array([self.latt_vec])
        self.layer_zcoords = [0.0]
        self.nsite = 2
        
