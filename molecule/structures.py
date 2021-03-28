import numpy as np
import copy
from functools import reduce
from tBG.utils import frac2cart, cart2frac, rotate_on_vec, mirror_operate_2d, angle_between_vec_x
from tBG.molecule.round_disk import RoundDisk, _MethodsHamilt, _Output
from tBG.crystal.structures import _LayeredStructMethods

#### geometry functions ####
def get_interaction_two_lines(line1, line2):
    """
    between a1x + b1y + c1 = 0 and a2x + b2y + c2 = 0
    """
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    x = (c2*b1-b2*c1)/(a1*b2-b1*a2)
    y = (a2*c1-a1*c2)/(a1*b2-b1*a2)
    return np.array([x,y])

def get_line_through_two_points(p0, p1):
    """
    line is described by ax + by + c = 0
    """
    if abs(p0[1]-p1[1])<=1.e-14:
        a = 0
        b = 1
        c = -p0[1]
    else:
        a = 1
        # solve equation set: yb+c=-x and y'b+c=-x'
        b, c = np.linalg.solve(np.array([[p1[1], 1],[p0[1],1]]), [-p1[0], -p0[0]])
    return (a, b, c)

def site_inds_relative_a_line_including_specific_pnt(coords, line, pnt):
    """
    coords: the coordinates of all sites
    line: a list [a, b, c] to describe a line ax+by+c=0
    pnt: region must cover pnt
    """
    a, b, c = line
    xs = coords[:,0]
    ys = coords[:,1]
    sign = np.sign(a*pnt[0]+b*pnt[1]+c)
    inds = np.where(np.sign(a*xs+b*ys+c)==sign)[0]
    return inds

###### functions for polygon ##########
def get_vertices_regular_polygon(n, R):
    """
    get coordinates of all vertices of regular polygon with n sides
    n: the number of sides of regular polygon
    R: the distance from center to one vertics
    Note: one vertex is on x axis with coordinate [R, 0]
    """
    vert0 = R*np.array([1,0])
    vertices = []
    for i in range(n):
        theta_i = 360/n*i
        vertices.append(rotate_on_vec(theta_i, vert0))
    return vertices

def get_sides_from_vertices(vertices):
    sides = []
    for i in range(-1, len(vertices)-1):
        side_i = get_line_through_two_points(vertices[i], vertices[i+1])
        sides.append(side_i)
    return sides

def filter_sites_inside_polygon(coords, sides):
    """
    coords: atoms' coordinates for handling
    sides: all sides of the polygon
    """
    inds = site_inds_relative_a_line_including_specific_pnt(coords, sides[0], np.array([0,0]))
    for i in range(1, len(sides)):
        side = sides[i]
        inds_tmp = site_inds_relative_a_line_including_specific_pnt(coords, side, np.array([0,0]))
        inds = np.intersect1d(inds, inds_tmp) 
    return coords[inds]      
######################################################################################


class _SymmetryOperate:

    @staticmethod
    def _inds_pair(coords_0, coords_1):
        """
        coords_0: coords without symmetry operation
        coords_1: coords with symmetry operation
        return:
            a list [m, n, p, q, ......]
            describing atom positions changes according to
              0 ---> m
              1 ---> n
              2 ---> p
              3 ---> q
              ..............
        """
        coords_0 = np.round(coords_0, 3)
        coords_1 = np.round(coords_1, 3)

        coords_0[coords_0==0.] = 0.
        coords_1[coords_1==0.] = 0.

        def coords_to_strlist(coords):
            string = ['~'.join([str(x) for x in i]) for i in coords]
            return string
        coords_str0 = coords_to_strlist(coords_0)
        coords_str1 = coords_to_strlist(coords_1)
        if set(coords_str0)!=set(coords_str1):
            raise ValueError('Not a symmetry operation!!!')

        sorter = np.argsort(coords_str0)
        inds_end = sorter[np.searchsorted(coords_str0, coords_str1, sorter=sorter)]

        #return coords_str0, coords_str1, inds_end
        return inds_end

    @staticmethod
    def _inds_pair_old(coords_0, coords_1):
        """
        coords_0: coords without symmetry operation
        coords_1: coords with symmetry operation
        return:
            a list [m, n, p, q, ......]
            describing atom positions changes according to
              0 ---> m
              1 ---> n
              2 ---> p
              3 ---> q
              ..............
        """
        coords_0 = np.round(coords_0, 3)
        coords_1 = np.round(coords_1, 3)

        coords_0[coords_0==0.] = 0.
        coords_1[coords_1==0.] = 0.

        def coords_to_strlist(coords):
            string = ['~'.join([str(x) for x in i]) for i in coords]
            return string
        coords_str0 = coords_to_strlist(coords_0)
        coords_str1 = coords_to_strlist(coords_1)
        if set(coords_str0)!=set(coords_str1):
            raise ValueError('Not a symmetry operation!!!')

        sorter = np.argsort(coords_str0)
        inds_end = sorter[np.searchsorted(coords_str0, coords_str1, sorter=sorter)]

        #return coords_str0, coords_str1, inds_end
        return inds_end

    @staticmethod
    def _inds_pairs(coords_0, coords_1_list):
        inds_all = []
        n_op = len(coords_1_list)
        for i in range(n_op):
            coords_1 = coords_1_list[i]
            inds = _SymmetryOperate._inds_pair(coords_0, coords_1)
            inds_all.append(inds)
        return np.array(inds_all)

    def symmetry_operations(self):
        """
        Cns: Cn_i i=1,...,n-1 rotations around primitive axis
        C2s: C2 rotation around 2-fold axis perpendicular to primitive axis 
        sigma_vs: all vertical mirror operations 
        sigma_h: the horizontal mirror operation
        Sns: Sn_i i=1,...,n-1 rotation reflection around primitive axis, Sn_i = sigma_h * Cn_i
        S2n_odd: S2n_2j+1 j=0,...,n-1 rotation reflection around primitive axis S2n_2j+1 = sigma_h * C2n_2j+1
        **Notes**
            for sigma_vs and C2s, the vertical mirrors and C2 axes are obtained by
            rotating x axis by 180/n*i degree (i=0,...,n-1)
        """
        funcs_symmetry = {'Cns':self.Cns, 'C2s':self.C2s, 'sigma_vs':self.sigma_vs, 'C2s_QC':self.C2s_QC, \
                          'sigma_h':self.sigma_h, 'Sns':self.Sns, 'S2n_odds':self.S2n_odds}
        symmetry_ops = symmetry_operations(self.point_group)
        ops = {}
        for i in symmetry_ops:
            ops[i] = funcs_symmetry[i]()
        ops_inds = {i:self._inds_pairs(self.coords, ops[i]) for i in ops}
        return ops_inds

    def symmetryop2matrix(self, op):
        ndim = len(self.coords)
        M = np.zeros([ndim, ndim], dtype=np.int)
        def put_value(i):
            M[op[i],i] = 1
        [put_value(i) for i in range(ndim)]
        return M

    def _coords(self, coords):
        if coords is None:
            coords = copy.deepcopy(self.coords)
        else:
            coords = copy.deepcopy(coords)
        return coords

    def Cns(self, coords=None):
        """
        Description: all rotations related to n-fold principal axis
        operators are Cn_i with i=1,...,n-1
        """
        n = self.nfold
        coords_0 = self._coords(coords)
        coords_end = []
        for i in range(1, n):
            theta = 360/n*i
            coords_1 = rotate_on_vec(theta, coords_0)
            coords_end.append(coords_1)
        return np.array(coords_end)

    def sigma_h(self, coords=None):
        """
        return coordinates after horizontal mirror operation
        """
        y0 = self.h/2
        coords_0 = self._coords(coords)
        coords_0[:,2] = -( coords_0[:,2] - y0 ) + y0
        return np.array([coords_0])

    def sigma_vs(self, coords=None):
        """
        return coordinates after all vertical mirrors (including all sigma_v and sigma_d),
        All vertical mirrors are obtained by rotating x axis by (180/n)*i i=0,...,n-1 
        """
        coords_0 = self._coords(None)
        zs = coords_0[:,2]
        axes = all_C2_axes(self.nfold)
        coords_end = []
        for axis in axes:
            coords_m = self._sigma_v(axis, coords=coords)
            coords_end.append(coords_m)
        return np.array(coords_end)

    def _sigma_v(self, mirror_v, coords=None):
        """
        Return coordinates after a given vertical mirror operation.
        The verticle mirror is described by a vector [x,y], which means the mirror is through
        origin point and point [x, y]
        """
        coords_0 = self._coords(None)
        zs = coords_0[:,2]
        m = get_line_through_two_points([0,0], mirror_v)
        coords_m = mirror_operate_2d(m, coords_0)
        coords_m = np.append(coords_m, zs.reshape(-1,1), axis=1)    
        return coords_m

    def S2n_odds(self, coords=None):
        """
        These operations are symmetry operations for D3d D6d point group
        These include S2n_2i+1 with 2i+1<2n
        """
        n = self.nfold
        coords_0 = self._coords(coords)
        #### C2n_odds first ####
        coords_end = []
        for i in range(0, n):
            theta = 180/n*(2*i+1)
            coords_1 = rotate_on_vec(theta, coords_0)
            coords_end.append(coords_1)
        coords_C2n_odds = np.array(coords_end)
        ###############################################
        coords_Cns_sigma_h = [self.sigma_h(coords)[0] for coords in coords_C2n_odds]
        return np.array(coords_Cns_sigma_h)

    def Sns(self, coords=None):
        """
        Sns = Cns * sigma_h
        """
        coords_0 = self._coords(coords)
        coords_Cns = self.Cns(coords_0)
        coords_Cns_sigma_h = [self.sigma_h(coords)[0] for coords in coords_Cns]
        return np.array(coords_Cns_sigma_h)

    def C2s(self, coords=None):
        """
        C2s = sigma_h * sigma_v
        one of vertical mirrors of sigma_v passes through x axis
        """
        coords_0 = self._coords(coords)
        coords_sigma_v = self.sigma_vs(coords_0)
        coords_sigma_v_sigma_h = [self.sigma_h(coords)[0] for coords in coords_sigma_v]
        return np.array(coords_sigma_v_sigma_h)

    def C2s_QC(self, coords=None):
        """
        C2s_QC = sigma_h * sigma_v_QC
        one of vertical mirrors of sigma_v passes through x axis
        """
        n = self.nfold
        coords_0 = self._coords(coords)
        def all_C2_axes_QC():
            axes = []
            for i in range(n):
                theta = 180/(2*n)*(2*i+1)
                axis = rotate_on_vec(theta,[1,0])
                axes.append(axis)
            return axes 

        def sigma_vs_QC():
            zs = coords_0[:,2]
            axes = all_C2_axes_QC()
            coords_end = []
            for axis in axes:
                coords_m = self._sigma_v(axis, coords=coords_0)
                coords_end.append(coords_m)
            return np.array(coords_end)
        coords_sigma_v_QC = sigma_vs_QC()
        coords_sigma_v_sigma_h = [self.sigma_h(coords)[0] for coords in coords_sigma_v_QC]
        return np.array(coords_sigma_v_sigma_h)

class _GemetryOperate:
    def get_layer_nsites(self):
        nsite_unrott = len(np.where(self.coords[:,2]==0.)[0])
        nsite_rott = len(np.where(self.coords[:,2]>0.)[0])
        return [nsite_unrott, nsite_rott]        

    def get_layer_nsites_sublatt(self):
        def get_nsites_sublatt_onelayer(coords, latt_vec):
            fracs = cart2frac(coords[:,0:2], latt_vec)
            fracs_site0 = np.float16(self.site0 - fracs)
            fracs_redu = np.modf(fracs_site0)[0] 
            ind_0 = np.where(fracs_redu[:,0]==0.)[0] 
            ind_1 = np.where(fracs_redu[:,1]==0.)[0] 
            n0 = len(np.intersect1d(ind_0, ind_1))
            n1 = len(fracs) - n0
            return n0, n1
        layer_nsites = self.get_layer_nsites()
        coords_bott = self.coords[:layer_nsites[0]]
        coords_top = self.coords[layer_nsites[0]:]
        nb0,nb1= get_nsites_sublatt_onelayer(coords_bott, self.layer_latt_vecs[0])
        nt0,nt1= get_nsites_sublatt_onelayer(coords_top, self.layer_latt_vecs[1])
        return [[nb0, nb1],[nt0,nt1]]
    
    def _initial_round_disk(self, R, twist_angle, overlap='hole', rm_single_bond=False):
        """
        prepare a big initial round disk for cutting to get various shapes
        """
        self.twist_angle = twist_angle
        rd = RoundDisk()
        rd.make_structure(R, rotation_angle=twist_angle, a=self.a, h=self.h, \
                                  overlap=overlap, rm_dangling=rm_single_bond)
        self.coords = rd.coords
        self.latt_vec_bott = rd.latt_vec_bott
        self.latt_vec_top = rd.latt_vec_top
        self.layer_latt_vecs = rd.layer_latt_vecs
        self.site0 = rd.site0
        self.site1 = rd.site1
        self.layer_zcoords = rd.layer_zcoords
        self.layer_types = rd.layer_types
    
    def _remove_single_bonds(self):
        cc1 = self.a/np.sqrt(3)
        cc1_cc2_mid = (cc1+self.a)/2
        while True:
            pmg_st = self.pymatgen_struct()
            bonds = pmg_st.get_neighbor_list(cc1_cc2_mid)[0]
            counts = np.array([np.count_nonzero(bonds==i) for i in range(len(self.coords))])
            inds_rm = np.where(counts<=1)[0]
            inds = np.in1d(range(pmg_st.num_sites), inds_rm)
            self.coords = self.coords[~inds]
            self.layer_nsites = self.get_layer_nsites()
            self.layer_nsites_sublatt = self.get_layer_nsites_sublatt()
            if not len(inds_rm):
                break

def symmetry_operations(point_group):
    """
    sigma_vs[0] includes x axis
    """
    if point_group in ['D2h', 'D3h', 'D6h']:
        return ['Cns', 'C2s', 'Sns', 'sigma_vs', 'sigma_h']
    elif point_group in ['D2', 'D3', 'D6']:
        return ['Cns', 'C2s']
    elif point_group in ['C3v', 'C6v']:
        return ['Cns', 'sigma_vs']
    elif point_group in ['D3d','D6d']:
        return ['Cns', 'C2s_QC', 'S2n_odds', 'sigma_vs']

def all_C2_axes(n):
    """
    orient_1st: the orient of the 1st 2-fold axis 'armchair or zigzag'
    """
    axis0 = np.array([1,0])
    axes = []
    for i in range(n):
        axis_i = rotate_on_vec(180/n*i, axis0)
        axes.append(axis_i)
    return np.array(axes)

class _Disorder:
    def add_vacancy(self, concentration, indices_chosen):
        """
        return the coords after adding vacancies with concentration 
        args:
            concentration: vacancy concentration
            indices_chosen: from which vacancies are induced
        """
        natom = len(self.coords)
        nvac = int(natom*concentration)
        indices_chosen = np.array(indices_chosen)

        indices_vac = np.random.choice(len(indices_chosen), nvac, replace=False)
        indices_vac = indices_chosen[indices_vac]
        print('n_vac: %s' % len(indices_vac))
        indices = np.setdiff1d(range(natom), indices_vac)
        n_site_bott = self.layer_nsites[0]
        nvac_bott = len(np.where(indices_vac<n_site_bott)[0])
        nvac_top = len(np.where(indices_vac>=n_site_bott)[0])
        self.layer_nsites = np.array(self.layer_nsites) - np.array([nvac_bott, nvac_top])
        self.coords = self.coords[indices]

class _INIT:
    def __init__(self, a=2.46, h=3.35):
        self.a = a
        self.h = h
        self.B = 0
        self.E = 0
        self.Es_onsite = 0

class QuantumDot(_INIT, _MethodsHamilt, _GemetryOperate, _SymmetryOperate, \
                 _Disorder,_LayeredStructMethods, _Output):
    """
    the class for a quantum dot of common twisted bilayer graphene
    x axis is always along one C2 axis (perpendicular to the primitive axis)
    """
    def __init__(self, a=2.46, h=3.35):
        _INIT.__init__(self, a, h)
    
    def _angle_C2_axis0_x(self):
        """
        get the 1st 2-fold axis, which is described by a vector [x, y]
        """
        if self.orient == 'armchair':
            vec = self.latt_vec_bott[0]+self.latt_vec_bott[1]
        elif self.orient == 'zigzag':
            vec = self.latt_vec_bott[0]
        vec = vec/np.linalg.norm(vec)
        return self.twist_angle/2 + angle_between_vec_x(vec)

    def rotate_struct_axis0_to_x_axis(self):
        angle_axis0_x = self._angle_C2_axis0_x()
        self.coords = rotate_on_vec(-angle_axis0_x, self.coords)
        layer_latt_vecs = []
        for i in range(len(self.layer_latt_vecs)):
            latt_vec_new = rotate_on_vec(-angle_axis0_x, self.layer_latt_vecs[i])
            layer_latt_vecs.append(latt_vec_new)
        self.layer_latt_vecs = layer_latt_vecs

    def rectangle(self, w, h, twist_angle, overlap='side_center', rm_single_bond=True, new_cut_style=False):
        if overlap not in ['side_center', 'hole']:
            raise ValueError("Overlap can only be 'side_center' or 'hole' to keep the symmetry")
        self.orient = 'armchair'
        self.nfold = 2
        R = np.sqrt(w**2/4+h**2/4)
        self._initial_round_disk(R+2, twist_angle, overlap=overlap)
        self.rotate_struct_axis0_to_x_axis()
        ## get vertices of rectangle
        vert0 = self.a*np.array([ w/2,  h/2])
        vert1 = self.a*np.array([-w/2,  h/2])
        vert2 = self.a*np.array([-w/2, -h/2])
        vert3 = self.a*np.array([ w/2, -h/2])
        vertices = np.array([vert0, vert1, vert2, vert3])
        if new_cut_style:
            vertices = rotate_on_vec(-twist_angle/2, vertices) # cut with reference of bottom layer
        ## get all sides
        sides = get_sides_from_vertices(vertices)
        # get atoms inside polygon 
        self.coords = filter_sites_inside_polygon(self.coords, sides)
        if new_cut_style:
            coords_bottom =  self.coords[np.where(self.coords[:,-1]==0.0)[0]] ## left only bottom 
            coords_top = rotate_on_vec(twist_angle, coords_bottom)
            coords_top[:,-1] = self.h
            self.coords = np.concatenate([coords_bottom, coords_top]) 
        
        self.layer_nsites = self.get_layer_nsites()
        if rm_single_bond:
            self._remove_single_bonds()
        self.point_group = self.get_point_group()
        self.layer_nsites = self.get_layer_nsites()
        self.layer_nsites_sublatt = self.get_layer_nsites_sublatt()
        
    def regular_polygon(self, n, R, twist_angle, overlap='hole', rm_single_bond=True, \
                         orient='armchair', new_cut_style=False):
        """
        n: the regular polygon with n sides (n=3, 6, 12)
        R: the distance from center to vertex (in units of a: the lattice constant of graphene)
        twist_angle: the twist angle between two layers
        rm_single_bond: whether the atom with only one negibors are removed
        orient: zigzag or armchair along x-axis
        new_cut_style: False: rotate and cut, True: cut and rotate
        """
        ######################## check inputs ################################################
        if overlap not in ['atom', 'hole', 'atom1']:
            raise ValueError('Overlap %s is not recogenized!' % overlap)

        if orient not in ['armchair', 'zigzag']:
            raise ValueError('Oriention %s is not recogenized!' % orient)
        
        ## Four following lines make sure the highest symmetry
        if n==3 and orient=='zigzag' and overlap!='hole':
            raise ValueError('For Triangle and zigzag orientation, overlap can only be hole!')

        if n in [6,12] and overlap!='hole':
            raise ValueError('For Hexgon or Decagon, overlap can only be hole!')
        ######################### check inputs done ###########################################

        self.orient = orient
        self.nfold = n

        ######################## informations of regular polygon ###############################
        vertices = get_vertices_regular_polygon(self.nfold, self.a*(R+1.e-4))
        if new_cut_style:
            vertices = rotate_on_vec(-twist_angle/2, vertices) # cut with reference of bottom layer
        sides = get_sides_from_vertices(vertices)
        # get atoms inside polygon 
        self._initial_round_disk(R+2, twist_angle, overlap=overlap)
        self.rotate_struct_axis0_to_x_axis()
        self.coords = filter_sites_inside_polygon(self.coords, sides)
        if new_cut_style:
            coords_bottom =  self.coords[np.where(self.coords[:,-1]==0.0)[0]] ## left only bottom 
            coords_top = rotate_on_vec(twist_angle, coords_bottom)
            coords_top[:,-1] = self.h
            self.coords = np.concatenate([coords_bottom, coords_top]) 
            
        ########################################################################################

        if rm_single_bond:
            self._remove_single_bonds()
        self.layer_nsites = self.get_layer_nsites()
        self.layer_nsites_sublatt  = self.get_layer_nsites_sublatt()
        self.point_group = self.get_point_group()

    def get_point_group(self):
        if self.twist_angle == 0.0:
            if self.nfold in [2, 3, 6]:
                return 'D%sh' % self.nfold
            elif self.nfold == 12:
                return 'D6h'
        else:
            if self.nfold in [2, 3, 6]:
                return 'D%s' % self.nfold
            elif self.nfold == 12:
                return 'D6'


class QuantumDotQC(_INIT,_MethodsHamilt, _GemetryOperate, _SymmetryOperate, \
                   _Disorder,_LayeredStructMethods, _Output):
    """
    A quantum dot of 30 degree twisted bilayer graphene with rotation center at the hole of two layers
    x axis: armchair direction of bottom layer and zigzag direction of the top layer
    y axis: zigzag direction of bottom layer and armchair direction of the top layer
    """
    def __init__(self, a=2.46, h=3.35):
        _INIT.__init__(self, a, h)
    
    def regular_polygon(self, n, R, OV_orientation=0, overlap='hole', rm_single_bond=True):
        """
        n: the fold number
        R: the distance from origin to any vertex in units of a
        OV_orient: the angle of origin-vertex relative x axis in units of degree
        overlap: the rotation axis
        rm_single_bond: whether the dangling bonds are removed
        """
        self.nfold = n
        self._initial_round_disk(R+2, 30., overlap=overlap)
        ######################## informations of regular polygon ###############################
        vertices = get_vertices_regular_polygon(n, self.a*(R+1.e-4))
        if OV_orientation:
            vertices = rotate_on_vec(OV_orientation, vertices)
        sides = get_sides_from_vertices(vertices)
        # get atoms inside polygon 
        self.coords = filter_sites_inside_polygon(self.coords, sides)
        ########################################################################################
        if rm_single_bond:
            self._remove_single_bonds()
        self.layer_nsites = self.get_layer_nsites()
        self.layer_nsites_sublatt = self.get_layer_nsites_sublatt()
        if n in [3, 6]:
            self.point_group = 'C%iv' % n
        if n == 12:
            self.nfold = 6
            self.point_group = 'D6d'

       
    def round_disk(self, R, rm_single_bond=True):
        self._initial_round_disk(R, 30., overlap='hole', rm_single_bond=rm_single_bond)
        self.point_group = 'D6d'
        self.nfold = 6
        self.layer_nsites = self.get_layer_nsites()
        self.layer_nsites_sublatt = self.get_layer_nsites_sublatt()

    def set_electric_field(self, E=0):
        """
        field is along the z aixs
        """
        if E:
            self.E = E
            self.Es_onsite = self.coords[:,2]*E
            if self.point_group == 'D6d':
                self.point_group = 'C6v'


class QuantumDotAB(_INIT,_MethodsHamilt, _GemetryOperate, _SymmetryOperate,_Disorder,\
                   _LayeredStructMethods, _Output):
    def __init__(self, a=2.46, h=3.35):
        _INIT.__init__(self, a, h) 

    def regular_polygon(self, n, R, overlap='atom-atom', rm_single_bond=True):
        self.nfold = n
        b = self.a/np.sqrt(3)
        self._initial_round_disk(R+2, 60., overlap='atom')
        
        self.nfold = 3
        self.point_group = 'C3v'
        
        if overlap=='atom-atom':
            if n in [6, 12]:
                self.point_group = 'D3d'
        else:
            if overlap=='atom-hole': # top hole over bottom atom directly
                orig = np.array([b, 0, 0])
            elif overlap=='hole-atom': # top atom over bottom hole directly
                orig = np.array([-b, 0, 0])
            self.coords = self.coords - orig
        ######################## informations of regular polygon ###############################
        vertices = get_vertices_regular_polygon(n, self.a*(R+1.e-4))
        sides = get_sides_from_vertices(vertices)
        # get atoms inside polygon 
        self.coords = filter_sites_inside_polygon(self.coords, sides)
        ########################################################################################
        self.layer_nsites = self.get_layer_nsites()
        if rm_single_bond:
            self._remove_single_bonds()
        self.layer_nsites = self.get_layer_nsites()
        self.layer_nsites_sublatt = self.get_layer_nsites_sublatt()

