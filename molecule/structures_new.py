import numpy as np
import copy
from functools import reduce
from tBG.utils import frac2cart, cart2frac, rotate_on_vec, mirror_operate_2d, angle_between_vec_x
from tBG.molecule.round_disk_new import RoundDisk, coords_to_strlist
from tBG.crystal.structures import _LayeredStructMethods
from scipy.spatial.transform import Rotation
from pymatgen.symmetry.analyzer import SymmOp

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
    if abs(p0[1]-p1[1])<=1.e-13:
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
    A function to filter sites in the same side as the refree point relative to a line
    Return the site indices of these filtered sites 

    coords: the coordinates of all sites
    line: a list [a, b, c] to describe a line ax+by+c=0
    pnt: the coordinate of the refree point
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
    A function to get all vertices of a n-sided regular polygon with one vertice on +x axis

    n: number of sides of the regular polygon
    R: distance from origin to vertices 
    """
    vert0 = R*np.array([1,0])
    vertices = []
    for i in range(n):
        theta_i = 360/n*i
        vertices.append(rotate_on_vec(theta_i, vert0))
    return vertices

def get_sides_from_vertices(vertices):
    """
    a function to get sides constructed from two adjacent vertices
    """
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

class _SymmOps:
    """
    operations in xyz space
    """
    @staticmethod
    def _inds_pair(coords_0, coords_1):
        """
        coords_0: coords before operation
        coords_1: coords after operation
        return:
            a list [m, n, p, q, ......]
            describing atom positions changes 
              0 ---> m  means 0th site in coords_0 change to mth site in coords_1 
              1 ---> n
              2 ---> p
              3 ---> q
              ..............
        """
        coords_str0 = coords_to_strlist(coords_0)
        coords_str1 = coords_to_strlist(coords_1)
        if set(coords_str0)!=set(coords_str1):
            raise ValueError('Not a symmetry operation!!!')

        sorter = np.argsort(coords_str0)
        inds_end = sorter[np.searchsorted(coords_str0, coords_str1, sorter=sorter)]
        return inds_end

    @staticmethod
    def _Oppz2mat(op):
        ndim = len(op)
        M = np.zeros([ndim, ndim], dtype=np.int)
        def put_value(i):
            M[op[i],i] = 1
        [put_value(i) for i in range(ndim)]
        return M

    def _ops_in_pz(self, ops_xyz):
        def get_op_pz(op_xyz):
            mol = self.pymatgen_molecule()
            coords0 = mol.cart_coords
            op_pmg = SymmOp.from_rotation_and_translation(op_xyz)
            mol.apply_operation(op_pmg)
            coords_new = mol.cart_coords
            op = self._inds_pair(coords0, coords_new)
            return op
        ops_pz = np.array([get_op_pz(op_xyz) for op_xyz in ops_xyz])
        return ops_pz

    def _SU2_mat_from_eular(self, alpha, beta, gamma):
        a11 = np.exp(-1j/2*(alpha+gamma)) * np.cos(beta/2)
        a12 = -np.exp(-1j/2*(alpha-gamma)) * np.sin(beta/2)
        a21 = np.exp(1j/2*(alpha-gamma)) * np.sin(beta/2)
        a22 = np.exp(1j/2*(alpha+gamma)) * np.cos(beta/2)
        return np.array([[a11, a12],[a21, a22]])

    def _ops_in_spin(self, ops_xyz):
        def get_op_spin(op_xyz):
            det = np.linalg.det(op_xyz)
            op_xyz_no_I= det*op_xyz # get rid of inversion 
            alpha, beta, gamma = Rotation.from_matrix(op_xyz_no_I).as_euler('ZYZ') # intrinstic definition for eular anglars
            op_spin = self._SU2_mat_from_eular(alpha, beta, gamma)
            return op_spin
        ops_spin = np.array([get_op_spin(op_xyz) for op_xyz in ops_xyz])
        return ops_spin
            

    def Cns(self, spin=False):
        n = self.nfold
        ops_xyz = Rotation.from_rotvec([[0,0,i*2*np.pi/n] for i in range(1,n)]).as_matrix()
        ops_pz = self._ops_in_pz(ops_xyz)
        return [ops_pz, self._ops_in_spin(ops_xyz)] if spin else ops_pz

    def sigma_h(self, spin=False):
        ops_xyz = np.array([[[1,0,0],[0,1,0],[0,0,-1]]])
        ops_pz = self._ops_in_pz(ops_xyz)
        return [ops_pz, self._ops_in_spin(ops_xyz)] if spin else ops_pz

    def sigma_vs(self, spin=False):
        """
        one sigma_v mirror include x-z plane
        sigma 
        """
        n = self.nfold
        I = -1
        axis0 = np.array([0, np.pi, 0]) #
        ops = []
        for i in range(0, n):
            r = Rotation.from_rotvec([0,0,np.pi/n*i])
            axis = r.apply(axis0)
            op = I*Rotation.from_rotvec(axis).as_matrix()
            ops.append(op)
        ops_xyz = np.array(ops)
        ops_pz = self._ops_in_pz(ops_xyz)
        return [ops_pz, self._ops_in_spin(ops_xyz)] if spin else ops_pz

    def S2n_odds(self, spin=False):
        """
        These operations are symmetry operations for D3d D6d point group
        These include S2n_2i+1 with 2i+1<2n
        """
        ops = []
        sigma_h = np.array([[1,0,0],[0,1,0],[0,0,-1]])
        n = self.nfold
        #### C2n_odds first ####
        for i in range(0, n):
            theta = np.pi/n*(2*i+1)
            op = np.matmul(sigma_h, Rotation.from_rotvec([0,0,theta]).as_matrix())
            ops.append(op)
        ops_xyz =  np.array(ops)
        ops_pz = self._ops_in_pz(ops_xyz)
        return [ops_pz, self._ops_in_spin(ops_xyz)] if spin else ops_pz

    def Sns(self, spin=False):
        """
        Sns = Cns * sigma_h
        """
        sigma_h = np.array([[1,0,0],[0,1,0],[0,0,-1]])
        Cns = self.Cns()[0]
        ops_xyz = np.array([np.matmul(sigma_h, i) for i in Cns])
        ops_pz = self._ops_in_pz(ops_xyz)
        return [ops_pz, self._ops_in_spin(ops_xyz)] if spin else ops_pz

    def C2s(self, spin=False):
        n = self.nfold
        axis0 = np.array([np.pi, 0, 0])
        ops = []
        for i in range(0,n):
            r = Rotation.from_rotvec([0, 0, np.pi/n*i])
            axis = r.apply(axis0)
            op = Rotation.from_rotvec(axis).as_matrix()
            ops.append(op)
        ops_xyz =  np.array(ops)
        ops_pz = self._ops_in_pz(ops_xyz)
        return [ops_pz, self._ops_in_spin(ops_xyz)] if spin else ops_pz

    def C2s_QC(self, spin=False):
        """
        C2s_QC = sigma_h * sigma_v_QC
        one of vertical mirrors of sigma_v passes through x axis
        """
        n = self.nfold
        ops = []
        axis0 = np.array([np.pi, 0, 0])
        for i in range(n):
            theta = np.pi/(2*n)*(2*i+1)
            r = Rotation.from_rotvec([0, 0, theta])
            axis = r.apply(axis0)
            op = Rotation.from_rotvec(axis).as_matrix()
            ops.append(op)
        ops_xyz = np.array(ops)
        ops_pz = self._ops_in_pz(ops_xyz)
        return [ops_pz, self._ops_in_spin(ops_xyz)] if spin else ops_pz

    def symmetry_operations(self, spin=False):
        """
        Cns: Cn_i i=1,...,n-1 rotations around z-axis
        C2s: C2 rotation around 2-fold axis perpendicular to primitive axis, one of which is x-axis
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
            ops[i] = funcs_symmetry[i](spin)
        return ops

def symmetry_operations(point_group):
    """
    sigma_vs[0] includes x axis
    """
    # For the first three cases: C2s axes are inside sigma_vs mirrors
    # and one of them is x-axis
    if point_group in ['D2h', 'D3h', 'D6h']:
        return ['Cns', 'C2s', 'Sns', 'sigma_vs', 'sigma_h']
    elif point_group in ['D2', 'D3', 'D6']:
        return ['Cns', 'C2s']
    elif point_group in ['C3v', 'C6v']:
        return ['Cns', 'sigma_vs']
    # For the last case: C2s_QC axes are not inside sigma_vs mirrors
    # one sigma_vs mirror with x-axis inside
    # but C2s_Qs axes between sigma_v mirrors
    elif point_group in ['D3d','D6d']:
        return ['Cns', 'C2s_QC', 'S2n_odds', 'sigma_vs']


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

class _QuantumDotBase(RoundDisk):
    def _twisted_bilayer(self, layer_origins, layer_orientations, layer_vertices, a=2.46, h=3.35, rm_dangling=True):
        """
        origins: two layer origins, such as ['hole', 'hole'] 
        orients: two layer orientations, such as [0, 30] 
        vertices: polygon vertices for two layers as outline shapes
        """
        R_max = np.max([np.linalg.norm(i, axis=1) for i in layer_vertices])/a
        self.twisted_multilayer(R_max+2, a=a, orientations=layer_orientations, origins=layer_origins, h=h, rm_dangling=False)
        layer_ids = self._layer_inds()
        coords_cut = []
        for i in range(len(layer_origins)):
            coords = self.coords[layer_ids[i][0]:layer_ids[i][1]+1]
            sides = get_sides_from_vertices(layer_vertices[i])
            coords = filter_sites_inside_polygon(coords, sides)
            coords_cut.append(coords)
        self.coords = np.concatenate(coords_cut, axis=0)
        self.layer_nsites = self.get_layer_nsites()
        self.layer_nsites_sublatt = self.get_layer_nsites_sublatt()
        if rm_dangling:
            self._remove_single_bonds()
        self.layer_nsites = self.get_layer_nsites()
        self.layer_nsites_sublatt = self.get_layer_nsites_sublatt()

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

    def get_layer_nsites(self):
        nsite_unrott = len(np.where(self.coords[:,2]<=0.)[0])
        nsite_rott = len(np.where(self.coords[:,2]>0.)[0])
        return [nsite_unrott, nsite_rott]        

    def get_layer_nsites_sublatt(self):
        def get_nsites_sublatt_onelayer(coords, latt_vec, frac_site0):
            fracs = cart2frac(coords[:,0:2], latt_vec)
            fracs_site0 = np.float16(frac_site0 - fracs)
            fracs_redu = np.modf(fracs_site0)[0] 
            ind_0 = np.where(fracs_redu[:,0]==0.)[0] 
            ind_1 = np.where(fracs_redu[:,1]==0.)[0] 
            n0 = len(np.intersect1d(ind_0, ind_1))
            n1 = len(fracs) - n0
            return n0, n1
        layer_nsites = self.get_layer_nsites()
        coords_bott = self.coords[:layer_nsites[0]]
        coords_top = self.coords[layer_nsites[0]:]
        nb0,nb1= get_nsites_sublatt_onelayer(coords_bott, self.layer_latt_vecs[0], self.layer_fracs_sublatt[0][0])
        nt0,nt1= get_nsites_sublatt_onelayer(coords_top, self.layer_latt_vecs[1],self.layer_fracs_sublatt[1][0])
        return [[nb0, nb1],[nt0,nt1]]
    

def get_pointgroup_nfold(qd):
    """
    a function to get the point group of a structure qd
    method:
        (1)if qd has sites less than 100, the point group will be analyzed by using the pymatgen molecule itsself
        (2)if sites number is more than 100, a slow symmetry analysis by using the pymatgen molecule itsself is expected.
        So we use an alternative structure to analyze the point group, *which is expected to have the same symmetry as the original one.*
        The alternative structure is constructed from scratch by using a small origin-vertices distance to cut a smaller sample. Meanwhile,
        the shape remains and the site number is the smallest larger than 60 in each case.
    """
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
    if len(qd.coords)<=100:
        mol = qd.pymatgen_molecule()
    else:
        n = 1.5
        while True:
            layer_vertices = [shrink_vertices(i,n) for i in qd.info['layer_vertices']]
            qd0 = _QuantumDotBase()
            qd0._twisted_bilayer(qd.info['layer_origins'], qd.info['layer_orientations'], layer_vertices, a=2.46, h=3.35, rm_dangling=False)
            if len(qd0.coords)<=60:
                n = n + 1 
            else:
                break
        print(len(qd0.coords))
        mol = qd0.pymatgen_molecule()
    pga = PointGroupAnalyzer(mol, tolerance=1.e-5)
    point_group = pga.get_pointgroup().sch_symbol
    try:
        nfold = int(point_group[1])
    except:
        nfold = 0
    return point_group, nfold, pga

def shrink_vertices(vertices, n):
    """
    a function to get new vertices with origin-verticres distance smaller but orientations (origin-vertices) unchanged

    *** the distances from origin to all given vertices should be the same when this function is used ***
    """
    a = 2.46
    R = n*a
    norm = np.linalg.norm(vertices, axis=1)
    vertices_new = R*np.array([vertices[i]/norm[i] for i in range(len(norm))])
    return vertices_new

class QuantumDot(_Disorder, _QuantumDotBase, _SymmOps):
    """
    A class for bilayer graphene quantum dot
    
    Note for symmetry operations:
    Except for QC, x-axis is always one C2 axis and xz plane is always one sigma_v mirror
    For QC, xz plane is also one sigma_v mirror but C2 axes (labeled by C2_QC) bisect sigma_v mirrors.  
    """
    def _get_struct_and_pointgroup(self, layer_origins, layer_orientations, layer_vertices, a, h, rm_dangling):
        self._twisted_bilayer(layer_origins, layer_orientations, layer_vertices, a=a, h=h, rm_dangling=rm_dangling)
        self.info = {'layer_orientations':layer_orientations, 'layer_origins':layer_origins,'layer_vertices':layer_vertices}
        self.coords[:,-1] = self.coords[:,-1] - h/2
        self.point_group, self.nfold, self.pga = get_pointgroup_nfold(self) 

    def rectangle_tBG(self, W, H, twist_angle, overlap='side_center', rm_dangling=True, new_cut_style=False, h=3.35, a=2.46):
        if overlap not in ['side_center', 'hole']:
            raise ValueError("Overlap can only be 'side_center' or 'hole' to keep the symmetry")
        ## get vertices of rectangle
        vert0 = a*np.array([ W/2,  H/2])
        vert1 = a*np.array([-W/2,  H/2])
        vert2 = a*np.array([-W/2, -H/2])
        vert3 = a*np.array([ W/2, -H/2])
        vertices = np.array([vert0, vert1, vert2, vert3])
        layer_vertices = [vertices, vertices]
        if new_cut_style:
            vertices_bott = rotate_on_vec(-twist_angle/2, vertices) # cut with reference of bottom layer
            vertices_top = rotate_on_vec(twist_angle/2, vertices) # cut with reference of bottom layer
            layer_vertices = [vertices_bott, vertices_top]
        layer_orientations = [-twist_angle/2, twist_angle/2]
        layer_origins = [overlap, overlap]
        self._get_struct_and_pointgroup(layer_origins, layer_orientations, layer_vertices, a, h, rm_dangling)
        #self._twisted_bilayer(layer_origins, layer_orientations, layer_vertices, a=a, h=h, rm_dangling=rm_dangling)
        #self.info = {'layer_orientations':layer_orientations, 'layer_origins':layer_origins,'layer_vertices':layer_vertices}
        #self.point_group, self.nfold = get_pointgroup_nfold(self) 
        
    def regular_polygon_tBG(self, n, R, twist_angle, overlap='hole', rm_dangling=True, \
                         orient='armchair', new_cut_style=False, a=2.46, h=3.35):
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

        if twist_angle in [0, 30, 60]:
            raise ValueError('For 0, 30 and 60 degree cases, please using other functions!')


        ######################## informations of regular polygon ###############################
        vertices = get_vertices_regular_polygon(n, a*(R+1.e-4))
        layer_vertices = [vertices, vertices]
        if new_cut_style:
            vertices_bott = rotate_on_vec(-twist_angle/2, vertices) # cut with reference of bottom layer
            vertices_top = rotate_on_vec(twist_angle/2, vertices) # cut with reference of bottom layer
            layer_vertices = [vertices_bott, vertices_top]
        if orient == 'armchair':
            layer_orientations = [-twist_angle/2, twist_angle/2]
        elif orient == 'zigzag':
            layer_orientations = [30-twist_angle/2, 30+twist_angle/2]
        layer_origins = [overlap, overlap]
        self._get_struct_and_pointgroup(layer_origins, layer_orientations, layer_vertices, a, h, rm_dangling)

    def regular_polygon_QC(self, n, R, OV_orientation=0, overlap='hole', rm_dangling=True, a=2.46, h=3.35):
        """
        n: the fold number
        R: the distance from origin to any vertex in units of a
        OV_orient: the angle of origin-vertex relative x axis in units of degree
        overlap: the rotation axis
        rm_single_bond: whether the dangling bonds are removed
        """
        layer_orientations = [0, 30]
        layer_origins = [overlap, overlap]
        self.twisted_bilayer(R+2, 30, h, a, overlap, rm_dangling=False)
        ######################## informations of regular polygon ###############################
        vertices = get_vertices_regular_polygon(n, self.a*(R+1.e-4))
        if OV_orientation:
            vertices = rotate_on_vec(OV_orientation, vertices)
        layer_vertices = [vertices, vertices]
        self._get_struct_and_pointgroup(layer_origins, layer_orientations, layer_vertices, a, h, rm_dangling)

    def regular_polygon_AA(self, n, R, overlap='atom-atom', rm_dangling=True, a=2.46, h=3.35):
        """
        overlap: 'atom-atom', 'atom-hole' or 'hole-atom' for AB 
                 'hole-hole' for AA
        
        """
        if overlap not in ['atom-atom','hole-hole']:
            raise ValueError('For AA stacking, only atom-atom and hole-hole are allowed!')
        layer_orientations = [0, 0]
        layer_origins = overlap.split('-') 
        vertices = get_vertices_regular_polygon(n, a*(R+1.e-4))
        layer_vertices = [vertices, vertices]
        self._get_struct_and_pointgroup(layer_origins, layer_orientations, layer_vertices, a, h, rm_dangling)

    def regular_polygon_AB(self, n, R, overlap='atom-atom', stacking='AB', rm_dangling=True, a=2.46, h=3.35):
        """
        overlap: 'atom-atom', 'atom-hole' or 'hole-atom' for AB 
                 'hole-hole' for AA
        
        """
        if overlap not in ['atom-atom','atom-hole','hole-atom']:
            raise ValueError('For AB stacking, only atom-atom, atom-hole and hole-atom are allowed!')
        layer_orientations = [0, 0]
        layer_origins = overlap.split('-') 
        if overlap == 'atom-atom':
            layer_orientations = [0, 60]
        vertices = get_vertices_regular_polygon(n, a*(R+1.e-4))
        layer_vertices = [vertices, vertices]
        self._get_struct_and_pointgroup(layer_origins, layer_orientations, layer_vertices, a, h, rm_dangling)

