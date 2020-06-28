import numpy as np
import copy
from functools import reduce
from tBG.utils import frac2cart, cart2frac, rotate_on_vec, mirror_operate_2d
from tBG.round_disk import Structure

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

class Lattice:
    def __init__(self, latt_vec, sites_frac):
        self.latt_vec = np.array(latt_vec)
        self.sites_frac = np.array(sites_frac)
        self.sites_cart = frac2cart(sites_frac, latt_vec)

class _MethodsHamilt:
    def get_Hamiltonian(self):
        ndim = len(self.coords)
        H = np.zeros((ndim,ndim), dtype=float)
        def put_value(pair, t):
            H[pair[0],pair[1]] =  t
            H[pair[1],pair[0]] =  t
        pairs, ts = self.hopping
        tmp = [put_value(pairs[i], ts[i]) for i in range(len(ts))]
        return H

    def get_current_mat(self):
        c = 1.0
        ndim = len(self.coords)
        jx = np.zeros((ndim,ndim), dtype=float)
        jy = np.zeros((ndim,ndim), dtype=float)
        jz = np.zeros((ndim,ndim), dtype=float)
        def put_value(pair):
            rij = self.coords[pair[0]]-self.coords[pair[1]]
            jx[pair[0],pair[1]], jy[pair[0],pair[1]], jz[pair[0],pair[1]] =  rij
            jx[pair[1],pair[0]], jy[pair[1],pair[0]], jz[pair[1],pair[0]] =  -rij
        pairs, ts = self.hopping
        tmp = [put_value(pairs[i]) for i in range(len(ts))]
        return c*np.array([jx, jy, jz])

class _SymmetryOperate:

    @staticmethod
    def _inds_pair(coords_0, coords_1):
        """
        coords_0: coords without symmetry operation
        coords_1: coords with symmetry operation
        return:
            a list [m, n, p, q, ......]
            atom positions changes operation:
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
        ops = {}
        ops['Cns'] = self.Cns()
        ops['C2s'] = self.C2s()
        pg = self.get_point_group()
        if pg in ['D2h', 'D3h', 'D6h']:
            ops['sigma_vs'] = self.sigma_vs()
            ops['Sns'] = self.Sns()
            ops['sigma_h'] = [self.sigma_h()]
        ops_inds = {i:self._inds_pairs(self.coords, ops[i]) for i in ops}
        return ops_inds

    def _coords(self, coords):
        if coords is None:
            coords = copy.deepcopy(self.coords)
        else:
            coords = copy.deepcopy(coords)
        return coords

    def Cns(self, coords=None):
        """
        Description: all rotations related to n-fold principal axis
        operators are C_m^n for n in range(m) 
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
        y0 = self.h/2
        coords_0 = self._coords(coords)
        coords_0[:,2] = -( coords_0[:,2] - y0 ) + y0
        return coords_0

    def sigma_vs(self, coords=None):
        coords_0 = self._coords(None)
        zs = coords_0[:,2]
        axes = self.all_C2_axes(self.orient)
        ms = [get_line_through_two_points([0,0], axis) for axis in axes]
        coords_end = []
        for m in ms:
            coords_m = mirror_operate_2d(m, coords_0)
            coords_m = np.append(coords_m, zs.reshape(-1,1), axis=1)    
            coords_end.append(coords_m)
        return coords_end

    def Sns(self, coords=None):
        """
        Sns = Cns * sigma_h
        """
        coords_0 = self._coords(coords)
        coords_Cns = self.Cns(coords_0)
        coords_Cns_sigma_h = [self.sigma_h(coords) for coords in coords_Cns]
        return np.array(coords_Cns_sigma_h)

    def C2s(self, coords=None):
        """
        C2s = sigma_h * sigma_v
        """
        coords_0 = self._coords(coords)
        coords_sigma_v = self.sigma_vs(coords_0)
        coords_sigma_v_sigma_h = [self.sigma_h(coords) for coords in coords_sigma_v]
        return np.array(coords_sigma_v_sigma_h)

class _GemetryOperate:
    def get_layer_nsites(self):
        nsite_unrott = len(np.where(self.coords[:,2]==0.)[0])
        nsite_rott = len(np.where(self.coords[:,2]>0.)[0])
        return [nsite_unrott, nsite_rott]        

    def pmg_molecule(self):
        from pymatgen.core.structure import Molecule
        nsite = len(self.coords)
        molecu = Molecule(['C']*nsite, self.coords)
        return molecu
    
    def pmg_struct(self):
        from pymatgen.core.structure import Structure
        coords = copy.deepcopy(self.coords)
        xmin, ymin, _ = np.min(coords, axis=0)
        xmax, ymax, _ = np.max(coords, axis=0)
        coords[:,0] = coords[:,0]-xmin+20
        coords[:,1] = coords[:,1]-ymin+20
        coords[:,2] = coords[:,2]+8
        nsite = len(coords)
        latt = [[xmax-xmin+40,0,0],[0,ymax-ymin+40,0],[0,0,20]]
        return Structure(latt, ['C']*nsite, coords, coords_are_cartesian=True)

class QuantumDot(Structure, _MethodsHamilt, _GemetryOperate, _SymmetryOperate):
    def __init__(self, a=2.46, h=3.35):
        self.a = a
        self.h = h

    def round_disk(self, R, twist_angle, overlap='hole', rm_single_bond=False):
        self.make_structure(R, rotation_angle=twist_angle, a=self.a, h=self.h, \
                                  overlap=overlap, rm_dangling=rm_single_bond)
        self.twist_angle = twist_angle

    def rectangle(self, w, h, twist_angle, overlap='side_center', rm_single_bond=True):
        self.nfold = 2
        R = np.sqrt(w**2/4+h**2/4)
        self.round_disk(R+2, twist_angle, overlap=overlap)
        self.orient = 'armchair'
        ## get vertices of rectangle
        axes = self.all_C2_axes('armchair')
        p_m0 = (w/2+1.e-3)*self.a*axes[0]
        p_m1 = (h/2+1.e-3)*self.a*axes[1]
        vert0 = p_m0 + p_m1
        vert1 = -p_m0 + p_m1
        vert2 = -p_m0 - p_m1
        vert3 = p_m0 - p_m1
        vertices = np.array([vert0, vert1, vert2, vert3])
        ## get all sides
        sides = QuantumDot.get_sides_from_vertices(vertices)
        # get atoms inside polygon 
        self.coords = QuantumDot.get_atoms_inside_polygon(self.coords, sides)

        self.layer_nsites = self.get_layer_nsites()
        if rm_single_bond:
            self._remove_single_bonds()

    def _C2_1st_axis(self, orient):
        """
        get vector of the 1st 2-fold axis 
        """
        if orient == 'armchair':
            vec = self.latt_bottom[0]+self.latt_bottom[1]
        elif orient == 'zigzag':
            vec = self.latt_bottom[0]
        vec = vec/np.linalg.norm(vec)
        return rotate_on_vec(self.twist_angle/2, vec)

    def all_C2_axes(self, orient_1st):
        """
        orient_1st: the orient of the 1st 2-fold axis 'armchair or zigzag'
        """
        axis0 = self._C2_1st_axis(orient_1st)
        axes = []
        for i in range(self.nfold):
            axis_i = rotate_on_vec(180/self.nfold*i, axis0)
            axes.append(axis_i)
        return np.array(axes)

    @staticmethod
    def get_sides_from_vertices(vertices):
        sides = []
        for i in range(-1, len(vertices)-1):
            side_i = get_line_through_two_points(vertices[i], vertices[i+1])
            sides.append(side_i)
        return sides

    @staticmethod
    def get_atoms_inside_polygon(coords, sides):
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

        
    def regular_polygon(self, n, R, twist_angle, overlap='hole', rm_single_bond=True, orient='armchair'):
        """
        n: the regular polygon with n sides
        R: the distance from center to vertex (in units of a: the lattice constant of graphene)
        twist_angle: the twist angle between two layers
        rm_single_bond: whether the atom with only one negibors are removed
        """
        if overlap not in ['atom', 'hole', 'atom1']:
            raise ValueError('Overlap %s is not recogenized!' % overlap)

        if orient not in ['armchair', 'zigzag']:
            raise ValueError('Oriention %s is not recogenized!' % orient)
        
        ## Four following lines make sure the highest symmetry
        if n==3 and orient=='zigzag' and overlap!='hole':
            raise ValueError('For Triangle and zigzag orientation, overlap can only be hole!')

        if n in [6,12] and overlap!='hole':
            raise ValueError('For Hexgon or Decagon, overlap can only be hole!')

        if n not in [3, 6, 12]:
            print('Warnning: n != 3, 6, or 12, only structure is reasonable!!')

        self.round_disk(R+2, twist_angle, overlap=overlap)
        self.nfold = n
        self.orient = orient

        ### get new R to include sites of both layers at R
        theta_inter = (n-2)*180/n
        def d2r(degree):
            return degree*np.pi/180
        theta_1 = d2r(theta_inter/2)
        theta_2 = d2r(twist_angle/2)
        #R_new = R/np.sin(theta_1)*np.sin(np.pi-theta_1-theta_2) + 1.e-5
        R_new = R + 1.e-5

        # profile of regular polygon with n sides
        def _vertices(R):
            axis_C2_1st = self._C2_1st_axis(orient)
            vert0 = R*axis_C2_1st*self.a
            vertices = []
            for i in range(self.nfold):
                theta_i = 360/self.nfold*i
                vertices.append(rotate_on_vec(theta_i, vert0))
            return vertices

        vertices = _vertices(R_new)
        sides = QuantumDot.get_sides_from_vertices(vertices)

        # get atoms inside polygon 
        self.coords = QuantumDot.get_atoms_inside_polygon(self.coords, sides)

        self.layer_nsites = self.get_layer_nsites()
        if rm_single_bond:
            self._remove_single_bonds()

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

    def _plot_C2_axes(self, fig, ax):
        self.plot(fig, ax, site_size=3.0, dpi=600, lw=0.6, edge_cut=0)
        axes = self.all_C2_axes(self.orient)
        R = np.max(np.linalg.norm(self.coords[:,0:2], axis=1))
        for axis in axes:
            p = R*axis
            ax.plot([-p[0],p[0]],[-p[1],p[1]], color='blue', ls='dashed')

    def _remove_single_bonds(self):

        while True:
            pmg_st = self.pmg_struct()
            bonds = pmg_st.get_neighbor_list(2.0)[0]
            counts = np.array([np.count_nonzero(bonds==i) for i in range(len(self.coords))])
            inds_rm = np.where(counts<=1)[0]
            inds = np.in1d(range(pmg_st.num_sites), inds_rm)
            self.coords = self.coords[~inds]
            self.layer_nsites = self.get_layer_nsites()
            if not len(inds_rm):
                break
