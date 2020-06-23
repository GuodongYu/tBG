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

    def rotate_Cms(self, m=3):
        """
        Description: m-fold rotation (main axis along z-axis) axis
        operators are C_m^n for n in range(m) 
        m = 3 (triangle) and 6 (hexegon)
        """
        pairs = []
        for n in range(1, m):
            theta = 360/m*n
            coords_0 = copy.deepcopy(self.coords)
            coords_1 = rotate_on_vec(theta, coords_0)
            pairs_n = _SymmetryOperate._inds_pair(coords_0, coords_1)
            pairs.append(pairs_n)
        return np.array(pairs)

    def rotate_C2s_D2(self):
        """
        three C2 rotations with rotation axes in xy plane and 120 degree orientations seperated
        axis_orient: the angle with x-axis of one 2-fold rotation axis
        Note: For bottom 
        """
        def get_mirrors(twist_angle):
            p_orig = self.latt_bottom[0]+self.latt_bottom[1]
            theta0 = twist_angle/2 
            theta1 = twist_angle/2 + 90
            p0 = rotate_on_vec(theta0, p_orig)
            p1 = rotate_on_vec(theta1, p_orig)
            orig = np.array([0., 0.])
            m0 = get_line_through_two_points(orig, p0)
            m1 = get_line_through_two_points(orig, p1)
            return m0, m1

        if len(self.layer_nsites)==2:# the case for twist bilayer QD
            m0, m1 = get_mirrors(self.twist_angle)
            zs = np.array([self.h]*self.layer_nsites[1] + [0.]*self.layer_nsites[0])
        elif len(self.layer_nsites)==1: # the case for graphene QD
            m0, m1 = get_mirrors(0.)
            zs = np.array([0.]*self.layer_nsites[0])
        
        def inds_after_mirror(m):
            coords_mirror = mirror_operate_2d(m, self.coords)
            coords_mirror = np.append(coords_mirror, zs.reshape(-1,1), axis=1)    
            return _SymmetryOperate._inds_pair(self.coords, coords_mirror)

        return np.array([inds_after_mirror(m0), inds_after_mirror(m1)])

    def rotate_C2s_D3(self):
        return self._rotate_three_C2s(axis=self.orient)

    def rotate_C2s_D6(self):
        inds0 = self._rotate_three_C2s(axis='armchair')
        inds1 = self._rotate_three_C2s(axis='zigzag')
        return np.append(inds0, inds1, axis=0)

    def _rotate_three_C2s(self, axis='armchair'):
        """
        three C2 rotations with rotation axes in xy plane and 120 degree orientations seperated
        axis_orient: the angle with x-axis of one 2-fold rotation axis
        Note: For bottom 
        """
        if axis == 'armchair':
            axis_orient = 0.
        elif axis == 'zigzag':
            axis_orient = 30.
        def get_mirrors(twist_angle):
            p_orig = self.latt_bottom[0]+self.latt_bottom[1]
            theta0 = twist_angle/2 + axis_orient
            theta1 = twist_angle/2 + 120. + axis_orient
            theta2 = twist_angle/2 + 240. + axis_orient
            p0 = rotate_on_vec(theta0, p_orig)
            p1 = rotate_on_vec(theta1, p_orig)
            p2 = rotate_on_vec(theta2, p_orig)
            orig = np.array([0., 0.])
            m0 = get_line_through_two_points(orig, p0)
            m1 = get_line_through_two_points(orig, p1)
            m2 = get_line_through_two_points(orig, p2)
            return m0, m1, m2

        if len(self.layer_nsites)==2:# the case for twist bilayer QD
            m0, m1, m2 = get_mirrors(self.twist_angle)
            zs = np.array([self.h]*self.layer_nsites[1] + [0.]*self.layer_nsites[0])
        elif len(self.layer_nsites)==1: # the case for graphene QD
            m0, m1, m2 = get_mirrors(0.)
            zs = np.array([0.]*self.layer_nsites[0])
        
        def inds_after_mirror(m):
            coords_mirror = mirror_operate_2d(m, self.coords)
            coords_mirror = np.append(coords_mirror, zs.reshape(-1,1), axis=1)    
            return _SymmetryOperate._inds_pair(self.coords, coords_mirror)

        return np.array([inds_after_mirror(m0), inds_after_mirror(m1), inds_after_mirror(m2)])
        

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

    def _system_rotate(self, theta):
        pass

    def round_disk(self, R, twist_angle, overlap='hole'):
        self.make_structure(R, rotation_angle=twist_angle, a=self.a, h=self.h, \
                                  overlap=overlap, rm_dangling=False)
        self.twist_angle = twist_angle

    def rectangle(self, w, h, twist_angle, overlap='side_center', rm_single_bond=True):
        R = np.sqrt(w**2/4+h**2/4)
        self.round_disk(R+2, twist_angle, overlap=overlap) 
        x0 = w/2*self.a
        y0 = h/2*self.a
        p0_unrott = [x0, y0]
        p1_unrott = [-x0, y0]
        p2_unrott = [-x0, -y0]
        p3_unrott = [x0, -y0]
        ps_unrott = np.array([p0_unrott, p1_unrott, p2_unrott, p3_unrott])
        if twist_angle==0.:
            verts = ps_unrott
        else:
            ps_rott = rotate_on_vec(twist_angle, ps_unrott)
            lines0 = [get_line_through_two_points(ps_unrott[i%4], ps_unrott[(i+1)%4]) for i in range(4)]
            lines1 = [get_line_through_two_points(ps_rott[i-1], ps_rott[i]) for i in range(4)]
            verts = np.array([get_interaction_two_lines(lines0[i], lines1[i]) for i in range(4)])
        lines = [get_line_through_two_points(verts[i], verts[(i+1)%4]) for i in range(4)]

        inds = site_inds_relative_a_line_including_specific_pnt(self.coords, lines[0], np.array([0,0]))
        for i in range(1, len(lines)):
            line = lines[i]
            inds_tmp = site_inds_relative_a_line_including_specific_pnt(self.coords, line, np.array([0,0]))
            inds = np.intersect1d(inds, inds_tmp) 
        self.coords = self.coords[inds]       
        self.layer_nsites = self.get_layer_nsites()
        if rm_single_bond:
            self._remove_single_bonds()
        
    def regular_polygon(self, n, R, twist_angle, overlap='hole', rm_single_bond=True, orient='armchair'):
        """
        n: the regular polygon with n sides
        R: the distance from center to vertex (in units of a: the lattice constant of graphene)
        twist_angle: the twist angle between two layers
        rm_single_bond: whether the atom with only one negibors are removed
        """
        self.round_disk(R+2, twist_angle, overlap=overlap)

        #R = self.a*R
        theta = 360/n
        self.orient = orient
        if orient == 'armchair':
            p0_unrott = R/np.sqrt(3)*(self.latt_bottom[0]+self.latt_bottom[1])
        elif orient == 'zigzag':
            p0_unrott = self.latt_bottom[0]*R
            
        p1_unrott = rotate_on_vec(theta, p0_unrott)
        line0 = get_line_through_two_points(p0_unrott, p1_unrott)
        a1, b1, c1 = line0

        ############################################################
        if self.twist_angle == 0.0:
            p = p0_unrott
        else:
            p0_rott = rotate_on_vec(twist_angle, p0_unrott)

            p1_rott_CCW = rotate_on_vec(theta, p0_rott)
            line1_CCW = get_line_through_two_points(p0_rott, p1_rott_CCW)
            a2_CCW, b2_CCW, c2_CCW = line1_CCW
            p0_intsec = get_interaction_two_lines([a1,b1,c1], [a2_CCW,b2_CCW,c2_CCW])

            p1_rott_CW = rotate_on_vec(-theta, p0_rott)
            line1_CW = get_line_through_two_points(p0_rott, p1_rott_CW)
            a2_CW, b2_CW, c2_CW = line1_CW
            p1_intsec = get_interaction_two_lines([a1,b1,c1], [a2_CW,b2_CW,c2_CW])

            p = p0_intsec if np.linalg.norm(p0_intsec)>=np.linalg.norm(p1_intsec) else p1_intsec

        lines = []
        for i in range(n):
            p0 = rotate_on_vec(theta*i, p)
            p1 = rotate_on_vec(theta*(i+1), p)
            lines.append(get_line_through_two_points(p0, p1))


        inds = site_inds_relative_a_line_including_specific_pnt(self.coords, lines[0], np.array([0,0]))
        for i in range(1, len(lines)):
            line = lines[i]
            inds_tmp = site_inds_relative_a_line_including_specific_pnt(self.coords, line, np.array([0,0]))
            inds = np.intersect1d(inds, inds_tmp) 
        self.coords = self.coords[inds]       
        self.layer_nsites = self.get_layer_nsites()
        if rm_single_bond:
            self._remove_single_bonds()

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
