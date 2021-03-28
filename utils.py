import numpy as np
from pymatgen.core.structure import Structure
import itertools


def frac2cart(frac, latt_vec):
    return np.matmul(frac, latt_vec)

#def cart2frac(cart, latt_vec):
#    mat = np.transpose(np.array(latt_vec))
#    cart_ = np.transpose(np.array(cart))
#    frac = np.linalg.solve(mat, cart_)
#    return np.transpose(frac)

def cart2frac(cart, latt_vec):
    return np.matmul(cart, np.linalg.inv(latt_vec))

def rotate_operator(theta, dim=2):
    """
    [0, 0] is the rotation centor
    z-axis is rotation axis
    """
    rad = theta*np.pi/180.
    mat = np.array([[np.cos(rad), -np.sin(rad)],[np.sin(rad), np.cos(rad)]])
    if dim==2:
        return mat
    elif dim==3:
        return np.array([[mat[0][0], mat[0][1], 0],
                         [mat[1][0], mat[1][1], 0],
                         [0, 0, 1]])

def rotate_on_vec(theta, vec):
    vec = np.array(vec)
    if len(vec.shape)==1: ## one vector
        mat = rotate_operator(theta, dim=len(vec))
        return np.matmul(mat, np.array(vec).reshape(-1,1)).reshape(2,)
    elif len(vec.shape)==2: ## vector list
        mat = rotate_operator(theta, dim=vec.shape[1])
        return np.matmul(mat, np.array(vec).T).T

def rotate_a_line(theta, line):
    prec = 1.e-10
    a, b, c = line
    if abs(a)<=prec:
        p0 = [1, -c/b]
        p1 = [2, -c/b]
    else:
        p0 = [-c/a, 0]
        p1 = [-(b+c)/a, 1]
    p0_new = rotate_on_vec(theta, p0)
    p1_new = rotate_on_vec(theta, p1)
    if abs(p0_new[0]-p1_new[0])<=prec:
        x = p0_new[0]
        a_new, b_new, c_new = 1, 0, -x
    else:
        k = (p1_new[1]-p0_new[1])/(p1_new[0]-p0_new[0])
        b = p1_new[1] - k*p1_new[0]
        a_new, b_new, c_new = k, -1, b
    return [a_new, b_new, c_new]


def angle_between_vec_x(vec):
    """
    get the angle in degree between a in-plane vec and x axis 
    """
    if np.abs(vec[0]) <= 1.0e-12:
        if vec[1] > 0.:
            return 90.
        elif vec[1] < 0.:
            return 270.
    if np.abs(vec[1]) <= 1.0e-12:
        if vec[0] > 0.:
            return 0.
        elif vec[0] < 0.:
            return 180.
    theta = np.arctan(vec[1]/vec[0]) /np.pi * 180.
    if vec[0]*vec[1]>0.:
        if vec[0] > 0.:
            return theta
        elif vec[0] <0.:
            return 180.+theta
    elif vec[0]*vec[1] <0.:
        if vec[0] >0.:
            return theta + 360.
        elif vec[0] <0.:
            return theta + 180.

def mirror_operate_2d(mirror_line, points):
    """
    mirror_line: the mirror line in xy plane ax + by + c = 0 
    """
    a, b, c = mirror_line
    def mirrorImaged(x1, y1):
        temp = -2 * (a * x1 + b * y1 + c) /(a * a + b * b)
        x = temp * a + x1
        y = temp * b + y1
        return (x, y)
    xs, ys = mirrorImaged(points[:,0],points[:,1])
    return np.array([xs, ys]).T


def rotate_angle(vec):
    """
    the rotate angle of the vec relative to the x axis
    """
    if np.abs(vec[0]) <= 1.0e-12:
        if vec[1] > 0.:
            return 90.
        elif vec[1] < 0.:
            return 270.
    if np.abs(vec[1]) <= 1.0e-12:
        if vec[0] > 0.:
            return 0.
        elif vec[0] < 0.:
            return 180.
    theta = np.arctan(vec[1]/vec[0]) /np.pi * 180.
    if vec[0]*vec[1]>0.:
        if vec[0] > 0.:
            return theta
        elif vec[0] <0.:
            return 180.+theta
    elif vec[0]*vec[1] <0.:
        if vec[0] >0.:
            return theta + 360.
        elif vec[0] <0.:
            return theta + 180. 

class MatrixExpand:
    """
    expand a matrix by replacing each scalar element with an matrix
    """
    def __init__(self, mat, ndim_unit):
        """
        mat: the original matrix
        ndim_unit: the dimension of the  
        """
        self.mat_orig = copy.deepcopy(mat)

def get_pmg_strut(latt_vec_2D, fracs):
    latt_vec = latt_vec_2Dto3D(latt_vec_2D)
    fracs = np.append(fracs, [[0.]]*len(fracs), axis=1)
    return Structure(latt_vec, ['C']*len(fracs), fracs)

def latt_vec_2Dto3D(latt_vec_2D):
    latt_vec = np.append(latt_vec_2D, [[0., 0.,]], axis=0)
    latt_vec = np.append(latt_vec, [[0.], [0.], [100.]], axis=1)
    return latt_vec

def get_bonding_lines_for_plot_in_matplotlib(pmg_st, bond_length):
    a,b,c,d = pmg_st.get_neighbor_list(bond_length)
    pnts_a = pmg_st.cart_coords[a][:,0:2]
    frac_b = c + pmg_st.frac_coords[b]
    pnts_b = frac2cart(frac_b, pmg_st.lattice.matrix)[:,0:2]
    lines = [[pnts_a[i], pnts_b[i]] for i in range(len(a))]
    return lines

def lattice_plot_in_xy_in_groups(fig, ax, pmg_st, group_inds, colors, bond_lengths, sc=[5,5,1]):
    """
    plot a lattice (in real or reciprocal space) including sites and bonds 
    with different groups in different colors.
    inputs:
        fig and ax are the figure and subplot of matplotlib
        pmg_st: a pymatgen structure (the instance of pymatgen.core.structure.Structure)
        group_inds: a list of atom indices for different groups
        colors: a list of colors for plotting different groups
        bond_lengths: a list of bond lengths for different groups
        sc: the supercell size for plotting
    """
    import matplotlib.collections as mc
    site_properties = [ None ] * pmg_st.num_sites
    for i in range(len(group_inds)):
        for ind in group_inds[i]:
            site_properties[ind] = i
        
    pmg_st.add_site_property('group',site_properties)
    pmg_st.make_supercell(sc)
    for i in range(len(group_inds)):
        indices = np.where(np.array(pmg_st.site_properties['group'])==i)[0]
        coords = pmg_st.frac_coords[indices]
        eles = np.array(pmg_st.species)[indices]
        pmg_st_i = Structure(pmg_st.lattice, eles, coords)
        ax.scatter(pmg_st_i.cart_coords[:,0], pmg_st_i.cart_coords[:,1], color=colors[i])
        bonds = get_bonding_lines_for_plot_in_matplotlib(pmg_st_i, bond_lengths[i])
        line = mc.LineCollection(bonds, colors=colors[i])
        fig.canvas.draw()
        renderer = fig.canvas.renderer
        ax.add_collection(line)
        ax.draw(renderer)
