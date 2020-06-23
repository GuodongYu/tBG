import numpy as np
import itertools

def get_fig(nr=1, nc=1):
    from matplotlib import pyplot as plt
    return plt.subplots(nr, nc)

def frac2cart(frac, latt_vec):
    return np.matmul(frac, latt_vec)
    #return np.dot(np.transpose(np.array(latt_vec)), frac)

def cart2frac(cart, latt_vec):
    mat = np.transpose(np.array(latt_vec))
    cart_ = np.transpose(np.array(cart))
    frac = np.linalg.solve(mat, cart_)
    return np.transpose(frac)

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

def grouper(in_list, n):
    """Make n sublists from list.

    Parameters
    ----------
    in_list : list
        input list
    n : integer
        max number of sublists to return

    Returns
    ----------
    sublists : iterable
        iterable over sublists
    n_sublists : integer
        number of sublists
    """

    n_list = len(in_list)
    in_list = iter(in_list)
    len_sublists = int(np.ceil(1. * n_list / n))
    if (len_sublists * (n - 1) == n_list):
        # in this case the nth sublist is empty, so:
        n_sublists = n - 1
    else:
        n_sublists = n
    sublists = iter(lambda: list(itertools.islice(in_list, len_sublists)), [])
    return sublists, n_sublists

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
