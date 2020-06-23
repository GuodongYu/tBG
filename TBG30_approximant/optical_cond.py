import numpy as np
import math
import json
import time
import sys

pi = np.pi

def f2e(x):
    y = '%.1e' % x
    x, y = y.split('e')
    y = int(float(y))
    if float(x) == 1.0:
        return '10^%s' % y
    else:
        return '%sx10^{%s}' % (x, y)

def rotate_operator(theta):
    """
    [0, 0] is the rotation centor
    """
    rad = theta*np.pi/180.
    mat = np.array([[np.cos(rad), -np.sin(rad)],[np.sin(rad), np.cos(rad)]])
    return mat

def rotate_on_vec(theta, vec):
    mat = rotate_operator(theta)
    return np.dot(mat, vec)

def frac2cart(frac, latt_vec):
    return np.dot(np.transpose(np.array(latt_vec)), frac)

def cart2frac(cart, latt_vec):
    mat = np.transpose(np.array(latt_vec))
    frac = np.linalg.solve(mat, cart)
    return frac


class Jk(object):
    """
    the class for getting the current operator  matrix
    in tight-binding model
    """
    def __init__(self, struct, direction=0):
        """
        struct: the Struct calss object with hopping
        E_on: the onsite energy 
        direction: the direction of the Jk operator
            0, 1 and 2 correspond to the x, y and z directions
        """
        self.direction = direction
        self.struct = struct
        self.coords = struct.coords
        self.natom = len(self.coords)
        self.norb = self.natom
        self.latt_comp = struct.latt_compatible
        self.J0 = self._get_J0()
        
    def _get_J0(self):
        alpha = self.direction
        J0 = np.zeros((self.norb, self.norb), dtype=complex)
        hopping = self.struct.hopping['0_0']
        for pair in hopping:
            i,j = [int(w) for w in pair.split('_')]
            J0[i,j] = 1j*hopping[pair]*(self.coords[i][alpha]-self.coords[j][alpha])
            J0[j,i] = 1j*hopping[pair]*(self.coords[j][alpha]-self.coords[i][alpha])
        return J0

    def _Jk(self, k):
        alpha = self.direction
        a0 = self.latt_comp[0]
        a1 = self.latt_comp[1]
        Jk = copy.deepcopy(self.J0)
        for neig_cell in ['0_1','1_0','1_1','-1_1']:
            m,n = [int(i) for i in neig_cell.split('_')]
            R = m*a0 + n*a1
            hopping = self.struct.hopping[neig_cell]
            for pair in hopping:
                i,j = [int(i) for i in pair.split('_')]
                rj = R + self.coords[j][:-1]
                R_dot_k = np.dot(k, R)
                Jk[i,j] = Jk[i,j] + 1j*np.exp(1j*R_dot_k) * hopping[pair] * (self.coords[i][alpha] - rj[alpha])
                Jk[j,i] = Jk[j,i] + 1j*np.exp(-1j*R_dot_k) * hopping[pair] * (rj[alpha] - self.coords[i][alpha])
        return Jk

