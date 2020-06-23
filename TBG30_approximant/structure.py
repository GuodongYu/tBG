import numpy as np
import numpy.linalg as npla
import math
import json
import time
import copy
from monty.json import jsanitize
import struct
import matplotlib as mpl
import pandas as pd


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

class Structure(object):
    def read_from_file(self, filename):
        """
        read the xyz file for site coords. 
        don't forget to add hopping manually after reading.
        """
        with open(filename) as f:
            f.readline()
            f.readline()
            f.readline()
            nsite = int(f.readline())
        data = read_last_n_lines(filename, n=nsite+4)
        latt_info = data[:3]
        print(latt_info)
        coord_info = np.array([i.split() for i in data[4:]])
        atom_ids = np.array(coord_info[:,0], dtype=int)
        coord_info = coord_info[atom_ids.argsort()]
        xlo_bound, xhi_bound, xy = [float(i) for i in latt_info[0].split()]
        ylo_bound, yhi_bound, xz = [float(i) for i in latt_info[1].split()]
        zlo_bound, zhi_bound, yz = [float(i) for i in latt_info[2].split()]
        xlo = xlo_bound - min(0.0,xy,xz,xy+xz)
        xhi = xhi_bound - max(0.0,xy,xz,xy+xz)
        ylo = ylo_bound - min(0.0,yz)
        yhi = yhi_bound - max(0.0,yz)
        zlo = zlo_bound
        zhi = zhi_bound
        a = (xhi-xlo,0,0)
        b = (xy,yhi-ylo,0)
        c = (xz,yz,zhi-zlo)
        latt = np.array([a,b,c])
        coords_frac = np.array(coord_info[:,2:], dtype=float)
        
        self.coords = np.matmul(latt.T, coords_frac.T).T
        self.latt_compatible = latt[:-1,:-1]
        self.a_top = 2.456
        self.a = 2.456
        self.h = 3.461

        atom_types =  set(coord_info[:,1])
        nlayer = len(atom_types)
        self.layer_types = ['A']*nlayer
        self.layer_nsites=[]
        for i in atom_types:
            self.layer_nsites.append(np.count_nonzero(coord_info[:,1]==i))
        del coord_info

    def read_from_POSCAR_file(self, poscar_file):
        from pymatgen.io.vasp import Poscar
        pos = Poscar.from_file(poscar_file)
        self.latt_compatible = pos.structure.lattice.matrix[:-1,:-1]
        self.coords = pos.structure.cart_coords
        self.a = 2.456
        self.a_top = 2.456
        z_coords = np.unique(self.coords[:,-1])
        atom_types = range(1,len(z_coords)+1)
        self.layer_types = ['A']*len(z_coords)
        self.layer_nsites=[len(np.where(self.coords[:,-1]==i)[0]) for i in z_coords]
        self.layer_zcoords = z_coords    

        
        

    def output_poscar(self):
        z_coords = self.coords[:,-1]
        min_z = np.min(z_coords)
        self.coords[:,-1] = self.coords[:,-1] - min_z + 1
        coord_str = np.array(self.coords, dtype=str)
        coord_str = '\n'.join([' '.join(i) for i in coord_str])
        with open('POSCAR', 'w') as f:
            f.write('poscar\n')
            f.write('   1.0\n')
            f.write(' %s %s 0.00\n' % (self.latt_compatible[0][0], self.latt_compatible[0][1]))
            f.write(' %s %s 0.00\n' % (self.latt_compatible[1][0], self.latt_compatible[1][1]))
            f.write(' 0.00 0.00 %s\n' % (max(z_coords)+10))
            f.write(' C\n')
            f.write(' %s\n' % len(self.coords))
            f.write('Cartesian\n')
            f.write(coord_str)
            

    def make_structure(self, n_bottom, a=2.456, h=3.461):
        """
        Class for constructing periodic bilayer graphene quasicrystal with hoppings.

        Details:
            setup the M/N approximant of bilayer graphene quasicrystal (30 degree TBG)
            The bottom layer is placed on the xy (z=0) plane, and its a1 lattice vector along x axis, and its a2 lattice
            vector has the anticlockwise 60 degree with respective to a1. The top layer is placed on the z=h plane, and 
            it rotates 30 degree anticlockwisely relative to the bottom layer.
        Args:
            n_bottom: the M value for a M/N approximant
            the periodic bilayer graphene quasicrystal is n_bottom*3b x n_bottom*3b / n_top*a x n_top*a, namely
                n_bottom*3b = n_top*a
            h: the interlayer distance, default 3.461 obtained from relaxation using Lammps
            a: the lattice constant for bottom layer (pristine graphene), default 2.456

        Units:
            distance: Angstrom 
            hopping: eV
        
        Attributations:
            self.hopping: the hopping values calculated from the real site distances and orientation
            self.hopping_rescale:  the intralayer hopping in Atld and Btld recover Atld and Btld are prinstine graphene
                                   but the inerlayer hopping and intralayer hopping in A and B layers remain
            **Note: after application of function self.change_hops_intld_to_pristine_G(), self.hopping_rescale will be deleted
                    and self.hopping will save the values of self.hopping_rescale
            self.a: lattice constant of a prinstine graphene
            self.h: the interlayer distance
            self.a_top: lattice constant of Atld and Btld layers
            self.coords: the coords of all sites
            self.n_bottom: the M value of M/N approximant
            self.n_top: the N value of M/N approximant
            self.species: the site type 1, 2 for A and B sublattice
            self.layer_nsites_no_disorder: the number of sites of each without disorder
            self.layer_nsites: the nunber of sites of each layer (after add vacancy disorder it can change)
            self.layer_types: the type of each layer (A, B, Atld or B tld)
            self.layer_zcoords: the z coord of each layer
        """
        self.units={'energy':'eV','length':'angstrom'}
        self.a = a
        b = a/np.sqrt(3.)
        self.n_bottom = n_bottom # M value       
        self.n_top = int(round(np.sqrt(3)*n_bottom)) # N value
        a_top = 3*b*n_bottom / self.n_top
        self.a_top = a_top
        print('n_bottom: %s; n_top: %s' % (self.n_bottom, self.n_top)) 
        print('a_bottom: %s A; a_top: %s A' % (a, a_top))
        if math.gcd(self.n_bottom, self.n_top) != 1:
            raise ValueError('n_bottom and n_top share common dividor!')
        if not self.n_top%3:
            raise ValueError('n_top is times of 3!')
        self.h = h
        va_bottom = np.array([3*b, 0.])
        vb_bottom = np.array([3*b*np.cos(pi/3.), 3*b*np.sin(pi/3.)])
        s0_b = np.array([1/3., 0.])
        s1_b = np.array([2/3., 0.])
        s2_b = np.array([2/3., 1/3.])
        s3_b = np.array([1/3., 2/3.])
        s4_b = np.array([0., 2/3.])
        s5_b = np.array([0., 1/3.])

        va_top = np.array([a_top, 0.])
        vb_top = np.array([a_top*np.cos(pi/3.), a_top*np.sin(pi/3.)])
        s0_t = np.array([1/3., 1/3.])
        s1_t = np.array([2/3., 2/3.])

        self.latt_bottom = np.array([va_bottom, vb_bottom])
        self.sites_bottom_frac = np.array([s0_b, s1_b, s2_b, s3_b, s4_b, s5_b])
        self.latt_top = np.array([va_top, vb_top])
        self.sites_top_frac = np.array([s0_t, s1_t])
        self.latt_compatible, self.coords, nsites_bottom, nsites_top = self._lattice_compatible()      
        self.species = np.array([1,2]*int(len(self.coords)/2))
        self.layer_nsites = [nsites_bottom, nsites_top]
        self.layer_nsites_no_disorder = [nsites_bottom, nsites_top]
        self.layer_zcoords = [self.coords[0][-1], self.coords[-1][-1]]
        self.layer_types = ['A', 'Atld']
        print('nsites: %s' % np.sum(self.layer_nsites))

    def _lattice_compatible(self):
        time0 = time.time()
        latt_compt = self.latt_bottom*self.n_bottom
        latt_bottom = self.latt_bottom
        latt_top = self.latt_top
        sites_bottom_frac = self.sites_bottom_frac
        sites_top_frac = self.sites_top_frac
        coords_bottom_2d = []
        coords_bottom_append= coords_bottom_2d.append
        for i in range(self.n_bottom):
            for j in range(self.n_bottom):
                for s in sites_bottom_frac:
                    coords_bottom_append(frac2cart([s[0]+i, s[1]+j], latt_bottom))
        coords_bottom = np.append(coords_bottom_2d, [[0.0]]*len(coords_bottom_2d), axis=1)
        coords_top = []
        coords_top_append = coords_top.append
        for i in range(self.n_top):
            for j in range(self.n_top):
                for s in sites_top_frac:
                    coords_top_append(frac2cart([s[0]+i, s[1]+j], latt_top))
        coords_top = np.append(coords_top, [[self.h]]*len(coords_top), axis=1)
        nsites_bottom = len(coords_bottom)
        nsites_top = len(coords_top)
        coords = np.append(coords_bottom, coords_top, axis=0)
        time1 = time.time()
        print('Time for coords: %s s' % (time1 -time0))
        return latt_compt, coords , nsites_bottom, nsites_top

    def pymatgen_struct(self):
        from pymatgen.core.structure import Structure as pymat_Struct
        latt = self.latt_compatible 
        latt = np.append(latt, np.array([0,0]).reshape(-1,1), axis=1)
        latt = np.append(latt, [[0,0,100]], axis=0)
        return pymat_Struct(latt, ['C']*np.sum(self.layer_nsites), self.coords, coords_are_cartesian=True)

    def _coords_xy(self, layer):
        """
        layer: 'A', 'B', 'Atld' or 'Btld'
        """
        xy_0 = copy.deepcopy(self.coords[:self.layer_nsites[0]][:,:-1])
        xy_1 = copy.deepcopy(self.coords[self.layer_nsites[0]:self.layer_nsites[0]+self.layer_nsites[1]][:,:-1])
        if layer == 'A':
            return xy_0
        elif layer == 'B':
            xy_0[:,0] = xy_0[:,0] + self.a/np.sqrt(3)
            return xy_0
        elif layer == 'Atld':
            return xy_1
        elif layer == 'Btld':
            xy_1[:,0] = xy_1[:,0] + self.a_top/np.sqrt(3)*np.cos(np.pi/6)
            xy_1[:,1] = xy_1[:,1] + self.a_top/np.sqrt(3)*np.sin(np.pi/6)
            return xy_1

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

        for layer in layers:
            coord_xy = self._coords_xy(layer)
            nsite = len(coord_xy)
            for i in layers[layer]:
                z = self.h*i
                coord = np.concatenate((coord_xy, [[z]]*nsite), axis=1)
                self.coords = np.concatenate((self.coords, coord), axis=0)
                self.layer_nsites.append(nsite)
                self.layer_zcoords.append(z)
                self.layer_types.append(layer)
        self.species = np.array([1,2]*int(len(self.coords)/2))

    def _layer_inds(self):
        layer_inds = []
        for i in range(len(self.layer_nsites)):
            ind0 = sum(self.layer_nsites[:i])
            ind1 = ind0 + self.layer_nsites[i]-1
            layer_inds.append([ind0,ind1])
        return layer_inds
    
    def add_vacancy_disorder(self, con):
        """
        con is the concentration of vacancy
        """
        def which_layer(ind):
            layer_inds = self._layer_inds()
            for i in range(len(layer_inds)):
                if layer_inds[i][0]<=ind<=layer_inds[i][1]:
                    return i
        nsite = np.sum(self.layer_nsites)
        nvac = int(con*nsite)
        print('vacancy number: %s' % nvac)
        vac_inds = np.random.randint(0, nsite, nvac)
        for ind in vac_inds:
            layer = which_layer(ind)
            self.layer_nsites[layer] =  self.layer_nsites[layer] - 1    
        self.coords = np.delete(self.coords, vac_inds, 0)
        self.species = np.delete(self.species, vac_inds, 0)

    def output_Lammps_struct_file(self, bottom_fix=False, atom_style='full', random_z_add=True):
        n_atom_type = len(self.layer_zcoords)
        layer_inds = self._layer_inds()
        latt = self.latt_compatible
        a = latt[0]
        b = latt[1]
        ### basis informations saved to info.csf file ###
        try:
            info = {'atom_id_by_layer':layer_inds, 'layer_zcoords':self.layer_zcoords,\
                             'layer_types':self.layer_types, 'species':self.species}
            with open('info_nbot%s.json' % self.n_bottom, 'w') as f:
                json.dump(jsanitize(info), f)
        except:
            pass

        if bottom_fix:
            nsite_bot = self.layer_nsites[0]
            with open('group.in','w') as f:
                f.write('group bottom id 1:%s\n' % nsite_bot)
                f.write('velocity bottom set 0 0 0\n')
                f.write('fix 0 bottom setforce 0.0 0.0 0.0\n')
        try:
            save_to = 'data.nbot%s' % self.n_bottom
        except:
            save_to = 'data.struct'
        coords = self.coords
        if random_z_add:
            coords[:,-1] = coords[:,-1] +np.random.uniform(-0.1, 0.1, len(coords))

        with open(save_to, 'w') as f:
            try:
                f.write(' 30 TBG approximant %s/%s\n\n' % (self.n_bottom, self.n_top))
            except:
                f.write(' structure info\n\n')
            f.write('    %s atoms\n\n' % len(self.coords))
            f.write('    %s atom types\n\n' % n_atom_type)
            f.write(' %.8f %.8f xlo xhi\n' % (0., a[0]))
            f.write(' %.8f %.8f ylo yhi\n' % (0., b[1]))
            f.write(' %.8f %.8f zlo zhi\n' % (-100., 100.))
            f.write(' %.8f %.8f %.8f xy xz yz\n\n' % (b[0], 0., 0.))
            f.write(' Masses\n\n')
            for i in range(n_atom_type):
                f.write('    %s 12.0107\n' % (i+1))
            f.write('\n')
            f.write(' Atoms\n\n')
            for i in range(len(layer_inds)):
                for j in range(layer_inds[i][0], layer_inds[i][1]+1):
                    if atom_style=='full':
                        f.write('    %s %s %s 0.0 %.8f %.8f %.8f\n' % \
                               (j+1, i+1, i+1, coords[j][0], coords[j][1], coords[j][2]))
                    else:
                        f.write('    %s %s  %.8f %.8f %.8f\n' % (j+1, i+1, coords[j][0], coords[j][1], coords[j][2]))

    def add_hopping(self, max_dist=5.0, g0=3.12, g1=0.48, rc=6.14, lc=0.265, q_dist_scale=2.218):
        time0 = time.time() 
        self.hopping_parameters={}
        self.hopping_parameters['max_dist'] = max_dist
        self.hopping_parameters['g0'] = g0
        self.hopping_parameters['g1'] = g1
        self.hopping_parameters['rc'] = rc
        self.hopping_parameters['lc'] = lc
        self.hopping_parameters['q_dist_scale'] = q_dist_scale
        self.hopping, self.hopping_rescale = self.get_hopping()        
        time1 = time.time()
        print('Time for hopping: %s s' % (time1-time0))

    def hops_func(self, dr, dz):
        """
        the function defines the hopping value between two orbitals
        r is the relative vector between the two orbitals
        Note:
            The units of all quantities below is angstrom
        """
        a0 = 1.42
        h0 = 3.349
        n = dz / dr
        q_dist_scale = self.hopping_parameters['q_dist_scale']
        V_pppi = -self.hopping_parameters['g0'] * np.exp(q_dist_scale * (a0 - dr))
        V_ppsigma = self.hopping_parameters['g1'] * np.exp(q_dist_scale * (h0 - dr))
        hop = n**2 * V_ppsigma + (1 - n**2) * V_pppi
        return hop / (1 + np.exp((dr - self.hopping_parameters['rc']) / self.hopping_parameters['lc'])) # smooth cutoff

    def get_hopping(self):

        ##################################### prepare ################################################
        hopping00 = {} 
        hopping10 = {}
        hopping01 = {}
        hopping11 = {}
        hopping_11 = {}
        hopping00_rescale = {} 
        hopping10_rescale = {}
        hopping01_rescale = {}
        hopping11_rescale = {}
        hopping_11_rescale = {}
        orbital_coords = np.array(self.coords)
        
        t_func = self.hops_func
        t_cutoff = self.hopping_parameters['max_dist']
    
        L_cutoff = 1.01 * max([t_cutoff, t_cutoff*self.a_top/self.a])

        vx, vy = self.latt_compatible
        vx = np.append(vx, [0.])
        vy = np.append(vy, [0.])
        
        Nbins = int(np.floor(vx[0]*np.sin(pi/3)/L_cutoff)) # uniformly cut lattice into Nbins pieces
        l_bin = vx[0]/Nbins # length of the small bin
        bin_xvec = [l_bin, 0., 0.]
        bin_yvec = [l_bin*np.cos(pi/3.), l_bin*np.sin(pi/3), 0.]
    
        # transform [i,j] to Cartesian bin location
        T = np.array([[bin_xvec[0], bin_yvec[0]],
                      [bin_xvec[1], bin_yvec[1]]])
        # transform Cart location to [i,j]
        T_inv = npla.inv(T)
        ##########################################################################################



        ########################### define bins and divide sites into bins ###########################
        def get_bindices(site):
            inds = np.dot(T_inv, np.array([site[0], site[1]]))
            i, j = int(np.floor(inds[0])), int(np.floor(inds[1]))
            if i == -1:
                i = 0 
            elif i == Nbins:
                i = Nbins-1
            if j == -1:
                j = 0
            elif j == Nbins:
                j = Nbins-1
            return i, j
            
        # devide coordinates into bins
        layer_inds = self._layer_inds()
        bins = {(i,j):[[] for _ in range(len(layer_inds))] for i in range(Nbins) for j in range(Nbins)}
        k = 0
        for site in orbital_coords:
            i,j = get_bindices(site)
            bins[(i,j)][self._siteind2layerind(k,layer_inds)].append(k)
            k = k + 1
        #################################################################################################


        ############################### add hopping in [0,0] unit cell ##################################
        tld_layers = [i for i in range(len(layer_inds)) if self.layer_types[i] in ['Atld','Btld']]
        # function for adding hoppings between bins [i0,j0] [i1,j1]
        def add_hoppings(i0, j0, i1, j1):
            for ly_ind0 in range(len(layer_inds)):
                tld0 = True if ly_ind0 in tld_layers else False
                for s0 in bins[(i0,j0)][ly_ind0]:
                    r0 = orbital_coords[s0]
                    for ly_ind1 in range(len(layer_inds)):
                        tld = tld0 and ly_ind0 == ly_ind1
                        for s1 in bins[(i1,j1)][ly_ind1]:
                            if s0 >= s1:
                                continue
                            r1 = orbital_coords[s1] 
                            r = r1 - r0
                            dr = npla.norm(r)
                            dz = abs(r[-1])
                            if dr < t_cutoff:
                                hop = t_func(dr, dz)
                                hopping00['%s_%s' % (s0, s1)] = hop
                            if tld:
                                dr_rescale = dr * self.a/self.a_top
                                if dr_rescale < t_cutoff:
                                    hop_rescale = t_func(dr_rescale, 0)
                                    hopping00_rescale['%s_%s' % (s0, s1)] = hop_rescale
                            else:
                                try:
                                    hopping00_rescale['%s_%s' % (s0, s1)] = hop
                                except:
                                    pass
                            try:
                                del hop
                            except:
                                pass
        # set hoppings to nearby bins
        check_bins = [-1, 0, 1]
        for i0 in range(Nbins):
            for j0 in range(Nbins):
                for di in check_bins:
                    for dj in check_bins:
                        try:
                            add_hoppings(i0, j0, i0+di, j0+dj)
                        except:
                            pass
        #############################################################################################


        #####################  add hopping between [0,0] and [0,1], [1,0], [1,1], [-1, 1]#############
        def add_hoppings_border(i0, j0, i1, j1, v_plus, hopping, hopping_rescale):
            """
            Args:
                hopping: hoppingxy, xy can be 01, 10, 11 or _11
                         a dictory saving hopping from unitcell 00 to xy
                (i0, j0): the bin index in unitcell xy 
                (i1, j1): the bin index in unitcell 00
                v_plus: the translational vector from unitcell 00 to xy 
            """
            for ly_ind0 in range(len(layer_inds)):
                tld0 = True if ly_ind0 in tld_layers else False
                for s0 in bins[(i0,j0)][ly_ind0]:
                    r0 = orbital_coords[s0] + v_plus
                    for ly_ind1 in range(len(layer_inds)):
                        tld = tld0 and ly_ind0 == ly_ind1
                        for s1 in bins[(i1,j1)][ly_ind1]:
                            r1 = orbital_coords[s1] 
                            r = r1 - r0
                            dr = npla.norm(r)
                            dz = abs(r[-1])
                            if dr < t_cutoff:
                                hop = t_func(dr, dz)
                                # hopping from s1 in unitcell 00 to s0 in unitcell xy
                                hopping['%s_%s' % (s1, s0)] = hop 
                            if tld:
                                dr_rescale = dr * self.a/self.a_top
                                if dr_rescale < t_cutoff:
                                    hop_rescale = t_func(dr_rescale, 0)
                                    hopping_rescale['%s_%s' % (s1, s0)] = hop_rescale
                            else:
                                try:
                                    hopping_rescale['%s_%s' % (s1, s0)] = hop
                                except:
                                    pass
                            try:
                                del hop
                            except:
                                pass
                        
        ### up down connection (0,0) and (0,1) cells
        for i in range(Nbins):
            add_hoppings_border(i,0,i, Nbins-1,vy, hopping01, hopping01_rescale)
            try:
                add_hoppings_border(i,0,i+1, Nbins-1,vy, hopping01, hopping01_rescale)
            except:
                pass
            try:
                add_hoppings_border(i,0,i-1, Nbins-1,vy, hopping01, hopping01_rescale)
            except:
                pass

        ### left right connection (0,0) and (1,0) cells
        for i in range(Nbins):
            add_hoppings_border(0,i,Nbins-1,i,vx, hopping10, hopping10_rescale)
            try:
                add_hoppings_border(0,i,Nbins-1,i+1,vx, hopping10, hopping10_rescale)
            except:
                pass
            try:
                add_hoppings_border(0,i,Nbins-1,i-1,vx, hopping10, hopping10_rescale)
            except:
                pass

        ### connect left-down and right-up cornors (0,0) and (1,1) cells
        add_hoppings_border(0,0,Nbins-1,Nbins-1,vx+vy, hopping11, hopping11_rescale)
        ### connect right-down and left-up cornors (0,0) and (-1,1) cells
        add_hoppings_border(Nbins-1, 0, 0, Nbins-1,vy-vx, hopping_11, hopping_11_rescale)
        ###############################################################################################

        hops = {'0_0':hopping00, '0_1':hopping01, '1_0':hopping10, '-1_1':hopping_11, '1_1':hopping11}
        hops_rescale = {'0_0':hopping00_rescale, '0_1':hopping01_rescale, '1_0':hopping10_rescale, \
                       '-1_1':hopping_11_rescale, '1_1':hopping11_rescale}
        return  hops, hops_rescale

    def change_hops_intld_to_pristine_G(self):
        del self.hopping
        self.hopping = self.hopping_rescale
        del self.hopping_rescale

    def save_to(self, fname='struct.json', memory_clear=False):
        out={}
        out['h'] = self.h
        out['a'] = self.a
        out['a_top'] = self.a_top
        out['coords'] = self.coords
        out['hopping'] = self.hopping
        out['latt_compatible']=self.latt_compatible
        out['hopping_parameters'] = self.hopping_parameters
        out['layer_nsites'] = self.layer_nsites
        out['layer_zcoords'] = self.layer_zcoords
        with open(fname, 'w') as f:
            f.write(json.dumps(jsanitize(out)))

    def _plot_stack(self, params=None):
        from matplotlib import pyplot as plt
        if params:
            plt.rcParams.update(parmas)
        z_min = min(self.layer_zcoords)
        z_max = max(self.layer_zcoords)
        scale = self.h
        for i in range(len(self.layer_zcoords)):
            z = self.layer_zcoords[i]/scale
            label = self.layer_types[i]
            if 'tld' in label:
                label = '$\widetilde{' + label[0]+'}$'
            else:
                label = '$'+label+'$'
            plt.plot([0, 0.5], [z, z], color='black', linewidth=2.0)
            plt.text(0.52, z, label, verticalalignment='center', horizontalalignment='left', fontsize=20)
        plt.xlim((0,1.0))
        plt.ylim((z_min/scale-2, z_max/scale+2))
        plt.axis('off')
        plt.savefig('stack.pdf')

    @staticmethod
    def _siteind2layerind(atom_ind, layer_inds):
        for i in range(len(layer_inds)):
            i0, i1 = layer_inds[i]
            if i0<=atom_ind<=i1:
                return i
        
    def _hopping_classify(self, hopping):
        layer_inds = self._layer_inds()
        hopping_classify = {}
        for uc_neigh in hopping:
            hopping_classify['uc:'+uc_neigh] = {}
            for pair in hopping[uc_neigh]:
                ind0, ind1 = np.array(pair.split('_'), dtype=int)
                layer0 = self._siteind2layerind(ind0, layer_inds)
                layer1 = self._siteind2layerind(ind1, layer_inds)
                try:
                    hopping_classify['uc:'+uc_neigh]['layer:%s_%s' % (layer0, layer1)][pair]=self.hopping[uc_neigh][pair]
                except:
                    hopping_classify['uc:'+uc_neigh]['layer:%s_%s' % (layer0, layer1)] = {}
                    hopping_classify['uc:'+uc_neigh]['layer:%s_%s' % (layer0, layer1)][pair]=self.hopping[uc_neigh][pair]
        return hopping_classify

    def _plot(self, ax=None, fig_name='PeriodicBiGrapheneQC.pdf',size=[2, 2], site_size=0.0, draw_dpi=600, lw=0.5):
        import matplotlib.collections as mc
        plot = False
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            plot = True

        layer_inds = self._layer_inds()
        cs = {'A':'black','B':'grey','Atld':'red','Btld':'orange'}

        coords=np.array(self.coords)

        layer_hops = [{} for _ in range(len(layer_inds))]
        for uc_pair in self.hopping:
            for pt_pair in self.hopping[uc_pair]:
                i,j = [int(i) for i in pt_pair.split('_')]
                for layer in range(len(layer_inds)):
                    ind0, ind1 = layer_inds[layer]
                    if ind0<=i<=ind1 and ind0<=j<=ind1:
                        try:
                            layer_hops[layer][uc_pair].append(pt_pair)
                        except:
                            layer_hops[layer][uc_pair]=[pt_pair]

        layer_lines = [[] for _ in range(len(layer_inds))]
        for layer in range(len(layer_inds)):
            ind0, ind1 = layer_inds[layer]
            for m in range(size[0]):
                for n in range(size[1]):
                    vec_add = m*self.latt_compatible[0] + n*self.latt_compatible[1]
                    ax.scatter(coords[:,0][ind0:ind1+1]+vec_add[0], coords[:,1][ind0:ind1+1]+vec_add[1], \
                               s=site_size, c=cs[self.layer_types[layer]], linewidths=0)
                    for uc_pair in layer_hops[layer]:
                        uc_i, uc_j = [int(i) for i in uc_pair.split('_')]
                        vec_add_to = vec_add + uc_i*self.latt_compatible[0] + uc_j*self.latt_compatible[1]
                        for pt_pair in layer_hops[layer][uc_pair]:
                            i,j = [int(i) for i in pt_pair.split('_')]
                            if ind0<=i<=ind1 and ind0<=j<=ind1:
                                try:
                                    layer_lines[layer].append([coords[i][:2]+vec_add,coords[j][:2]+vec_add_to])
                                except:
                                    pass
        for layer in np.array(self.layer_zcoords).argsort():
            layer_type = self.layer_types[layer]
            hops = layer_lines[layer]
            line = mc.LineCollection(hops, [0.1]*len(hops),colors=cs[layer_type], lw=lw)
            ax.add_collection(line)

        for i in range(size[0]+1):
            x = self.latt_compatible[0]*i
            y = self.latt_compatible[0]*i + self.latt_compatible[1]*size[1]
            ax.plot([x[0],y[0]],[x[1],y[1]], color='blue', ls='dashed',lw=0.5)

        for j in range(size[1]+1):
            x = self.latt_compatible[0]*size[0] + self.latt_compatible[1]*j
            y = self.latt_compatible[1]*j
            ax.plot([x[0],y[0]],[x[1],y[1]], color='blue', ls='dashed',lw=0.5)
                
            
        if plot:
            plt.axis('equal')
            plt.draw()
            plt.axis('off')
            plt.savefig(fig_name, bbox_inches='tight', draw_dpi=draw_dpi, pad_inches=0)
            plt.close()
    

class StructureRead(Structure):
    def __init__(self, **d):
        self.a = d['a']
        self.a_top = d['a_top']
        self.h = d['h']
        self.coords = d['coords']
        self.layer_nsites = d['layer_nsites']
        self.layer_zcoords = d['layer_zcoords']
        self.hopping_parameters = d['hopping_parameters']
        self.hopping = d['hopping']
        self.latt_compatible = d['latt_compatible']

    @staticmethod
    def from_file(fname):
        d = json.load(open(fname))
        return StructureRead(**d)


class BZs(object):
    def __init__(self, struct):
        self.nsite_bott = struct.layer_nsites_no_disorder[0]
        self.nsite_top = struct.layer_nsites_no_disorder[1]
        self.n_PC_top = int(self.nsite_top/2)
        self.n_PC_bott = int(self.nsite_bott/2)
        self.latt_comp = struct.latt_compatible
        self._get_reciprocal_latt_each_layer()

    def _get_reciprocal_latt_each_layer(self):
        def recip_latt(latt):
            return np.linalg.inv(latt).transpose()*2*np.pi
        l_comp = np.linalg.norm(self.latt_comp[0])
        a_top = l_comp/np.sqrt(self.n_PC_top)
        self.latt_top = np.array([[a_top, 0.],[0.5*a_top, np.sin(np.pi/3)*a_top]]) 

        b_bott = 1./3* l_comp/np.sqrt(int(self.nsite_bott/6))
        a_bott = np.sqrt(3)*b_bott
        self.latt_bott = np.array([rotate_on_vec(30., np.array([a_bott, 0.])), \
                                   rotate_on_vec(30., np.array([0.5*a_bott, np.sin(np.pi/3)*a_bott])) ])
        self.recip_latt_bott = recip_latt(self.latt_bott)
        self.recip_latt_top = recip_latt(self.latt_top)
        self.recip_latt_comp = recip_latt(self.latt_comp)
    
    def _symmetried_kpts(self, layer):
        if layer == 'bottom':
            recip_latt = self.recip_latt_bott
        elif layer == 'top':
            recip_latt = self.recip_latt_top
        elif layer == 'hetrostruct':
            recip_latt = self.recip_latt_comp
        Ms = [[0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [-0.5, -0.5], [-0.5, 0.0], [0.0, -0.5]]
        Ks = [[1/3, 2/3], [2/3, 1/3], [-1/3, 1/3], [-1/3, -2/3], [1/3, -1/3], [-2/3, -1/3]]
        return {'Ms':np.array([frac2cart(i, recip_latt) for i in Ms]), 
                'Ks':np.array([frac2cart(i, recip_latt) for i in Ks])}

    def _spectial_kpoints(self):
        K_bot = self._symmetried_kpts('bottom')['Ks'][0]
        K0 = self._symmetried_kpts('bottom')['Ks'][2]
        M_top = self._symmetried_kpts('top')['Ms'][1]
        K_top = self._symmetried_kpts('top')['Ks'][1]
        K1 = self._symmetried_kpts('top')['Ks'][0]
        M_bot = self._symmetried_kpts('bottom')['Ms'][0]
        D_top = K1 + self.recip_latt_top[0] - self.recip_latt_bott[0]
        D_bot = K0 + self.recip_latt_bott[0] + self.recip_latt_bott[1] - self.recip_latt_top[1]
        KR_bot = 2*M_top - K_bot
        KR_top = 2*M_bot - K_top
        G = np.array([0.0, 0.0])
        M_bot_orig = self._symmetried_kpts('bottom')['Ms'][2]
        M_top_end = self._symmetried_kpts('top')['Ms'][2]
        return {'K_top':K_top, 'K_bot':K_bot, 'KR_bot':KR_bot, 'KR_top':KR_top, \
                'M_bot':M_bot, 'M_top':M_top, 'G':G, 'D_top':D_top, 'D_bot':D_bot, \
                'M_bot_orig':M_bot_orig, 'M_top_end':M_top_end}

    def _plot_BZs(self, BZ_comp=False, path=True, text=True, dots=True, D_show=False,\
                  a_quarter=False,recip_latt_vec=False, params=None):
        """
        Args: 
            BZ_comm: 
            path: the path for effective band structure
            text: text label for some special kpoints
            dots: dots for all special kpoints
            a_quarter: only a quarter part of BZ (top right) is plot
            recip_latt_vec: the recipral lattice vectors
        """
        from matplotlib import pyplot as plt
        if params:
            plt.rcParams.update(params)
        else:
            try:
                import tBG
                plt.rcParams.update(tBG.params)
            except:
                pass
        fig, (ax) = plt.subplots(1,1)
        K0 = self._symmetried_kpts('bottom')['Ks']
        M0 = self._symmetried_kpts('bottom')['Ms']
        K1 = self._symmetried_kpts('top')['Ks']
        M1 = self._symmetried_kpts('top')['Ms']
        K2 = self._symmetried_kpts('hetrostruct')['Ks']
        
        if dots:

            # K points
            for i in range(len(K0)):
                ax.scatter(K0[i][0], K0[i][1], c='black', clip_on=False)
                #ax.text(K0[i][0], K0[i][1],str(i), color='red')
            for i in range(len(K1)):
                ax.scatter(K1[i][0], K1[i][1], c='red', clip_on=False)
                #ax.text(K1[i][0], K1[i][1], str(i), color='blue')

            # M points
            for i in M0:
                ax.scatter(i[0], i[1], c='black', marker='s')
            for i in M1:
                ax.scatter(i[0], i[1], c='red', marker='s')

            # KR points (K1)
            K0_M1 = {0:1, 1:2, 2:4, 3:5, 4:0, 5:3}
            for i in K0_M1:
                K = K0[i]
                M = M1[K0_M1[i]]
                KR = 2*M -K
                ax.scatter(KR[0], KR[1], color='red', marker='o')
            K1_M0 = {0:2, 1:0, 2:1, 3:3, 4:5, 5:4}
            for i in K1_M0:
                K = K1[i]
                M = M0[K1_M0[i]]
                KR = 2*M -K
                ax.scatter(KR[0], KR[1], color='black', marker='o')

            if D_show:
                D_0 = K1[0] + self.recip_latt_top[0] - self.recip_latt_bott[0]
                D_1 = K1[1] - self.recip_latt_top[1] + self.recip_latt_bott[1]
                D_2 = K1[4] - (self.recip_latt_top[0]+self.recip_latt_top[1]) + self.recip_latt_bott[0] + self.recip_latt_bott[1]
                D_3 = K1[3] - self.recip_latt_top[0] + self.recip_latt_bott[0]
                D_4 = K1[5] + self.recip_latt_top[1] - self.recip_latt_bott[1]
                D_5 = K1[2] + self.recip_latt_top[0] + self.recip_latt_top[1] - self.recip_latt_bott[0] - self.recip_latt_bott[1]
                for i in [D_0, D_1, D_2, D_3, D_4, D_5]:
                    ax.scatter(i[0], i[1], color='black', marker='o')

                D0 = K0[2] + self.recip_latt_bott[0] + self.recip_latt_bott[1] - self.recip_latt_top[1]
                D1 = K0[0] + self.recip_latt_bott[0] - self.recip_latt_top[0] - self.recip_latt_top[1]
                D2 = K0[1] - self.recip_latt_bott[1] - self.recip_latt_top[0]
                D3 = K0[4] - self.recip_latt_bott[0] - self.recip_latt_bott[1] + self.recip_latt_top[1]
                D4 = K0[3] - self.recip_latt_bott[0] + self.recip_latt_top[0] + self.recip_latt_top[1]
                D5 = K0[5] + self.recip_latt_bott[1] + self.recip_latt_top[0]
                for i in [D0, D1, D2, D3, D4, D5]:
                    ax.scatter(i[0], i[1], color='red', marker='o')

            # Gamma point
            ax.scatter(0,0, c='black')

        # plot BZs
        for i,j in [[0,1],[0,2],[2,5],[5,3],[3,4],[4,1]]:
            ax.plot([K0[i][0],K0[j][0]],[K0[i][1],K0[j][1]],c='black')
            ax.plot([K1[i][0],K1[j][0]],[K1[i][1],K1[j][1]],c='red')
            if BZ_comp:
                ax.plot([K2[i][0],K2[j][0]],[K2[i][1],K2[j][1]],c='green', linestyle='dashed')


        # plot path for band
        if path:
            ax.plot(K0[0], [0, 0], linestyle='dashed', color='blue', lw=2)
            ax.arrow(K0[0][0]/2, K0[0][1]/2, -K0[0][0]/20, -K0[0][1]/20, color='blue', lw=0, head_width=0.1,clip_on=False)
            ax.plot([0,0], K1[1], linestyle='dashed', color='blue', lw=2)
            ax.arrow(K1[1][0]/2, K1[1][1]/2, K1[1][0]/20, K1[1][1]/20,  color='blue', lw=0, head_width=0.1, clip_on=False)

        # reciprocal lattice vectors 
        if recip_latt_vec:
            ax.arrow(0, 0, self.recip_latt_bott[0][0], self.recip_latt_bott[0][1],  color='red', lw=0, head_width=0.1)
            ax.arrow(0, 0, self.recip_latt_bott[1][0], self.recip_latt_bott[1][1],  color='red', lw=0, head_width=0.1)
            ax.arrow(0, 0, self.recip_latt_top[0][0], self.recip_latt_top[0][1],  color='blue', lw=0, head_width=0.1)
            ax.arrow(0, 0, self.recip_latt_top[1][0], self.recip_latt_top[1][1],  color='blue', lw=0, head_width=0.1)
        
        if text:
            ax.text(K0[0][0],K0[0][1]+0.05,'$K$', color='black', horizontalalignment='center')
            ax.text(M0[0][0]+0.01,M0[0][1]-0.15,'$M$', color='black')
            ax.text(K1[1][0]+0.04,K1[1][1],'$\widetilde{K}$', color='red', verticalalignment='center')
            ax.text(M1[1][0]+0.03,M1[1][1]+0.03,'$\widetilde{M}$', color='red', )
            ax.text((2*M1[1]-K0[0])[0]+0.03,(2*M1[1]-K0[0])[1],'$\widetilde{K}_1$', color='red', verticalalignment='center')
            ax.text((2*M0[0]-K1[1])[0],(2*M0[0]-K1[1])[1]+0.05,'$K_1$', color='black', \
                    horizontalalignment='center')
            ax.text(0,0,'$\Gamma$', color='black')
            if D_show:
                ax.text(D_0[0], D_0[1], '$K_2$', color='black')
                ax.text(D0[0], D0[1], '$\widetilde{K}_2$', color='red')

        # xlim and ylim
        left = min([min(K1[:,0]), min(K0[:,0])])
        bottom = min([min(K1[:,1]), min(K0[:,1])])
        if a_quarter:
            left = 0
            bottom = 0
        right = max([max(K1[:,0]), max(K0[:,0])])
        top = max([max(K1[:,1]), max(K0[:,1])])
        ax.axis('off')
        ax.axis('equal')
        ax.set_adjustable('box-forced') 
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.tick_params(axis="y",direction="in", pad=-22, labelsize=0.)
        ax.tick_params(axis="x",direction="in", pad=-15, labelsize=0.)
        plt.savefig('BZs.pdf',bbox_inches='tight',pad_inches=0.0)
        plt.clf()

