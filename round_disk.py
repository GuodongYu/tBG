import numpy as np
import numpy.linalg as npla
import json
import time
import sys
import tipsi
import copy
from monty.json import jsanitize
import multiprocessing as mp
from multiprocessing import Pool
import itertools
import os
from tBG.hopping import divide_sites_2D, calc_hoppings, hop_list_graphene_wannier
from tBG.utils import *
from tBG.hopping import SparseHopDict

pi = np.pi

"""
TO DO: merge this file into quantum_dot.py file
"""
class _Methods:
    def pymatgen_struct(self):
        from pymatgen.core.structure import Structure as pmg_struct
        coords = copy.deepcopy(self.coords)
        nsite  = len(coords)
        x_min, y_min, z_min = np.min(coords, axis=0)
        x_max, y_max, z_max = np.max(coords, axis=0)
        coords[:,0] = coords[:,0] - x_min + 5 
        coords[:,1] = coords[:,1] - y_min + 5 
        coords[:,2] = coords[:,2] - z_min + 5 
        latt_vec = np.array([[x_max-x_min+20, 0, 0],[0, y_max-y_min+20, 0],[0, 0, z_max-z_min+20]])
        return  pmg_struct(latt_vec, ['C']*int(nsite/2)+['Fe']*int(nsite/2), coords, coords_are_cartesian=True)

class Structure(_Methods):
    def read_relaxed_struct_from_file(self, filename):
        """
        read the xyz file for site coords. 
        don't forget to add hopping manually after reading.
        """
        with open(filename, 'r') as f:
            nl = int(f.readline())
        data = read_last_n_lines(filename, n=nl)
        data = np.array([[j for j in i.split()] for i in data])
        nsite_bot = np.count_nonzero(data[:,0]=='1')
        nsite_top = np.count_nonzero(data[:,0]=='2')
        self.coords = np.array(data[:,1:], dtype=float)
        self.layer_nsites = [nsite_bot, nsite_top]
        self.h = 3.461
        self.a = 2.456

    def read_struct_and_hopping_from_file(self, filename):
        d = np.load(filename)
        self.coords = d['coords']
        self.hopping = (d['hop_keys'], d['hop_vals'])

    def output_struct_to_xyz_file(self):
        atom_type = np.concatenate(tuple((np.repeat([i+1],self.layer_nsites[i]) for i in range(len(self.layer_nsites)))))
        coord_str = np.append(np.array([atom_type], dtype=str), self.coords.T, axis=0).T
        #coord_str = np.array(coord, ntype=str)
        coord_str = '\n'.join([' '.join(i) for i in coord_str])
        with open('struct_relaxed.xyz', 'w') as f:
            f.write('%s\n' % len(self.coords))
            f.write('Relaxed structure\n')
            f.write(coord_str)
            
    def make_structure(self, R, rotation_angle=30., a=2.456, h=3.461, overlap='hole', rm_dangling=True):
        """
        Class for constructing bilayer graphene and collecting all neighbors of a given site by distance.

        Details:
            The bottom layer is placed on the xy (z=0) plane, and its a1 lattice vector along x axis, and its a2 lattice
            vector has the anticlockwise 60 degree with respective to a1. The top layer is placed on the z=h plane, and 
            it rotates theta (30 here for quasicrystal) degree anticlockwisely with respective to the bottom layer.
        Args: 
            R: the radius of the bilayer graphene disk (in units of a the lattice constant of graphene), no default must give
            h: the interlayer distance, default 3.461 for quasicrystal. 3.349 for general
            rotation_angle: the rotation angle of the top layer relative to bottom layer
                            30. for the quasicrystal phase, default 30.0
            a: the lattice constant of graphene layer, default 2.46
            overlap: 'hole' or 'atom', choose the rotation center starting from an AA bilayer
            rm_dangling: whether to remove all the edge atom with only one nearest neighbors
        
        comments:
            AB stacking: rotation_angle=60 & overlap='atom'

        Units:
            distance: Angstrom
            rotation_angle: degree
        """
        self.R = R
        self.a = a
        self.h = h
        self.rotation_angle = rotation_angle
        self.rm_dangling = rm_dangling
        b = a/np.sqrt(3.)
        va_bottom = np.array([a*np.cos(30.*pi/180.), -a*np.sin(30.*pi/180.)])
        vb_bottom = np.array([a*np.cos(30.*pi/180.), a*np.sin(30.*pi/180.)])

        va_top = rotate_on_vec(rotation_angle, va_bottom)
        vb_top = rotate_on_vec(rotation_angle, vb_bottom)

        self.latt_bottom = np.array([va_bottom, vb_bottom])
        self.latt_top = np.array([va_top, vb_top])
        
        if overlap == 'hole':
            self.site0 = np.array([1./3., 1./3.])
            self.site1 = np.array([2./3., 2./3.])
        elif overlap == 'atom':
            self.site0 = np.array([0., 0.])
            self.site1 = np.array([1./3., 1./3.])
        elif overlap == 'side_center':
            self.site0 = np.array([-1/6., -1/6.])
            self.site1 = np.array([1./6., 1./6.])
            

        self.radius = R*a
        self.NN = {'1to0':tuple([[0,0],[1,0],[0,1]]), '0to1':tuple([[0,0],[-1,0],[0,-1]])}
        ## bottom vecs to three NN
        v0_to_NN = (va_bottom + vb_bottom)/3
        v1_to_NN = rotate_on_vec(120, v0_to_NN)
        v2_to_NN = rotate_on_vec(240, v0_to_NN)
        vs_to_NN_bottom = np.array([v0_to_NN, v1_to_NN, v2_to_NN])
        vs_to_NN_bottom = np.append(vs_to_NN_bottom, [[0],[0],[0]],axis=1)
        ## top vecs_to three NN
        v0_to_NN = (va_top + vb_top)/3
        v1_to_NN = rotate_on_vec(120, v0_to_NN)
        v2_to_NN = rotate_on_vec(240, v0_to_NN)
        vs_to_NN_top = np.array([v0_to_NN, v1_to_NN, v2_to_NN])
        vs_to_NN_top = np.append(vs_to_NN_top, [[0],[0],[0]],axis=1)
        self.vecs_to_NN = {'A':[vs_to_NN_bottom, -vs_to_NN_bottom],'Atld':[vs_to_NN_top, -vs_to_NN_top]}
        self.vecs_to_NN['B'] = self.vecs_to_NN['A']
        self.vecs_to_NN['Btld'] = self.vecs_to_NN['Atld']
        self._make_structure()


    def _get_cells(self):
        s0 = self.site0
        s1 = self.site1
        def get_i_range(site):
            """
            cut a unit-cell-shaped rhomboid to cover the circle totally
            """
            if site=='site0':
                r = np.sqrt(3.)/3.+ self.R
            elif site=='site1':
                r = 2.*np.sqrt(3.)/3. + self.R
            i = np.ceil(2./np.sqrt(3)*r)
            return range(-int(i), int(i)+1)

        def get_norm_frac_array(frac_array):
            """
            get the norm of an array with frac-coord
            """
            return np.array([np.sqrt(i[0]**2+i[1]**2+i[0]*i[1]) for i in frac_array])

        def get_cells_without_danglings(site, cells_edge):
            """
            the sites at the edge with only one nearest neighbor will be removed
            """
            if site == 'site0':
                NNs = self.NN['0to1']
                s = s1
            elif site== 'site1':
                NNs = self.NN['1to0']   
                s = s0
            NNs = np.array([[[cell[0]+i[0]+s[0], cell[1]+i[1]+s[1]] for i in NNs] for cell in cells_edge])
            norms = np.array([len(np.where(get_norm_frac_array(frac_arrys)>self.R)[0]) for frac_arrys in NNs])
            ids = np.where(norms>1)[0]
            return np.delete(cells_edge, ids, axis=0)

        def cut_circle(site):
            """
            cut the circle region from the rhomboid obtained by function get_i_range
            """
            i_range = get_i_range(site)
            if site=='site0':
                s = s0
            elif site == 'site1':
                s = s1
            cells = np.array([(i,j) for i in i_range for j in i_range])
            norms = get_norm_frac_array(np.array(cells)+s)
            ids = np.where(norms<=self.R)[0]
            cells_in_R = cells[ids]
            norms_in_R = get_norm_frac_array(cells_in_R+s)
            ids_edge = np.where(norms_in_R>self.R-1)[0]
            cells_edge = cells_in_R[ids_edge]
            cells_bulk = np.delete(cells_in_R, ids_edge, axis=0)
            return cells_bulk, cells_edge

        cells_s0_bulk, cells_s0_edge = cut_circle('site0')
        cells_s1_bulk, cells_s1_edge = cut_circle('site1')
        if self.rm_dangling:
            cells_s0_edge = get_cells_without_danglings('site0', cells_s0_edge)
            cells_s1_edge = get_cells_without_danglings('site1', cells_s1_edge)
        return cells_s0_bulk, cells_s0_edge, cells_s1_bulk, cells_s1_edge

    def _make_structure(self):
        site0 = self.site0
        site1 = self.site1

        cells_s0_bulk, cells_s0_edge, cells_s1_bulk, cells_s1_edge = self._get_cells()
        
        def get_cart_coords(site, cells):
            if site == 'site0':
                site = site0
            elif site == 'site1':
                site = site1
            ncell = len(cells)
            cart_bs = np.append([frac2cart(site+cell, self.latt_bottom) for cell in cells], np.array([[0.]]*ncell), axis=1)
            cart_ts = np.append([frac2cart(site+cell, self.latt_top) for cell in cells], np.array([[self.h]]*ncell), axis=1)
            return cart_bs, cart_ts

        coords_b1_bulk, coords_t1_bulk = get_cart_coords('site1', cells_s1_bulk)
        coords_b0_bulk, coords_t0_bulk = get_cart_coords('site0', cells_s0_bulk)
        coords_b1_edge, coords_t1_edge = get_cart_coords('site1', cells_s1_edge)
        coords_b0_edge, coords_t0_edge = get_cart_coords('site0', cells_s0_edge)
            
        self.coords = np.concatenate((coords_b0_bulk, coords_b0_edge, coords_b1_bulk, coords_b1_edge,\
                                 coords_t0_bulk, coords_t0_edge, coords_t1_bulk, coords_t1_edge), axis=0)
        def _get_sites(cells, sublat):
            return np.append(cells, np.array([sublat]*len(cells)).reshape(-1,1), axis=1)
        _sites_one_layer = np.concatenate([_get_sites(cells_s0_bulk, 0), _get_sites(cells_s0_edge,0),\
                                 _get_sites( cells_s1_bulk, 1),_get_sites(cells_s1_edge,1)])
        self._sites_one_layer = [tuple(i) for i in _sites_one_layer]

        nsites_bottom_0 = len(coords_b0_bulk) + len(coords_b0_edge)
        nsites_bottom_1 = len(coords_b1_bulk) + len(coords_b1_edge)
        nsites_top_0 = len(coords_t0_bulk) + len(coords_t0_edge)
        nsites_top_1 = len(coords_t1_bulk) + len(coords_t1_edge)
        nsites_bottom = nsites_bottom_0 + nsites_bottom_1
        nsites_top = nsites_top_0 + nsites_top_1
        self.layer_nsites = [nsites_bottom, nsites_top]
        self.layer_nsites_sublatt = [[nsites_bottom_0, nsites_bottom_1],[nsites_top_0, nsites_top_1]]
        self.layer_zcoords = [coords_b0_bulk[0][-1], coords_t0_bulk[0][-1]]
        self.layer_types = ['A', 'Atld']

        nsites_group = np.array([len(cells_s0_bulk), len(cells_s0_edge), len(cells_s1_bulk), len(cells_s0_edge),\
                  len(cells_s0_bulk), len(cells_s0_edge), len(cells_s1_bulk), len(cells_s1_edge)])
        self.site_ids_edge = {'bottom_site0':[np.sum(nsites_group[:1]), np.sum(nsites_group[:2])-1], \
                              'bottom_site1':[np.sum(nsites_group[:3]), np.sum(nsites_group[:4])-1],\
                              'top_site0':[np.sum(nsites_group[:5]), np.sum(nsites_group[:6])-1], \
                              'top_site1':[np.sum(nsites_group[:7]), np.sum(nsites_group[:8])-1]}

    def remove_top_layer(self):
        """
        after removing the top layer, it changes to be graphene
        """
        ids = self._layer_inds()[0]
        self.coords = self.coords[0:ids[1]+1]
        self.layer_nsites = [self.layer_nsites[0]]
        self.layer_zcoords = [self.layer_zcoords[0]]
        self.layer_types = [self.layer_types[0]]

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

    def _coords_xy(self, layer):
        """
        layer: 'A', 'B', 'Atld', 'Btld'
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
            xy_1[:,0] = xy_1[:,0] + self.a/np.sqrt(3)*np.cos(np.pi/6)
            xy_1[:,1] = xy_1[:,1] + self.a/np.sqrt(3)*np.sin(np.pi/6)
            return xy_1

    def get_append_dict(self, stack):
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

    def append_layers(self, layers):
        """
        A (bottom) and Atld (top) layers already exist at 0 and 1*

        layers: a dict info for the appended layers {layer: positions list ...}
                B|A bilayer has AB stacking 
                A|Atld bilayers has 30 degree twist angle
        such as {'A': [-1, -3], 'B':[-2, -4], 'Atld':[2, 4], 'Btld':[3,5]}
            
             at -1*h and -3*h  add A layers 
                -2*h and -4*h  add B layers
                2*h and 4*h add Atld layers
                3*h and 5*h add Btld layers
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

    def add_hopping_pz(self, split=False, max_dist=5.0, g0=3.12, a0=1.42, g1=0.48, h0=3.349, \
                                     rc=6.14, lc=0.265, q_dist_scale=2.218, nr_processes=1):
        from tBG.hopping import hop_func_pz 
        hop_func = hop_func_pz(g0=g0, a0=a0, g1=g1, h0=h0, rc=rc, lc=lc, q_dist_scale=q_dist_scale)

        if split:
            nlayer = len(self.layer_nsites)
            layid_sorted = np.argsort(self.layer_zcoords)
            layer_inds = self._layer_inds()

            def collect_layer_data(lay_id):
                id_range = layer_inds[lay_id]
                sites = self.coords[id_range[0]:id_range[1]+1]
                bins = divide_sites_2D(sites, bin_box=[[max_dist,0],[0,max_dist]], idx_from=id_range[0])
                return sites, bins

            ## intralayer hopping
            for i in range(nlayer):
                lay_id = layid_sorted[i]
                sites, bins = collect_layer_data(lay_id)
                key, value = calc_hoppings(sites, bins, hop_func=hop_func, max_dist=max_dist, nr_processes=nr_processes)
                try:
                    keys = np.concatenate([keys, key], axis=0)
                    values = np.concatenate([values, value], axis=0)
                except:
                    keys = key
                    values = value

            ## interlayer hopping
            for i in range(nlayer-1):
                lay0_id = layid_sorted[i]
                site0s, bin0s = collect_layer_data(lay0_id)

                lay1_id = layid_sorted[i+1]
                site1s, bin1s = collect_layer_data(lay1_id)

                key, value = calc_hoppings(site0s, bin0s, site1s, bin1s, hop_func=hop_func, max_dist=max_dist, nr_processes=nr_processes)

                keys = np.concatenate([keys, key], axis=0)
                values = np.concatenate([values, value], axis=0)
        else:
            sites = self.coords
            bins = divide_sites_2D(sites, bin_box=[[max_dist,0],[0,max_dist]], idx_from=0)
            keys, values = calc_hoppings(sites, bins, hop_func=hop_func, max_dist=max_dist, nr_processes=nr_processes)
        self.hopping = keys, values

    def add_hopping_wannier(self, max_dist=6.0, P=0, \
                            ts=[-2.8922, 0.2425, -0.2656, 0.0235, \
                              0.0524,  -0.0209, -0.0148, -0.0211], nr_processes=1):
        """
        max_dist: the interlayer hopping will be add if C-C distance less than max_dist
        lambda0,3,6, xi0,3,6 k0,6 and x3,6 are the params for interlayer hopping 
        ts: the hopping energies for the first 8-nearest neighbors for intralayer 
        """
        from tBG.hopping import hop_params_wannier_interlayer, hop_func_wannier_interlayer
        lambda0,xi0,k0,lambda3,xi3,x3,lambda6,xi6,x6,k6 = \
               hop_params_wannier_interlayer(P)

        nlayer = len(self.layer_nsites)
        layid_sorted = np.argsort(self.layer_zcoords)
        layer_inds = self._layer_inds()
        layer_inds_sublatt = self._layer_inds_sublatt()


        ## intra-layer hopping
        hop_list = hop_list_graphene_wannier(ts)
        hop_list_keys = [list(i.keys()) for i in hop_list]

        nsite_sublatt = self.layer_nsites_sublatt[0][0]
        nsite_per_layer = self.layer_nsites[0]
        cells = self._sites_one_layer
        links = dict(zip(self._sites_one_layer, range(len(cells))))


        def get_key(i,j):
            try:
                return [i, links[(cells[i][0]+j[0], cells[i][1]+j[1], j[2])]]
            except:
                return [None, None]

        def get_sub_hops(sublatt):
            id0 = sublatt*nsite_sublatt
            id1 = (sublatt+1)*nsite_sublatt
            keys = np.array([get_key(i,j) for i in range(id0,id1) for j in hop_list_keys[sublatt]])
            values = np.array([hop_list[sublatt][j] for i in range(id0,id1) for j in hop_list_keys[sublatt]])
            inds = np.where(keys)[0]
            return np.array(list(keys[inds]),dtype=int), values[inds]

        keys0, values0 = get_sub_hops(0)
        keys1, values1 = get_sub_hops(1)
        try:
            keys_one_layer = np.concatenate([keys0,keys1])
            values_one_layer = np.concatenate([values0, values1])
        except:
            # for the case of only nearest neighbor considered, keys1 and values1 are empty
            keys_one_layer = keys0
            values_one_layer = values0
        for i in range(nlayer):
            keys_i = keys_one_layer + [nsite_per_layer*i, nsite_per_layer*i]
            try:
                keys = np.concatenate([keys, keys_i])
                values = np.concatenate([values, values_one_layer])
            except:
                keys = keys_one_layer
                values = values_one_layer

        ## inter-layer hopping (only between nearest layers)
        def collect_sublatt_data(lay_id, sublatt):
            id_range = layer_inds_sublatt[lay_id][sublatt]
            sites = self.coords[id_range[0]:id_range[1]+1]
            bins = divide_sites_2D(sites, bin_box=[[max_dist,0],[0,max_dist]], idx_from=id_range[0])
            vec_to_NN = self.vecs_to_NN[self.layer_types[lay_id]][sublatt][0]
            return sites, bins, vec_to_NN


        for i in range(nlayer-1):
            lay0_id = layid_sorted[i]
            lay1_id = layid_sorted[i+1]
            for sub0 in [0,1]:
                site0s, bin0s, vec0_to_NN = collect_sublatt_data(lay0_id, sub0)
                site0s[:,-1] = 0.
                for sub1 in [0,1]:
                    site1s, bin1s, vec1_to_NN = collect_sublatt_data(lay1_id, sub1)
                    site1s[:,-1] = 0
                    hop_func = hop_func_wannier_interlayer(vec0_to_NN, vec1_to_NN, lambda0, xi0, k0, \
                                                           lambda3, xi3, x3, lambda6, xi6, x6, k6, self.a)
                    key, value = calc_hoppings(site0s, bin0s, site1s, bin1s, \
                         hop_func=hop_func, max_dist=max_dist, nr_processes=nr_processes)
                    keys = np.concatenate([keys, key], axis=0)
                    values = np.concatenate([values, value], axis=0)
        self.hopping = keys, values 


    def edge_site_ids_by_distance(self, dist_to_edge=5.):
        """
        get the site ids for the sites outside the circle self.R - R
        """
        coords = self.coords[:,:-1]
        ids = np.where(np.linalg.norm(coords, axis=1)>self.radius-dist_to_edge)[0]    
        return ids

    def output_Lammps_struct_file(self, atom_style='full'):
        """
        atom_style: 'full' or 'atomic' which determine the format of the
                     atom part in data file
        atoms in different layers were given differen atom_type and molecule-tag
        """
        from tools_lammps import write_lammps_datafile
        n_atom = np.sum(self.layer_nsites)
        n_atom_type = len(self.layer_zcoords)
        mass = [12.0107]*n_atom_type

        a = [min(self.coords[:,0])-100, max(self.coords[:,0])+100]
        b = [min(self.coords[:,1])-100, max(self.coords[:,1])+100]
        c = [min(self.coords[:,2])-100, max(self.coords[:,2])+100]
        box = [a, b, c]
        tilts = [0, 0, 0]

        atom_id = range(1, n_atom+1)
        atom_type = np.concatenate([[i]*self.layer_nsites[i-1] for i in range(1, n_atom_type+1)])
        molecule_tag = atom_type
        q = [0.0] * n_atom
        write_lammps_datafile(box, atom_id, self.coords, n_atom, n_atom_type, mass, atom_type, atom_style=atom_style, \
                              tilts=tilts, qs=q, molecule_tag=molecule_tag)

    def save_to(self, fname='struct'):
        out={}
        hop_keys, hop_vals = self.hopping
        out['hopping_parameters'] = self.hopping_parameters
        np.savez_compressed(fname, coords=self.coords, hop_keys=hop_keys, hop_vals=hop_vals, info=[out])
    
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

    def _plot_stack(self, fname='stack'):
        from matplotlib import pyplot as plt
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
        plt.savefig(fname+'.pdf')
        plt.clf()

    def plot(self, fname='BiGraphene.pdf',site_size=3.0, draw_dpi=600, lw=0.6, edge_cut=None):
        import matplotlib.pyplot as plt
        import matplotlib.collections as mc
        fig, ax = plt.subplots()
        nsites = len(self.coords)
        cs = {'A':'black', 'Atld':'red', 'B':'grey', 'Btld':'orange'}
        layer_inds = self._layer_inds()
        layer_hops = [[] for _ in range(len(self.layer_nsites))]
        for pair, hop in zip(self.hopping[0], self.hopping[1]):
            i,j = pair
            for k in range(len(layer_inds)):
                ind0,ind1 = layer_inds[k]
                if ind0<=i<=ind1 and ind0<=j<=ind1:
                    layer_hops[k].append([self.coords[i][:2],self.coords[j][:2]])
        for i in np.array(self.layer_zcoords).argsort():
            layer_type = self.layer_types[i]
            ind0,ind1 = layer_inds[i]
            line = mc.LineCollection(layer_hops[i], [0.1]*len(layer_hops[i]),colors=cs[layer_type], lw=lw)
            ax.add_collection(line)
            plt.draw()
            plt.scatter(self.coords[:,0][ind0:ind1+1], self.coords[:,1][ind0:ind1+1], \
                        s=site_size, color=cs[layer_type],linewidth=0)
        if edge_cut is None:
            for i in self.site_ids_edge:
                plt.scatter(self.coords[:,0][self.site_ids_edge[i][0]:self.site_ids_edge[i][1]+1],\
                            self.coords[:,1][self.site_ids_edge[i][0]:self.site_ids_edge[i][1]+1],\
                            s = site_size+50, color='purple', marker='*', linewidth=0)
        else:
            edge_site_ids = self.edge_site_ids_by_distance(edge_cut)
            plt.scatter(self.coords[:,0][edge_site_ids], self.coords[:,1][edge_site_ids],\
                            s = site_size+50, color='purple', marker='*', linewidth=0)
                
                    
            
        ax.set_aspect('equal')
        plt.axis('equal')
        plt.axis('off')
        #plt.draw()
        plt.savefig(fname, bbox_inches='tight', draw_dpi=draw_dpi)
        plt.close()

################# below for tipsi sample ###########
def siteset(nsite):
    siteset = tipsi.SiteSet()
    for k in range(nsite):
        siteset.add_site((0, 0, 0), k)
    return siteset

def lattice(coords):
    sites = np.array(coords)*0.1
    x_min, y_min, z_min = np.min(sites, axis=0) 
    x_max, y_max, z_max = np.max(sites, axis=0)
    lat_vecs = [[x_max-x_min, 0., 0.],[0., y_max-y_min, 0.]] 
    lattice = tipsi.Lattice(lat_vecs, sites)
    return lattice


def hopdict(struct, elec_field=0.0, **kws):
    hop_keys, hop_vals = struct.hopping
    z_coords = struct.coords[:,2]
    n_pairs = len(hop_keys)
    nsite = len(z_coords)
    hop_dict_list = [{} for i in range(nsite)]
    for ind in range(n_pairs):
        i,j = hop_keys[ind]
        hop = hop_vals[ind]
        hop_dict_list[i][(0,0,0) + (j,)] = hop
        hop_dict_list[j][(0,0,0) + (i,)] = hop
    if elec_field:
        if z_coords not in kws:
            raise ValueError
        E_on = kws['z_coords']*elec_field
        for i in range(nsite):
            hop_dict.set_element((0,0,0), (i,i), E_on[i])
    hop_dict = SparseHopDict(nsite)
    hop_dict.dict = hop_dict_list
    return hop_dict

def sample(struct, rescale=30.,elec_field=0.0, nr_processes=1, read_from_file=''):
    nsite = len(struct.coords)
    latt = lattice(struct.coords)
    site_set = siteset(nsite)
    if os.path.isfile('sample.hdf5'):
        sp = tipsi.Sample(latt, site_set, nr_processes=nr_processes, read_from_file='sample.hdf5')
    else:
        sp = tipsi.Sample(latt, site_set, nr_processes=nr_processes)
        t0 = time.time()
        hop_dict = hopdict(struct, elec_field=elec_field)
        t1 = time.time()
        print('hop % s' % (t1-t0))
        del struct
        sp.add_hop_dict(hop_dict)
        sp.save()
    sp.rescale_H(rescale)
    return sp
