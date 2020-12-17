import numpy as np
from scipy.linalg.lapack import zheev
import json
import time
import sys
import copy
from monty.json import jsanitize
import multiprocessing as mp
from multiprocessing import Pool
import itertools
import os
from tBG.hopping import divide_sites_2D, calc_hoppings, hop_list_graphene_wannier, calc_hopping_wannier_PBC_new
from tBG.utils import *
from tBG.crystal.structures import _LayeredStructMethods

class _MethodsHamilt:
    def get_Hamiltonian(self):
        """
        Hamiltonian is in units of eV
        """
        pairs, ts = self.hopping
        ndim = len(self.coords)
        H = np.zeros((ndim,ndim), dtype=float)
        def put_value(pair, t):
            H[pair[0],pair[1]] =  t
            H[pair[1],pair[0]] =  np.conj(t)
        [put_value(pairs[i], ts[i]) for i in range(len(ts))]
        if self.E:
            np.fill_diagonal(H, self.Es_onsite)
        return H

    def diag_Hamiltonian(self, fname='EIGEN', vec=True):
        """
        fname: the file saving the eigenvalues and eigenvectors
        vec: True or False, whether eigenvectors are calculated
        E: the electric field strength eV/Angstrom
        """
        if vec:
            vec_calc = 1
        else:
            vec_calc = 0
        H = self.get_Hamiltonian()
        vals, vecs, info = zheev(H, vec_calc)
        if info:
            raise ValueError('zheev failed')
        if vec:
            np.savez_compressed(fname, vals=vals, vecs=vecs)
        else:
            np.savez_compressed(fname, vals=vals)

    def get_current_mat(self):
        """
        the matrix of current operator, in units of e*angstrom/second
        """
        hbar_eVs =  6.582119514*10**(-16)
        e = 1.
        c = e/(1j*hbar_eVs)
        ndim = len(self.coords)
        H = self.get_Hamiltonian()
        X = np.zeros([ndim, ndim])
        np.fill_diagonal(X, self.coords[:,0])
        Y = np.zeros([ndim, ndim])
        np.fill_diagonal(Y, self.coords[:,1])
        Jx = c*(np.matmul(X,H)-np.matmul(H,X)) 
        Jy = c*(np.matmul(Y,H)-np.matmul(H,Y)) 
        return Jx, Jy
    
    def get_Lz_mat(self):
        """
        the matrix of Lz in units of hbar
        """
        m = 9.10956 * 10**(-31) #kg
        hbar_eVs = 6.582119514 *10**(-16) # eV.s
        hbar_Js = 1.05457266 *10**(-34) #J·s
        h_Js = 6.62607015*10**(-34)
        c = m/(1j*hbar_eVs)/ hbar_Js * 10**(-20)
        ndim = len(self.coords)
        H = self.get_Hamiltonian()
        X = np.zeros([ndim, ndim])
        np.fill_diagonal(X, self.coords[:,0])
        Y = np.zeros([ndim, ndim])
        np.fill_diagonal(Y, self.coords[:,1])
        XHY = np.matmul(X, np.matmul(H,Y))
        YHX = np.matmul(Y, np.matmul(H,X))
        return c*(YHX-XHY)


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
                              0.0524,  -0.0209, -0.0148, -0.0211]):
        """
        *** just for the case of two layers ***
        max_dist: the interlayer hopping will be add if C-C distance less than max_dist
        P: pressure 
        ts: the hopping energies for the first 8-nearest neighbors for intralayer 
        """
        pmg_st = self.pymatgen_struct()

        nlayer = len(self.layer_nsites)
        vecs_site0 = [np.sum(self.layer_latt_vecs[i], axis=0)/3. for i in range(nlayer)]
        layer_vec_to_NN =  np.array([[i, -i] for i in vecs_site0])

        layer_inds = self._layer_inds()
        layer_inds_sublatt = self._layer_inds_sublatt()
        hopping = calc_hopping_wannier_PBC_new(pmg_st, layer_inds, layer_inds_sublatt, layer_vec_to_NN, \
                                       max_dist=max_dist, P=P, ts=ts, a=self.a)

        keys = [[i,j[-1]] for i in range(len(hopping)) for j in hopping[i]]
        values = [hopping[i][j] for i in range(len(hopping)) for j in hopping[i]]
        self.hopping = [np.array(keys), np.array(values)]

    def set_magnetic_field(self, B=0):
        """
        field is along the z aixs
        """
        self.B = B
        pairs, ts = self.hopping
        ts_B = np.zeros(len(ts), dtype=np.complex)
        PHI0 = 4135.666734
        c = 1j*np.pi*B/PHI0 
        def add_Peierls_substitution(ind):
            x0, y0, _ = self.coords[pairs[ind][0]] 
            x1, y1, _ = self.coords[pairs[ind][1]]
            # 0.01 for change angstrom^2 to nanometer^2
            ts_B[ind] =  ts[ind] * np.exp(c*(y1-y0)*(x1+x0)*0.01)
        [add_Peierls_substitution(ind) for ind in range(len(ts))]
        self.hopping = pairs, ts_B

    def set_electric_field(self, E=0):
        """
        field is along the z aixs
        """
        if E:
            self.E = E
            self.Es_onsite = self.coords[:,2]*E

class _Read:
    def from_relaxed_struct_from_file(self, filename):
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

    def read_struct_and_hopping(self, filename):
        d = np.load(filename)
        self.coords = d['coords']
        self.hopping = (d['hop_keys'], d['hop_vals'])

class _Output:
    def output_xyz_struct(self):
        atom_type = np.concatenate(tuple((np.repeat([i+1],self.layer_nsites[i]) for i in range(len(self.layer_nsites)))))
        coord_str = np.append(np.array([atom_type], dtype=str), self.coords.T, axis=0).T
        #coord_str = np.array(coord, ntype=str)
        coord_str = '\n'.join([' '.join(i) for i in coord_str])
        with open('struct_relaxed.xyz', 'w') as f:
            f.write('%s\n' % len(self.coords))
            f.write('Relaxed structure\n')
            f.write(coord_str)

    def output_lammps_struct(self, atom_style='full'):
        """
        atom_style: 'full' or 'atomic' which determine the format of the
                     atom part in data file
        atoms in different layers were given differen atom_type and molecule-tag
        """
        from lammps import write_lammps_datafile
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

    def plot(self, fig, ax ,site_size=3.0, dpi=600, lw=0.6, edge_cut=False):
        import matplotlib.collections as mc
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
            fig.canvas.draw()
            renderer = fig.canvas.renderer
            ax.add_collection(line)
            ax.draw(renderer)
            ax.scatter(self.coords[:,0][ind0:ind1+1], self.coords[:,1][ind0:ind1+1], \
                        s=site_size, color=cs[layer_type],linewidth=0)
        #if not edge_cut:
        #    for i in self.site_ids_edge:
        #        ax.scatter(self.coords[:,0][self.site_ids_edge[i][0]:self.site_ids_edge[i][1]+1],\
        #                    self.coords[:,1][self.site_ids_edge[i][0]:self.site_ids_edge[i][1]+1],\
        #                    s = site_size+50, color='purple', marker='*', linewidth=0)
        #else:
        #    edge_site_ids = self.edge_site_ids_by_distance(edge_cut)
        #    ax.scatter(self.coords[:,0][edge_site_ids], self.coords[:,1][edge_site_ids],\
        #                    s = site_size+50, color='purple', marker='*', linewidth=0)
        ax.set_aspect('equal')

class RoundDisk(_MethodsHamilt, _LayeredStructMethods, _Output, _Read):
    def make_structure(self, R, rotation_angle=30., a=2.46, h=3.36, overlap='hole', rm_dangling=True):
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
        va_bottom = np.array([a*np.cos(30.*np.pi/180.), -a*np.sin(30.*np.pi/180.)])
        vb_bottom = np.array([a*np.cos(30.*np.pi/180.), a*np.sin(30.*np.pi/180.)])

        va_top = rotate_on_vec(rotation_angle, va_bottom)
        vb_top = rotate_on_vec(rotation_angle, vb_bottom)

        self.latt_vec_bott = np.array([va_bottom, vb_bottom])
        self.latt_vec_top = np.array([va_top, vb_top])
        self.layer_latt_vecs = np.array([self.latt_vec_bott, self.latt_vec_top])
        
        if overlap == 'hole':
            self.site0 = np.array([1./3., 1./3.])
            self.site1 = np.array([2./3., 2./3.])
        elif overlap == 'atom':
            self.site0 = np.array([0., 0.])
            self.site1 = np.array([1./3., 1./3.])
        elif overlap == 'atom1':
            self.site0 = np.array([-1/3, -1/3])
            self.site1 = np.array([0., 0.])
        elif overlap == 'side_center':
            self.site0 = np.array([-1/6., -1/6.])
            self.site1 = np.array([1./6., 1./6.])
        else:
            raise ValueError('overlap %s can not be recognized!!' % overlap)
         
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
        self.B = 0
        self.E = 0
        self.Es_onsite = np.zeros(len(self.coords))


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
            cart_bs = np.append([frac2cart(site+cell, self.latt_vec_bott) for cell in cells], np.array([[0.]]*ncell), axis=1)
            cart_ts = np.append([frac2cart(site+cell, self.latt_vec_top) for cell in cells], np.array([[self.h]]*ncell), axis=1)
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


    def add_hopping_wannier_divide(self, max_dist=6.0, P=0, \
                            ts=[-2.8922, 0.2425, -0.2656, 0.0235, \
                              0.0524,  -0.0209, -0.0148, -0.0211], nr_processes=1):
        """
              NOTE:  ***** only for round disk not for other quantum dots *****

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


def round_disks_multilayer(R, thetas, overlap, a=2.46, h=3.35, rm_dangling=True):
    """
    R: the radius of the round disk
    thetas: the twist angles between closest layers
    overlap: the rotation centor 'atom' or 'hole'
    rm_dangling: True or False to remove the dangling bonds at edge
    """
    rd = RoundDisk()
    rd.make_structure(R, rotation_angle=thetas[0], a=a, h=h, overlap=overlap, rm_dangling=rm_dangling)
    for i in range(1, len(thetas)):
        theta = thetas[i] # relative orientation between the top layer and the layer which will be added
        ## the infomation of the top layer
        inds_i = rd._layer_inds()[-1]
        coords_i = rd.coords[inds_i[0]:inds_i[1]+1]
        latt_vec_i = rd.layer_latt_vecs[-1]
        z_i = rd.layer_zcoords[-1]

        ## the infomation of the layer which will be added
        h_new = rd.layer_zcoords[-1] + h
        rd.layer_zcoords.append(h_new)
        coords_new = rotate_on_vec(theta, coords_i)
        coords_new[:,-1] = h_new
        latt_vec_new = rotate_on_vec(theta, latt_vec_i)
        rd.coords = np.concatenate([rd.coords, coords_new])
        rd.layer_latt_vecs = np.concatenate([rd.layer_latt_vecs, [latt_vec_new]])
        rd.layer_nsites.append(rd.layer_nsites[-1])
        rd.layer_nsites_sublatt.append(rd.layer_nsites_sublatt[-1])
    return rd

    
    
