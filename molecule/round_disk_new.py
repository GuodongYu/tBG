import numpy as np
from scipy.linalg.lapack import zheev
import multiprocessing as mp
from tBG.hopping import divide_sites_2D, calc_hoppings
from tBG.utils import *
from tBG.crystal.structures import _LayeredStructMethods
from tBG.hopping import grouper


## functions for prinstine monolayer graphene round disk ##
## construct structure func: _round_disk_monolayer
## 
def _round_disk_monolayer(R, a=2.46, origin='hole', rm_dangling=True):
    """
    """
    r = R*a
    latt_vec = a*np.array([[np.sqrt(3)/2, -1/2],
                            [np.sqrt(3)/2,  1/2]])
    va_bottom = np.array([a*np.cos(30.*np.pi/180.), -a*np.sin(30.*np.pi/180.)])
    vb_bottom = np.array([a*np.cos(30.*np.pi/180.), a*np.sin(30.*np.pi/180.)])
    latt_vec = np.array([va_bottom, vb_bottom])
    if origin == 'hole':
        sites = np.array([[1./3., 1./3.],
                             [2./3., 2./3.]])
    elif origin == 'atom':
        sites = np.array([[0.,       0.],
                             [1./3., 1./3.]])
    elif origin == 'atom1':
        sites = np.array([[-1/3, -1/3],
                             [0.,     0.]])
    elif origin == 'side_center':
        sites = np.array([[-1/6., -1/6.],
                             [1/6.,   1/6.]])
    else:
        raise ValueError("origin must be in ['hole', 'atom', 'atom1', 'side_center']")

    ### coords of special points ###
    p0 = np.array([-2*r, 0])
    p1 = np.array([ 2*r, 0])
    p2 = np.array([0., -2*r/np.sqrt(3)])
    p3 = np.array([0.,  2*r/np.sqrt(3)])
    ### coords  
    p4 = np.array([-r/2,  np.sqrt(3)*r/2])
    p5 = np.array([-r/2, -np.sqrt(3)*r/2])
    p6 = np.array([ r/2,  np.sqrt(3)*r/2])
    p7 = np.array([ r/2, -np.sqrt(3)*r/2])
    ps = np.array([p0, p1, p2, p3, p4, p5, p6, p7])
    ps_frac = cart2frac(ps, latt_vec)

    def get_mn_limit(subsite):
        mns = ps_frac - subsite
        m_min = np.floor(np.min(mns))
        m_max = np.ceil(np.max(mns))
        return [int(m_min), int(m_max)]
    mns = [get_mn_limit(sites[i]) for i in [0, 1]]

    def get_coords(sub):
        mn = mns[sub]
        site = sites[sub]
        ucs = np.array([[i,j] for i in range(mn[0], mn[1]+1) for j in range(mn[0], mn[1]+1)])
        coords_cart = frac2cart(ucs+site, latt_vec)
        norms = np.linalg.norm(coords_cart, axis=1)
        ids_in = np.where(norms<=r)[0]
        return ucs[ids_in], coords_cart[ids_in]

    ucs_site0, coords_site0 = get_coords(0)
    ucs_site1, coords_site1 = get_coords(1)
    ucs = np.array([ucs_site0, ucs_site1])
    coords = np.array([coords_site0, coords_site1])
    del ucs_site0, ucs_site1, coords_site0, coords_site1

    def coords_no_dangling(ucs, coords):
        dist_edge = 2.5 * a/np.sqrt(3)
        def get_ids_edge(sub):
            site = sites[sub]
            norms = np.linalg.norm(coords[sub], axis=1)
            ids_edge = np.where(norms>=r-dist_edge)[0]
            return ids_edge
        ids_edge = [get_ids_edge(i) for i in [0,1]]

        def get_ids_dangling(sub):
            if sub==0:
                delta = (latt_vec[0]+latt_vec[1])/3
            elif sub==1:
                delta = -(latt_vec[0]+latt_vec[1])/3
            neighs = np.array([rotate_on_vec(theta, delta) for theta in [0, 120, 240]])
            coords_edge = coords[sub][ids_edge[sub]]
            coords_edge_neigh = np.array([coords_edge+i for i in neighs])
            norms_edge_neigh = np.linalg.norm(coords_edge_neigh, axis=2)
            n_outside = np.sum(np.array(norms_edge_neigh>r, dtype=int),axis=0)
            ids_dangling = np.where(n_outside==2)[0]
            return ids_edge[sub][ids_dangling]
        ids_dangling_site0 =  get_ids_dangling(0)
        ids_dangling_site1 =  get_ids_dangling(1)
        return [np.delete(coords[i], get_ids_dangling(i), axis=0) for i in [0,1]]

    if rm_dangling:
        return coords_no_dangling(ucs, coords), latt_vec
    else:
        return coords, latt_vec

### functions to find intralayer neighbors #########################
def sublatt_to_neigh_vecs(sublatt, nth, latt_vec):
    """
    get the vectors shifting sublatt to its nth neighbors
    args:
        sublatt: the sublatt 
        nth: the nth nearest neighbor
        latt_vec: the lattice vector of graphene
    *** note: latt_vec must have the angle of 60 degree between two vectors ***
    """
    v1, v2 = latt_vec
    if nth==1:
        delta = (v1+v2)/3
        vecs = [rotate_on_vec(i*120, delta) for i in range(3)]
        tp = 'AB'
    elif nth==2:
        delta = v1
        vecs = [rotate_on_vec(i*60, delta) for i in range(6)]
        tp = 'AA'
    elif nth==3:
        delta = -2/3*(v1+v2)
        vecs = [rotate_on_vec(i*120, delta) for i in range(3)]
        tp = 'AB'
    elif nth==4:
        delta1 = v1 + (v1+v2)/3
        delta2 = v2 + (v1+v2)/3
        vecs1 = [rotate_on_vec(i*120, delta1) for i in range(3)]
        vecs2 = [rotate_on_vec(i*120, delta2) for i in range(3)]
        vecs = np.concatenate([vecs1, vecs2], axis=0)
        tp = 'AB'
    elif nth==5:
        delta = v1+v2
        vecs = [rotate_on_vec(i*60, delta) for i in range(6)]
        tp = 'AA'
    elif nth==6:
        delta = 2*v1
        vecs = [rotate_on_vec(i*60, delta) for i in range(6)]
        tp = 'AA'
    elif nth==7:
        delta1 = v2 + 2/3*(2*v2-v1)
        delta2 = (v2-v1) + 2/3*(2*v2-v1)
        vecs1 = [rotate_on_vec(i*120, delta1) for i in range(3)]
        vecs2 = [rotate_on_vec(i*120, delta2) for i in range(3)]
        vecs = np.concatenate([vecs1, vecs2], axis=0)
        tp = 'AB'
    elif nth==8:
        delta = 4/3*(v1+v2)
        vecs = [rotate_on_vec(i*120, delta) for i in range(3)]
        tp = 'AB'
    if sublatt==0:
        return np.array(vecs)
    elif sublatt==1:
        if tp == 'AA':
            return np.array(vecs)
        elif tp == 'AB':
            return -np.array(vecs)

def coords_to_strlist(coords):
    string = ['~'.join([str(x) for x in i]) for i in coords]
    return string

def _neigh_pairs_inter_sublatts(coords, neigh_vecs, idx_from=0):
    """
    *** get inter-sublattice neighbor pairs (from sublatt A to B) in a graphene ***
    coords: [coords_A,coords_B] with the sublattice A and B grouped.
    neigh_vecs: all moving vectors move sublatt A to sublatt B
    idx_from: index of the 1st atom in coords_A, used for multilayer systems
    """
    n_neigh = len(neigh_vecs)
    pnts_A, pnts_B = coords
    del coords
    n_atomA = len(pnts_A)
    pnts_A_neigh = np.array([pnts_A + neigh_vecs[i] for i in range(n_neigh)])

    ## handling for string match
    pnts_A_neigh = np.round(pnts_A_neigh, 3)
    pnts_A = np.round(pnts_A, 3)
    pnts_B = np.round(pnts_B, 3)
    pnts_A_neigh[pnts_A_neigh==0.0] = 0.0
    pnts_A[pnts_A==0.0] = 0.0
    pnts_B[pnts_B==0.0] = 0.0

    ## conver coords to string list
    pnts_A_neigh_str = np.array([coords_to_strlist(pnts_A_neigh[i]) for i in range(n_neigh)])
    pnts_B_str = coords_to_strlist(pnts_B)
    ind_B_str = dict(zip(pnts_B_str, range(len(pnts_B_str))))

    ##
    pairs = [[] for i in range(n_atomA)]
    def put_ind_pair(i, one_item):
        try:
            pairs[i].append(ind_B_str[one_item]+n_atomA)
        except:
            pass 
    for i_neigh in range(n_neigh):
        [put_ind_pair(i, pnts_A_neigh_str[i_neigh][i]) for i in range(n_atomA)]
    pairs_detail = np.array([[i, k] for i in range(len(pairs)) for k in pairs[i]])
    return pairs_detail + [idx_from, idx_from]

def _neigh_pairs_intra_sublatts(pnts, neigh_vecs, idx_from=0):
    """
    *** get intra-sublattice neighbor pairs (within sublatt A or B) in a graphene ***
    coords: coords of one sublattice
    neigh_vecs: all vectors move sublatt A to B
    idx_from: a given index for the first atom in coords_A
       **sometimes the coordinates given is just part of the system, such as sublatt B or multilayer
       ** and a non-zero idx_from is needed for getting the neighbor pairs
    """
    n_site = len(pnts)
    n_neigh = len(neigh_vecs)
    pnts_neigh = np.array([pnts + neigh_vecs[i] for i in range(n_neigh)])

    ## handling for string match
    pnts_neigh = np.round(pnts_neigh, 3)
    pnts = np.round(pnts, 3)
    pnts_neigh[pnts_neigh==0.0] = 0.0
    pnts[pnts==0.0] = 0.0

    ## conver coords to string list
    pnts_neigh_str = np.array([coords_to_strlist(pnts_neigh[i]) for i in range(n_neigh)])
    pnts_str = coords_to_strlist(pnts)
    ind_str = dict(zip(pnts_str, range(len(pnts_str))))

    ##
    pairs = [[] for i in range(n_site)]
    def put_ind_pair(i, one_item):
        try:
            pairs[i].append(ind_str[one_item])
        except:
            pass 
    for i_neigh in range(n_neigh):
        [put_ind_pair(i, pnts_neigh_str[i_neigh][i]) for i in range(n_site)]
    pairs_detail = np.array([[i, k] for i in range(len(pairs)) for k in pairs[i]])
    pairs_detail_filtered = np.unique(np.sort(pairs_detail, axis=1), axis=0)
    return pairs_detail_filtered + [idx_from, idx_from]

def get_neigh_pairs_in_graphene_OBC(nth, coords, latt_vec, idx_from=0):
    """
    args:
        nth: the nth nearested neighbors are considered
        coords: [coords_sublatt_a, coords_sublatt_b] 
        latt_vec: lattice vector 
        idx_from: the index of the 1st site in coords_sublatt_a, used for multilayer

    return 
        array [pairs_1st_list, pair_2nd_list, pair_3rd_list,...., pair_nth_list]
        only half pairs are saved, such as, if [1,2] is in, [2,1] is not.
    """    
    n_site_a = len(coords[0])
    n_site_b = len(coords[1])
    tps = {1:'inter', 2:'intra', 3:'inter', 4:'inter', 5:'intra', 6:'intra', 7:'inter', 8:'inter'}
    pairs =[[] for i in range(nth)] # pairs[i] for ith+1 nearest neighbor
    for i in range(nth):
        tp = tps[i+1]
        if tp == 'inter':
            neigh_vecs = sublatt_to_neigh_vecs(0, i+1, latt_vec)
            pairs_i = _neigh_pairs_inter_sublatts([coords[0][:,0:2], coords[1][:,0:2]], neigh_vecs, idx_from=idx_from)
        elif tp == 'intra':
            pairs_i = []
            for sub in [0,1]:
                neigh_vecs = sublatt_to_neigh_vecs(sub, i+1, latt_vec)
                pairs_i_sub=_neigh_pairs_intra_sublatts(coords[sub][:,0:2], neigh_vecs, idx_from=idx_from+sub*n_site_a)
                pairs_i.append(pairs_i_sub)
            pairs_i = np.concatenate(pairs_i, axis=0)
        pairs[i] = pairs_i
    return pairs

def get_neigh_pairs_in_graphene_OBC_in_parellel(nth, coords, latt_vec, idx_from=0, n_proc=1):
    """
    args:
        nth: the nth nearested neighbors are considered
        coords: [coords_sublatt_a, coords_sublatt_b] 
        latt_vec: lattice vector 
        idx_from: the index of the first site in coords_sublatt_a, used for multilayer

    return 
        array [pairs_1st_list, pair_2nd_list, pair_3rd_list,...., pair_nth_list]
        only half pairs are saved, such as, if [1,2] is in, [2,1] is not.
    """    
    if nth>8:
        raise ValueError('nth maximum exceed 8!')
    n_site_a = len(coords[0])
    n_site_b = len(coords[1])
    tps = {1:'inter', 2:'intra', 3:'inter', 4:'inter', 5:'intra', 6:'intra', 7:'inter', 8:'inter'}
    def neighs_nth_shot(nth_shot, conn=False):
        pairs_shot = []
        for i in nth_shot:
            tp = tps[i+1]
            if tp == 'inter':
                neigh_vecs = sublatt_to_neigh_vecs(0, i+1, latt_vec)
                pairs_i = _neigh_pairs_inter_sublatts([coords[0][:,0:2], coords[1][:,0:2]], neigh_vecs, idx_from=idx_from)
            elif tp == 'intra':
                pairs_i = []
                for sub in [0,1]:
                    neigh_vecs = sublatt_to_neigh_vecs(sub, i+1, latt_vec)
                    pairs_i_sub=_neigh_pairs_intra_sublatts(coords[sub][:,0:2], neigh_vecs, idx_from=idx_from+sub*n_site_a)
                    pairs_i.append(pairs_i_sub)
                pairs_i = list(np.concatenate(pairs_i, axis=0))
            pairs_shot.append(pairs_i)
        out = dict(zip(nth_shot, pairs_shot))
        if conn:
            conn.send(out)
            return
        else:
            return out
    nth_all = range(nth)
    if n_proc==1:
        pairs = neighs_nth_shot(nth_all, conn=False)
        pairs = [pairs[i] for i in nth_all]
    else:
        nth_shots, N = grouper(nth_all, n_proc)
        nth_shots = ([i for i in nth_shots])
        N = len(nth_shots)
        pipes = [mp.Pipe() for i in range(N)]
        processes = [None for i in range(N)]
        data = [None for i in range(N)]
        for i, tags in enumerate(nth_shots):
            pipe = pipes[i]
            processes[i] = mp.Process(target=neighs_nth_shot, \
                                      args=(tags, pipe[1]))
            processes[i].start()

        scan = [True for i in range(N)]
        while any(scan):
            for i in range(N):
                pipe = pipes[i]
                if scan[i] and pipe[0].poll():
                    # get data, close process
                    data[i] = pipe[0].recv()
                    scan[i] = False
                    pipe[0].close()
                    processes[i].join()
        pairs={}
        [pairs.update(i) for i in data]
        pairs = [pairs[i] for i in nth_all]
    return pairs
################################################################

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
                              0.0524,  -0.0209, -0.0148, -0.0211], nr_processes=1):
        """
        max_dist: the interlayer hopping will be add if C-C distance less than max_dist
        lambda0,3,6, xi0,3,6 k0,6 and x3,6 are the params for interlayer hopping 
        ts: the hopping energies for the first 8-nearest neighbors for intralayer 
        """
        from tBG.hopping import hop_params_wannier_interlayer, hop_func_wannier_interlayer
        lambda0,xi0,k0,lambda3,xi3,x3,lambda6,xi6,x6,k6 = \
               hop_params_wannier_interlayer(P)

        n_layer = len(self.layer_origins)

        pairs = []
        hops = []
        #### intra-layer hopping ###
        nth = len(ts)
        done_layers = {}
        for i_layer in range(n_layer):
            origin = self.layer_origins[i_layer]
            if origin not in done_layers:
                idx_a, idx_b = self._layer_inds_sublatt()[i_layer]
                #pairs_i_layer = get_neigh_pairs_in_graphene_OBC(nth, [self.coords[idx_a[0]:idx_a[1]+1], self.coords[idx_b[0]:idx_b[1]+1]],\
                #                                self.layer_latt_vecs[i_layer], idx_from=idx_a[0])
                pairs_i_layer = get_neigh_pairs_in_graphene_OBC_in_parellel(nth, \
                                             [self.coords[idx_a[0]:idx_a[1]+1], self.coords[idx_b[0]:idx_b[1]+1]],\
                                             self.layer_latt_vecs[i_layer], idx_from=idx_a[0], n_proc=nr_processes)
                pairs.append(np.concatenate(pairs_i_layer, axis=0))
                hops_i_layer = np.concatenate([np.repeat(ts[i], len(pairs_i_layer[i])) for i in range(nth)], axis=0)
                hops.append(hops_i_layer)
                done_layers[origin] = i_layer
            else:
                i_layer_exist = done_layers[origin]
                idx_from_exist = self._layer_inds_sublatt()[i_layer_exist][0][0]
                idx_from = self._layer_inds_sublatt()[i_layer][0][0]
                pairs.append(pairs[i_layer_exist]-idx_from_exist+idx_from)
                hops.append(hops[i_layer_exist])
        pairs = np.concatenate(pairs, axis=0)
        hops = np.concatenate(hops, axis=0)
            
        ## inter-layer hopping (only between nearest layers)
        def collect_sublatt_data(i_layer, sublatt):
            idx_range = self._layer_inds_sublatt()[i_layer][sublatt]
            sites = self.coords[idx_range[0]:idx_range[1]+1]
            bins = divide_sites_2D(sites, bin_box=[[max_dist,0],[0,max_dist]], idx_from=idx_range[0])
            v1, v2 = self.layer_latt_vecs[i_layer]
            vec_to_NN = (v1+v2)/3 if sublatt==0 else -(v1+v2)/3
            return sites, bins, vec_to_NN
       
        a = np.linalg.norm(self.layer_latt_vecs[0][0][0:2]) # lattice constant of graphene
        for i in range(n_layer-1):
            for sub0 in [0,1]:
                site0s, bin0s, vec0_to_NN = collect_sublatt_data(i, sub0)
                site0s[:,-1] = 0.
                for sub1 in [0,1]:
                    site1s, bin1s, vec1_to_NN = collect_sublatt_data(i+1, sub1)
                    site1s[:,-1] = 0.
                    hop_func = hop_func_wannier_interlayer(vec0_to_NN, vec1_to_NN, lambda0, xi0, k0, \
                                                           lambda3, xi3, x3, lambda6, xi6, x6, k6, a)
                    pair, hop = calc_hoppings(site0s, bin0s, site1s, bin1s, \
                         hop_func=hop_func, max_dist=max_dist, nr_processes=nr_processes)
                    pairs = np.concatenate([pairs, pair], axis=0)
                    hops = np.concatenate([hops, hop], axis=0)
        self.hopping = pairs, hops



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
    """
    A class for constructing graphene round disk quantum dots, including monolayer, bilayer and multilayer.
    The lattice vector [vec_0, vec1]: the one with 60 degree angle between vec_0 and vec_1 is always chosen.
    """
    def monolayer(self, R, a=2.46, orientation=0, origin='hole', rm_dangling=True):
        """
        Args: 
            R: round disk radius (in units of a)
            a: graphene lattice constant, default 2.46
            origin: four positions, 'hole', 'atom', 'atom1' and 'side_center', can be chosen
            rm_dangling: whether to remove the edge atoms with two dangling bonds
        Units:
            distance: Angstrom
            rotation_angle: degree
        """
        coords, latt_vec = _round_disk_monolayer(R, a=a, origin=origin, rm_dangling=rm_dangling)
        n_atom_a = len(coords[0])
        n_atom_b = len(coords[1])
        self.layer_latt_vecs = [latt_vec]
        self.layer_nsites_sublatt = [[n_atom_a, n_atom_b]]
        self.layer_nsites = [ n_atom_a+n_atom_b ]
        self.layer_origins = [origin]
        self.coords = np.concatenate(coords, axis=0)
        self.coords = np.append(self.coords, [[0]]*(n_atom_a+n_atom_b), axis=1)
        self.Es_onsite = np.zeros(len(self.coords))

    def twisted_bilayer(self, R, rotation_angle=30, h=3.35, a=2.46, overlap='hole', rm_dangling=True):
        """
        a twisted bilayer graphene from AA stacking (AA for 0 degree rotation angle)        

        Args: 
            R: round disk radius (in units of a)
            rotation_angle: relative roatation angle in units of degree
            a: graphene lattice constant, default 2.46
            h: interlayer distance
            overlap: rotation center 'hole', 'atom', 'atom1', 'side_center'
            rm_dangling: whether to remove the edge atoms with two dangling bonds
        Units:
            distance: Angstrom
            rotation_angle: degree
        """
        self.monolayer(R, a=a, origin=overlap, rm_dangling=rm_dangling)
        n_site = self.layer_nsites[0]
        self.layer_latt_vecs.append(rotate_on_vec(rotation_angle, self.layer_latt_vecs[0]))
        self.layer_nsites.append(n_site)
        self.layer_nsites_sublatt.append(self.layer_nsites_sublatt[0])
        self.layer_origins = [overlap]*2
        coords_second = np.append(rotate_on_vec(rotation_angle, self.coords[:,0:2]), [[h]]*n_site, axis=1)
        self.coords = np.append(self.coords, coords_second, axis=0)
        self.Es_onsite = np.zeros(len(self.coords))

    def twisted_multilayer(self, R, a=2.46, orientations=[0, 30], origins=['hole','hole'], h=3.35, rm_dangling=True):
        """
        R: round disk radius R*a
        a: lattice constant of graphene
        orientations: orientations of all layers in units of degree
        origins: the origin place of all layers
        h: interlayer distance in units of angstrom
        rm_dangling: whether to remove the single dangling bonds
        """
        ### check inputs 
        if len(orientations)!=len(origins):
            raise ValueError('Sizes of orientations and origins are different!')
        for i in origins:
            if i not in ['hole', 'atom', 'atom1', 'side_center']:
                raise ValueError("origin must be in ['hole', 'atom', 'atom1', 'side_center']")
        #########################################################

        n_layer = len(orientations)
        self.layer_origins = origins
        self.layer_nsites = []
        self.layer_nsites_sublatt = []
        self.layer_latt_vecs = []
        
        coords_all = []
        layers_done = {}
        for i in range(n_layer):
            orient = orientations[i]
            origin = origins[i]
            z_coord = h*i
            if origin in layers_done:
                ith_layer = layers_done[origin]
                coords_exist = coords_all[ith_layer]
                orient_exist = orientations[ith_layer]
                latt_vec_exist = self.layer_latt_vecs[ith_layer]
                theta = orient - orient_exist
                coords_new = rotate_on_vec(theta, coords_exist)
                del coords_exist
                latt_vec_new = rotate_on_vec(theta, latt_vec_exist)
                self.layer_nsites.append(self.layer_nsites[ith_layer])
                self.layer_nsites_sublatt.append(self.layer_nsites_sublatt[ith_layer])
                self.layer_latt_vecs.append(latt_vec_new)
                coords_all.append(coords_new) 
            else:
                layers_done[origin]=i
                coords, latt_vec = _round_disk_monolayer(R, a=a, origin=origin, rm_dangling=rm_dangling)
                n_site_a = len(coords[0])
                n_site_b = len(coords[1])
                n_site = n_site_a + n_site_b
                coords = np.concatenate(coords)
                if orient !=0:
                    coords = rotate_on_vec(orient, coords)
                    latt_vec = rotate_on_vec(orient, latt_vec)
                coords_all.append(coords)
                del coords
                self.layer_nsites.append(n_site)
                self.layer_nsites_sublatt.append([n_site_a, n_site_b])
                self.layer_latt_vecs.append(latt_vec)
        z_coords = np.concatenate([[h*i]*self.layer_nsites[i] for i in range(n_layer)])
        self.coords = np.append(np.concatenate(coords_all), z_coords.reshape(-1,1), axis=1)
        self.Es_onsite = np.zeros(len(self.coords))
                    
