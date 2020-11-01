import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import itertools
import copy

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

def divide_sites_2D(sites, bin_box=[[5,0],[0,5]], idx_from=0):
    """
    coords: the coords of all sites 
    bin_box: the 2D bin 
    index_start: the starting index of the sites
    """
    sites = sites[:,0:2]
    # define bins
    bin_xvec = bin_box[0]
    bin_yvec = bin_box[1]
    #print("bin numbers: ", Nw_bins, Nh_bins, "the spatial mesh: ", bin_xvec, bin_yvec)

    # transform [i,j] to Cartesian bin location
    T = np.array([[bin_xvec[0], bin_yvec[0]],
                  [bin_xvec[1], bin_yvec[1]]])
    # transform Cart location to [i,j]
    T_inv = np.linalg.inv(T)
        
    # define bin indices for coordinate
    def get_bindices(site):
        # take dot product
        inds = np.dot(T_inv, np.array([site[0], site[1]]))
        # get indices
        i, j = int(np.floor(inds[0])), int(np.floor(inds[1]))
        return i, j
        
    # organize coordinates into bins
    #bins = [[[] for i in range(Nh_bins)] for i in range(Nw_bins)]
    bins={}
    k = idx_from
    for site in sites:
        i,j = get_bindices(site)
        try:
            bins[(i,j)].append(k)
        except:
            bins[(i,j)]=[k]
        k = k + 1
    return bins, idx_from

class SparseHopDict:
    """Sparse HopDict class

    A hopping dictionary contains relative hoppings.
    
    Attributes
    ----------
    dict : list of dictionaries
        dictionaries containing hoppings
    """

    def __init__(self, n):
        """Initialize hop_dict object
        """
        self.dict = [{} for i in range(n)]

    def set_element(self, rel_unit_cell, element, hop):
        """Add single hopping to hopping matrix.
        
        Parameters
        ----------
        rel_unit_cell : 3-tuple of integers
            relative unit cell coordinates
        element : 2-tuple of integers
            element indices
        hop : complex float
            hopping value
        """
        self.dict[element[0]][rel_unit_cell + (element[1],)] = hop
    
    def add_conjugates(self):
        """Adds hopping conjugates to self.dict."""
        
        # declare new dict
        self.new_dict = copy.deepcopy(self.dict)
        
        # iterate over items
        for i in range(len(self.dict)):
            for rel_tag, hopping in self.dict[i].items():
                x, y, z, j = rel_tag
                reverse_tag = (-x, -y, -z, i)
                reverse_hopping = np.conjugate(np.transpose(hopping))
                if reverse_tag not in self.new_dict[j]:
                    self.new_dict[j][reverse_tag] = reverse_hopping
                
        # done
        self.dict = self.new_dict
        
    def sparse(self, nr_processes=1):

        self.add_conjugates()
        return self.dict

def hop_func_pz(g0=3.12, a0=1.42, g1=0.48, h0=3.349, rc=6.14, lc=0.265, q_dist_scale=2.218):
    """
    the function defines the hopping value between two orbitals
    r is the relative vector between the two orbitals
    Note:
        The units of all quantities below is angstrom
    """
    def hop_func(r1, r2):
        try:
            dr = np.linalg.norm(r1-r2, axis=1)
            dz = np.abs(r1[:,-1]-r2[:,-1]) 
        except:
            dr = np.linalg.norm(r1-r2, axis=0)
            dz = np.abs(r1[-1]-r2[-1]) 
        n = dz / dr
        V_pppi = -g0 * np.exp( q_dist_scale * (a0 - dr) )
        V_ppsigma = g1 * np.exp( q_dist_scale * (h0 - dr) )
        hop = n**2 * V_ppsigma + (1 - n**2) * V_pppi
        return hop / (1 + np.exp((dr - rc) / lc)) # smooth cutoff
    return hop_func

def hop_params_wannier_interlayer(P):
    """
    get interlayer hopping parameters under a specific pressure
    prb 98 085114(2018) 
    prb 99 205141 (2019)
    """
    cs={'lambda0':[0.310, -1.882, 7.741], 'xi0':[1.750, -1.618, 1.848],\
        'k0':[1.990, 1.007, 2.427], 'lambda3':[-0.068, 0.399, -1.739],\
        'xi3':[3.286, -0.914, 12.011], 'x3':[0.500, 0.322, 0.908],\
        'lambda6':[-0.008, 0.046, -0.183], 'xi6':[2.272, -0.721, -4.414],\
        'x6':[1.217, 0.027, -0.658],'k6':[1.562, -0.371, -0.134]}
    delta = 0.1048*np.log(1+P/5.73) # compress delta = 1 - d/d0
    return [cs[i][0]-cs[i][1]*delta+cs[i][2]*delta**2 for i in \
       ['lambda0','xi0','k0','lambda3','xi3','x3','lambda6','xi6','x6','k6']]

def hop_func_wannier_interlayer(vec1_to_NN, vec2_to_NN, lambda0, xi0, k0, lambda3, xi3, x3, lambda6, xi6, x6, k6, a):
    """
    hopping function between layers

    PRB 93 235153 (2016)
    related papers:
        prb 98 085114 (2018)
        prb 99 205141 (2019)
    """
    def hop_func(r1, r2):
        try:
            r = r1[:,0:2] - r2[:,0:2]
            dr = np.linalg.norm(r, axis=1)
        except:
            r = np.array([r1[0:2] - r2[0:2]])
            dr = np.array([np.linalg.norm(r, axis=0)])

        b = np.linalg.norm(vec1_to_NN[0:2])
        
        cos_theta12 = np.dot(-r, vec1_to_NN[0:2])/(dr*b)
        cos_theta12 = np.nan_to_num(cos_theta12)
        cos_theta21 = np.dot(r, vec2_to_NN[0:2])/(dr*b)
        cos_theta21 = np.nan_to_num(cos_theta21)

        cos_3theta12 = -3*cos_theta12 + 4*cos_theta12**3
        cos_3theta21 = -3*cos_theta21 + 4*cos_theta21**3

        cos_6theta12 = 2*cos_3theta12**2 -1
        cos_6theta21 = 2*cos_3theta21**2 -1

        V0 = lambda0 * np.exp(-xi0*(dr/a)**2) * np.cos(k0*(dr/a))
        V3 = lambda3 * (dr/a)**2 * np.exp(-xi3*((dr/a)-x3)**2)
        V6 = lambda6 * np.exp(-xi6*((dr/a)-x6)**2) * np.sin(k6*(dr/a))
        return V0 + V3*(cos_3theta12+cos_3theta21) + V6*(cos_6theta12+cos_6theta21)
    return hop_func

def hop_list_graphene_wannier(ts=[-2.8922, 0.2425, -0.2656, 0.0235, \
                                   0.0524,  -0.0209, -0.0148, -0.0211]):

    """
    hop list in one layer of graphene
    used for round_disk structure
    """
    hop = [{},{}]
    hop[0][(-1, 0, 1)]= ts[0] 
    hop[0][(0, -1, 1)]= ts[0] 
    hop[0][(0, 0, 1) ]= ts[0] 
    try:
        hop[0][(0, -1, 0)]= ts[1] 
        hop[0][(-1, 1, 0)]= ts[1] 
        hop[0][(-1, 0, 0)]= ts[1] 
    
        hop[1][(0, -1, 1)]= ts[1]
        hop[1][(-1, 1, 1)]= ts[1]
        hop[1][(-1, 0, 1)]= ts[1]
    except:
        return hop
    try:
        hop[0][(-1, -1, 1)]= ts[2] 
        hop[0][(1, -1, 1)]= ts[2]
        hop[0][(-1, 1, 1)]= ts[2]
    except:
        return hop
    try:
        hop[0][(0, -2, 1)]= ts[3] 
        hop[0][(-2, 1, 1)]= ts[3]
        hop[0][(-2, 0, 1)]= ts[3]
        hop[0][(1, -2, 1)]= ts[3]
        hop[0][(1, 0, 1)]= ts[3]
        hop[0][(0, 1, 1)]= ts[3]
    except:
        return hop
    try:
        hop[0][(-1, -1, 0)]= ts[4] 
        hop[0][(-2, 1, 0)]= ts[4]
        hop[0][(1, -2, 0)]= ts[4]

        hop[1][(-1, -1, 1)]= ts[4]
        hop[1][(-2, 1, 1)]= ts[4]
        hop[1][(1, -2, 1)]= ts[4]
    except:
        return hop
    try:
        hop[0][(0, -2, 0)]= ts[5] 
        hop[0][(-2, 2, 0)]= ts[5] 
        hop[0][(-2, 0, 0)]= ts[5] 

        hop[1][(0, -2, 1)]= ts[5]
        hop[1][(-2, 2, 1)]= ts[5]
        hop[1][(-2, 0, 1)]= ts[5]
    except:
        return hop
    try:
        hop[0][(-1, -2, 1)]= ts[6] 
        hop[0][(-2, -1, 1)]= ts[6]
        hop[0][(-2, 2, 1)]= ts[6]
        hop[0][(2, -1, 1)]= ts[6]
        hop[0][(2, -2, 1)]= ts[6]
        hop[0][(-1, 2, 1)]= ts[6]
    except:
        return hop
    try:
        hop[0][(-3, 1, 1)]= ts[7] 
        hop[0][(1, -3, 1)]= ts[7]
        hop[0][(1, 1, 1)]= ts[7]
    except:
        return hop

    return hop

def calc_hoppings(site0s, bin0s, site1s=[], bin1s={}, hop_func=hop_func_pz(), max_dist=5.0, nr_processes=1):
    """
    system 0: site0s (coordinates) and bin0s (classified site0s in bins)
    system 1: site1s (coordinates) and bin1s (classified site1s in bins)
 
    system 1 exists: to calculate hoppings between system 0 and 1
    system 1 absent: to calculate hoppings within system 0

    Other args:
        hop_func: the hopping function 
        nr_processes: number of cores for parallel calculations

    Note:
        For different layers, all hopping between bins are inter (_hops_between_bins)
        For one layer, the hoppings between different bins are inter (_hops_between_bins)
                             the hoppings in the same bin are intra (hops_within_bin)
    """
    bin0s, idx0_from = bin0s

    if len(site1s):
        ## the case of two systems
        bin1s, idx1_from = bin1s
        hop_type = 'inter'
        neighs = [(-1,-1),(-1,0),(-1,1), \
                  (0, -1),(0, 0),(0, 1), \
                  (1, -1),(1, 0),(1, 1)]
    else:
        ## the case of one system 
        bin1s = bin0s
        idx1_from = idx0_from
        site1s = site0s
        hop_type = 'intra'
        neighs = [(1,0), (0,1), (1,1), (1,-1)]
    ## hop_type of intra and inter does not mean interlayer or intralayer
    ## inter means that there are two parts of the system, we just want to get the hopping between them
    ## intra means that there is only one part of the system, we want to get the hopping within it
    ## in any case, the system are divided into small bins.
    
    def hops_within_bin(bin0):
        """
        hop between sites in the same bin
        this function is used for the intra-layer hopping 
        """
        keys = np.array([[id0,id1] for id0 in bin0s[bin0] for id1 in bin0s[bin0] if id0<id1])
        # for the case only one site inside
        if not len(keys):
            return (np.empty((0,2), dtype=int), [])
        r0s = site0s[keys[:,0]-idx0_from]
        r1s = site0s[keys[:,1]-idx0_from]
        dr = np.linalg.norm(r0s-r1s, axis=1)
        ids = np.where(dr<=max_dist)[0]
        keys = keys[ids]
        r0s = r0s[ids]
        r1s = r1s[ids] 
        values = hop_func(r0s, r1s)
        return (keys, values)

    def _hops_between_bins(bin0, bin1):
        """
        For the bins of two layers
        and the different bins in one layer 
        """
        keys = np.array([[id0,id1] for id0 in bin0s[bin0] for id1 in bin1s[bin1]])
        # for the case of no neighbors existing
        if not len(keys):
            return (np.empty((0,2), dtype=int), [])
        r0s = site0s[keys[:,0]-idx0_from]
        r1s = site1s[keys[:,1]-idx1_from]
        dr = np.linalg.norm(r0s-r1s, axis=1)
        ids = np.where(dr<=max_dist)[0]
        if not len(ids):
            return (np.empty((0,2), dtype=int), [])
        keys = keys[ids]
        r0s = r0s[ids]
        r1s = r1s[ids] 
        values = hop_func(r0s, r1s)
        return (keys, values)

    def hops_with_neighs(bin0):
        # go over the neighbors of bin0 
        # neighbor bins are saved in parameter neighs 
        # (9 and 4 neighbors for inter- and intra-layers respectively)
        neigh_bins = [(bin0[0]+i[0], bin0[1]+i[1]) for i in neighs]
        neigh_bins = set(neigh_bins) & set(bin1s.keys())
        out = [_hops_between_bins(bin0, bin1) for bin1 in neigh_bins]
        try:
            keys = np.concatenate([i[0] for i in out])
            values = np.concatenate([i[1] for i in out])
            return keys, values
        except:
            return np.empty((0,2), dtype=int), []
    
    def add_hop_bins(bins0, conn=False):
        ## bin0 and its neighbor bins
        out = [hops_with_neighs(bin0) for bin0 in bins0]
        keys = np.concatenate([i[0] for i in out]) 
        values = np.concatenate([i[1] for i in out])
        if hop_type == 'intra':
            ## within bin0
            out_intra = [hops_within_bin(bin0) for bin0 in bins0]
            keys_intra = np.concatenate([i[0] for i in out_intra])
            values_intra = np.concatenate([i[1] for i in out_intra])
            ## intra case includes hoppings between bin0 and its neighbors and hoppings within bin0
            keys = np.append(keys, keys_intra, axis=0)
            values = np.append(values, values_intra, axis=0)
        if conn:
            conn.send((keys, values))
            return
        else:
            return (keys, values)

    if nr_processes==1:
        keys, values = add_hop_bins(bin0s)
    else:
        sites_div, N = grouper(list(bin0s.keys()), nr_processes)
        pipes = [mp.Pipe() for i in range(N)]
        processes = [None for i in range(N)]
        data = [None for i in range(N)]
        for i, tags in enumerate(sites_div):
            pipe = pipes[i]
            processes[i] = mp.Process(target=add_hop_bins, \
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
        keys = np.concatenate(tuple(i[0] for i in data))
        values = np.concatenate(tuple(i[1] for i in data))
        del data
    return  np.array(keys), np.array(values)

def filter_neig_list(neig_list):
    """
    a, b, c, d = neigh_list
    a: the center inds
    b: the point inds
    c: the offsets of points
    d: the distances

    Note: if in neigh_list, begin and end sites belong to different group, please don't use this function

    input 
        neigh_list: the neighbor_list output by get_neighbor_list function in pymatgen.core.structure.Structrue.

    return
        a new list with repeated neighbor pairs removed 
        namely the element such as a[i]=0, c[i]=1, b[i]=[1,2,0] was removed if a[j]=1, c[j]=0, b[j]=[-1,-2,0] exists
    """
    a, b, c, d = neig_list
    c = np.array(c, dtype=int)
    x = np.concatenate((a.reshape(-1,1), c[:,0:2], b.reshape(-1,1)), axis=1)
    y = np.concatenate((b.reshape(-1,1), -c[:,0:2], a.reshape(-1,1)), axis=1)
    x_str = np.array([''.join(i) for i in np.array(x, dtype=str)])
    y_str = np.array([''.join(i) for i in np.array(y, dtype=str)])
    argsort_x = np.argsort(x_str)
    argsort_y = np.argsort(y_str)
    pairs = np.array([argsort_x, argsort_y]).T
    pairs = np.sort(pairs, axis=1)
    ids = np.unique(pairs[:,0])
    return [a[ids], b[ids], c[ids], d[ids]]

def calc_hopping_pz_PBC(pmg_struct, Rcut_inter=5.0, Rcut_intra=5.0, g0=3.12, a0=1.42, g1=0.48, h0=3.45, \
                           rc=6.14, lc=0.265, q_dist_scale=2.218, layer_inds=[]):
    """
    to calculate the pz-pz hoppings
    
    args:
        pmg_struct: the pymatgen structure (instance of pymatgen.core.structure.Structure)
        max_dist: only the hopping with C-C distance less than max_dist will be calculated
        g0, a0, g1, h0, rc, lc, q_dist_scale: the hopping parameters for  hop_func_pz function
        inter_hopping: (default True) whether the inter-layer hopping are calculated.
                   The interlayer hoppings are ignored when False, which is often used for comparison
        layer_inds: (ignore this when inter_hopping=True) the list saving the index range of all layer.
                    It should be the retured value of function tBG.TBG30_approximant.structure.Structure._layer_inds()
                    or tBG.graphene.structure.Structure._layer_inds()
                    example: for a case with two layers [[0,19],[20,39]], the site index in 1st and 2nd
                    layers are [0,19] and [20,39] including two ends
    """
    from functools import reduce
    ng_list = pmg_struct.get_neighbor_list(max(Rcut_inter, Rcut_intra))
    nlayer = len(layer_inds)
    if Rcut_inter != Rcut_intra:
        ids_intra = np.array([], dtype=int)
        ids_inter = np.array([], dtype=int)
        for i in range(nlayer):
            id0, id1 = layer_inds[i]
            ids_b0 = np.where(ng_list[0]>=id0)[0]
            ids_b1 = np.where(ng_list[0]<=id1)[0]
            ### intralayer ###
            ids_e0_intra = np.where(ng_list[1]>=id0)[0]
            ids_e1_intra = np.where(ng_list[1]<=id1)[0]
            ids_d_intra = np.where(ng_list[3]<=Rcut_intra)[0]
            ### interlayer ### 
            ids_e_inter = np.where(ng_list[1]>id1)[0]
            ids_d_inter = np.where(ng_list[3]<=Rcut_inter)[0]

            ids_intra = np.append(ids_intra, reduce(np.intersect1d, (ids_b0, ids_b1, ids_e0_intra, ids_e1_intra, ids_d_intra)), axis=0)
            ids_inter = np.append(ids_inter, reduce(np.intersect1d, (ids_b0, ids_b1, ids_e_inter, ids_d_inter)), axis=0)
        ng_list_inter = [ng_list[k][ids_inter] for k in range(4)]
        ng_list_intra = [ng_list[k][ids_intra] for k in range(4)]
        ng_list_intra = filter_neig_list(ng_list_intra)
        ng_list = [np.append(ng_list_inter[k], ng_list_intra[k], axis=0) for k in range(4)]
        ng_list[2] = np.array(ng_list[2], dtype=int)
    else:
        ng_list = filter_neig_list(ng_list)
    ng_list = np.concatenate((ng_list[0].reshape(-1,1), ng_list[2][:,0:2], ng_list[1].reshape(-1,1)), axis=1)
    coords = pmg_struct.cart_coords
    r0s = coords[ng_list[:,0]]
    offsets = np.matmul(ng_list[:,1:3], pmg_struct.lattice.matrix[:-1][:,:-1])
    r1s = np.append(offsets, np.array([0]*len(offsets)).reshape(-1,1), axis=1) + coords[ng_list[:,-1]]

    hop_func = hop_func_pz(g0=g0, a0=a0, g1=g1, h0=h0, rc=rc, lc=lc, q_dist_scale=q_dist_scale)
    ts = hop_func(r0s, r1s)

    hops = [{} for _ in range(pmg_struct.num_sites)]
    tmp = [hops[ng_list[j][0]].update({tuple(ng_list[j][1:]):ts[j]}) for j in range(len(ts))]
    return hops

def get_neighbor_shells_intralayer(filtered_neigh_list, n_neigh=8, precision=3):
    a, b, c, d = filtered_neigh_list
    c = np.array(c, dtype=int)
    d = np.array([round(i,precision) for i in d])
    dist_shells = np.unique(d)
    r_cut = dist_shells[n_neigh-1]
    dist_shells = dist_shells[:n_neigh]
    ids = np.where(d<=r_cut)[0]
    a, b, c, d = a[ids], b[ids], c[ids], d[ids]
    c = np.append(c, b.reshape(-1,1), axis=1)
    c = [tuple(i) for i in c]
    out = {i:{j:[] for j in dist_shells} for i in a}
    tmp = [out[a[i]][d[i]].append(c[i]) for i in range(len(a))]
    del tmp
    return out, dict(zip(dist_shells, range(len(dist_shells))))

def calc_hopping_wannier_PBC(pmg_struct, layer_inds, layer_vec_to_NN, latt_cont_max, max_dist=5.0, P=0,\
                                ts=[-2.8922, 0.2425, -0.2656, 0.0235, 0.0524, -0.0209, -0.0148, -0.0211], a=2.46):
    """
    add hoppings to graphene multilayer or approximant of graphene multilayer quasicrystal

    intra-layer and inter-layer hopping are splited 
    intra-layer hopping: ts list for N(=len(ts))-nearest neighbor hoppings
    inter-layer hopping: a hopping function with C-C distance cutoff, which depends on the C-C distance and relative orentations

    args:
        pmg_struct: the instance of pymatgen.core.structure.Structure
        layer_inds: the ind range of all layers. return of tBG.TBG30_approximant.structure.Structure._layer_inds() 
        layer_vec_to_NN: the vecs to nearest neighbors of sublattice in each layer [[vec0, vec1]...]
        latt_cont_max: the max of self.a and self.a_top of instance of tBG.TBG30_approximant.structure.Structure
        max_dist: the distance cutoff for inter-layer hopping
        P: the pressure for getting the hopping parameters inter-layer
        ts: the hopping list for intra-layer hopping. default: the 8-nearest neighbor hopping
        a: the graphene lattice constant to reduce the inter-atom distance between layers         
    """

    dist_to = max([5.94, max_dist]) # 5.94 is around the mean between 8th- and 9th-nearest neighbors length
    from functools import reduce
    ng_list = pmg_struct.get_neighbor_list(dist_to)
    center_inds, point_inds, offsets, dists = ng_list
    offsets = np.array(offsets[:,:-1], dtype=int)
    coords = pmg_struct.cart_coords
    nsite = pmg_struct.num_sites
    latt_vec = pmg_struct.lattice.matrix[0:2][:,0:2]
    hop = [{} for _ in range(nsite)]
    nlayer = len(layer_inds)

    #### calc intralayer hopping ####
    for layi in range(nlayer):
        id0, id1 = layer_inds[layi]
        ids = reduce(np.intersect1d, (np.where(center_inds>=id0)[0], np.where(center_inds<=id1)[0],\
                                      np.where(point_inds>=id0)[0], np.where(point_inds<=id1)[0]))
        beg, end, diff, dist = center_inds[ids], point_inds[ids], offsets[ids], dists[ids]
        dist = np.array([round(i,3) for i in dist])
        filtered_ng_list = filter_neig_list([beg, end, diff, dist])
        ng_shells, dist_shells = get_neighbor_shells_intralayer(filtered_ng_list, n_neigh=len(ts), precision=3)
        for i in ng_shells:
            for j in ng_shells[i]:
                for k in ng_shells[i][j]:
                    hop[i][k] = ts[dist_shells[j]]

    #### calc interlayer hopping ####
    ids_dist = np.where(dists<=max_dist)[0]
    lambda0, xi0, k0, lambda3, xi3, x3, lambda6, xi6, x6, k6= \
                                       hop_params_wannier_interlayer(P)
    def get_ids_sublatt(lay_id, site_inds):
        id0, id1 = layer_inds[lay_id]
        nsite = id1 - id0 + 1
        id0_sub0, id1_sub0 = id0, id0+int(nsite/2)-1
        id0_sub1, id1_sub1 = id0+int(nsite/2), id1
        ids_sub0 = np.intersect1d(np.where(site_inds>=id0_sub0)[0], np.where(site_inds<=id1_sub0)[0])
        ids_sub1 = np.intersect1d(np.where(site_inds>=id0_sub1)[0], np.where(site_inds<=id1_sub1)[0])
        return ids_sub0, ids_sub1
        
    for layi in range(nlayer-1):
        layj = layi + 1
        ids_layi = get_ids_sublatt(layi, center_inds)
        ids_layj = get_ids_sublatt(layj, point_inds)
        for i in [0,1]:
            ids_i = ids_layi[i]
            veci_to_NN = layer_vec_to_NN[layi][i]
            for j in [0,1]:
                ids_j = ids_layj[j]
                vecj_to_NN = layer_vec_to_NN[layj][j]
                #ids = np.intersect1d(ids_i, ids_j)
                ids = reduce(np.intersect1d, (ids_i, ids_j, ids_dist))                

                begs = center_inds[ids]
                ends = point_inds[ids]
                diff_vecs = offsets[ids]
                r0s = coords[begs]
                r1s = np.append(np.matmul(diff_vecs, latt_vec), [[0.]]*len(ids), axis=1) + coords[ends]
                ends_with_offset = [tuple(k) for k in np.append(diff_vecs, ends.reshape(-1,1), axis=1)]
                hop_func = hop_func_wannier_interlayer(veci_to_NN, vecj_to_NN, lambda0, xi0, k0, \
                                                                lambda3, xi3, x3, lambda6, xi6, x6, k6, a)
                t_hops = hop_func(r0s, r1s)
                tmp = [hop[begs[ii]].update({ends_with_offset[ii]:t_hops[ii]}) for ii in range(len(begs))]
                del tmp
    return hop

def calc_hopping_wannier_PBC_new(pmg_struct, layer_inds, layer_inds_sublatt, layer_vec_to_NN, max_dist=5.0, P=0,\
                                ts=[-2.8922, 0.2425, -0.2656, 0.0235, 0.0524, -0.0209, -0.0148, -0.0211], a=2.46):
    """
    add hoppings to graphene multilayer or approximant of graphene multilayer quasicrystal

    intra-layer and inter-layer hopping are splited 
    intra-layer hopping: ts list for N(=len(ts))-nearest neighbor hoppings
    inter-layer hopping: a hopping function with C-C distance cutoff, which depends on the C-C distance and relative orentations

    args:
        pmg_struct: the instance of pymatgen.core.structure.Structure
        layer_inds: the ind range of all layers. return of tBG.TBG30_approximant.structure.Structure._layer_inds() 
        layer_vec_to_NN: the vecs to nearest neighbors of sublattice in each layer [[vec0, vec1]...]
        latt_cont_max: the max of self.a and self.a_top of instance of tBG.TBG30_approximant.structure.Structure
        max_dist: the distance cutoff for inter-layer hopping
        P: the pressure for getting the hopping parameters inter-layer
        ts: the hopping list for intra-layer hopping. default: the 8-nearest neighbor hopping
        a: the graphene lattice constant to reduce the inter-atom distance between layers         
    """
    def get_neighbor_shells_intralayer(filtered_neigh_list, n_neigh=8, precision=3):
        a, b, c, d = filtered_neigh_list
        c = np.array(c, dtype=int)
        d = np.array([round(i,precision) for i in d])
        dist_shells = np.unique(d)
        r_cut = dist_shells[n_neigh-1]
        dist_shells = dist_shells[:n_neigh]
        ids = np.where(d<=r_cut)[0]
        a, b, c, d = a[ids], b[ids], c[ids], d[ids]
        c = np.append(c, b.reshape(-1,1), axis=1)
        c = [tuple(i) for i in c]
        out = {i:{j:[] for j in dist_shells} for i in a}
        tmp = [out[a[i]][d[i]].append(c[i]) for i in range(len(a))]
        del tmp
        return out, dict(zip(dist_shells, range(len(dist_shells))))

    dist_to = max([5.94, max_dist]) # 5.94 is around the mean between 8th- and 9th-nearest neighbors length
    from functools import reduce
    ng_list = pmg_struct.get_neighbor_list(dist_to)
    center_inds, point_inds, offsets, dists = ng_list
    offsets = np.array(offsets[:,:-1], dtype=int)
    coords = pmg_struct.cart_coords
    nsite = pmg_struct.num_sites
    latt_vec = pmg_struct.lattice.matrix[0:2][:,0:2]
    hop = [{} for _ in range(nsite)]
    nlayer = len(layer_inds)
    #### calc intralayer hopping ####
    for layi in range(nlayer):
        id0, id1 = layer_inds[layi]
        ids = reduce(np.intersect1d, (np.where(center_inds>=id0)[0], np.where(center_inds<=id1)[0],\
                                      np.where(point_inds>=id0)[0], np.where(point_inds<=id1)[0]))
        beg, end, diff, dist = center_inds[ids], point_inds[ids], offsets[ids], dists[ids]
        dist = np.array([round(i,3) for i in dist])
        filtered_ng_list = filter_neig_list([beg, end, diff, dist])
        ng_shells, dist_shells = get_neighbor_shells_intralayer(filtered_ng_list, n_neigh=len(ts), precision=3)
        # ng_shells = {ind_begin:{NN_dist:[all end_inds ]}}
        # dist_shells = {1.46:0, 2.46:1 ....} pair of dist to which nth nearest neighbor
        #    such as 1.46 is the 0th NN, 2.46 1st NN
        def hop_put_intra(i,j,k):
            hop[i][k] = ts[dist_shells[j]]
        [hop_put_intra(i,j,k) for i in ng_shells for j in ng_shells[i] for k in ng_shells[i][j]]
        #for i in ng_shells:
        #    for j in ng_shells[i]:
        #        for k in ng_shells[i][j]:
        #            hop[i][k] = ts[dist_shells[j]]

    #### calc interlayer hopping ####
    ids_dist = np.where(dists<=max_dist)[0]
    lambda0, xi0, k0, lambda3, xi3, x3, lambda6, xi6, x6, k6= \
                                       hop_params_wannier_interlayer(P)
    def get_ids_sublatt(lay_id, site_inds):
        ids_0, ids_1 = layer_inds_sublatt[lay_id]
        ids_sub0 = np.intersect1d(np.where(site_inds>=ids_0[0])[0], np.where(site_inds<=ids_0[1])[0])
        ids_sub1 = np.intersect1d(np.where(site_inds>=ids_1[0])[0], np.where(site_inds<=ids_1[1])[0])
        return ids_sub0, ids_sub1
        
    for layi in range(nlayer-1):
        layj = layi + 1
        ids_layi = get_ids_sublatt(layi, center_inds)
        ids_layj = get_ids_sublatt(layj, point_inds)
        for i in [0,1]:
            ids_i = ids_layi[i]
            veci_to_NN = layer_vec_to_NN[layi][i]
            for j in [0,1]:
                ids_j = ids_layj[j]
                vecj_to_NN = layer_vec_to_NN[layj][j]
                #begin is layi[i] sublatt, end is layj[j] sublatt with distance less than max_dist
                ids = reduce(np.intersect1d, (ids_i, ids_j, ids_dist))                

                begs = center_inds[ids]
                ends = point_inds[ids]
                diff_vecs = offsets[ids]
                r0s = coords[begs] 
                r1s = np.append(np.matmul(diff_vecs, latt_vec), [[0.]]*len(ids), axis=1) + coords[ends]
                ends_with_offset = [tuple(k) for k in np.append(diff_vecs, ends.reshape(-1,1), axis=1)]
                hop_func = hop_func_wannier_interlayer(veci_to_NN, vecj_to_NN, lambda0, xi0, k0, \
                                                                lambda3, xi3, x3, lambda6, xi6, x6, k6, a)
                t_hops = hop_func(r0s, r1s)
                tmp = [hop[begs[ii]].update({ends_with_offset[ii]:t_hops[ii]}) for ii in range(len(begs))]
                del tmp
    return hop
