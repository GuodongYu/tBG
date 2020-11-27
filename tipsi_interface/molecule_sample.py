import tipsi
from tBG.tipsi_interface.crystal_sample import SparseHopDict

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
