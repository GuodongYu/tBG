import tipsi
import os
import copy
import numpy as np

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



def siteset(nsite, size):
    siteset = tipsi.SiteSet()
    for k in range(nsite):
        for i in range(size[0]):
            for j in range(size[1]):
                siteset.add_site((i, j, 0), k)
    return siteset

def lattice(struct):
    sites = struct.coords*0.1 # from ang to nm
    lat_vecs = struct.latt_vec*0.1 # from ang to nm
    latt = tipsi.Lattice(lat_vecs, sites)
    return latt

def sample(struct, size, rescale=20, nr_processes=1, elec_field=0.0):
    def pbc_func(unit_cell_coords, orbital_ind):
        x, y, z = unit_cell_coords
        return (x%size[0], y%size[1], z), orbital_ind
    nsite = struct.nsite
    site_set = siteset(nsite, size)
    latt = lattice(struct)
    if os.path.isfile('sample.hdf5'):
        print('Reading sample ...')
        sp = tipsi.Sample(latt, site_set, pbc_func, nr_processes=nr_processes, read_from_file='sample.hdf5')
        print('Read done')
    else:
        print('Constructing sample from scratch ...')
        sp = tipsi.Sample(latt, site_set, pbc_func, nr_processes=nr_processes)
        hop_dict = SparseHopDict(nsite)
        hop_dict.dict = struct.hoppings_2to3()
        sp.add_hop_dict(hop_dict)
        sp.save()
    sp.rescale_H(rescale)
    return sp
