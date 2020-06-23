import numpy as np
import time
import tipsi
import copy
import os

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

        #self.add_conjugates()
        return self.dict

################# tipsi sample ###########
    
def siteset(nsite, size):
    siteset = tipsi.SiteSet()
    for k in range(nsite):
        for i in range(size[0]):
            for j in range(size[1]):
                siteset.add_site((i, j, 0), k)
    return siteset

def lattice(struct):
    sites = np.array(struct.coords)*0.1 # from ang to nm
    lat_vecs = np.append(struct.latt_compatible,[[0.],[0.]], axis=1)*0.1 # from ang to nm
    latt = tipsi.Lattice(lat_vecs, sites)
    return latt

def hopdict(struct, elec_field=0.0):
    nsite = len(struct.coords)
    hop_dict = SparseHopDict(nsite)
    hopping = struct.hopping
    for ral_uc in hopping:
        p,q = [int(w) for w in ral_uc.split('_')]
        for pair in hopping[ral_uc]:
            i,j = [int(w) for w in pair.split('_')]
            hop_dict.set_element((p,q,0), (i,j), hopping[ral_uc][pair])
            hop_dict.set_element((-p,-q,0), (j,i), hopping[ral_uc][pair])
    if elec_field:
        E_on = struct.coords[:,2]*elec_field
        for i in range(nsite):
            hop_dict.set_element((0,0,0), (i,i), E_on[i])
    return hop_dict

def sample(struct, size, rescale=20, nr_processes=1, elec_field=0.0):
    def pbc_func(unit_cell_coords, orbital_ind):
        x, y, z = unit_cell_coords
        return (x%size[0], y%size[1], z), orbital_ind
    nsite = len(struct.coords)
    site_set = siteset(nsite, size)
    latt = lattice(struct)
    if os.path.isfile('system.hdf5'):
        sp = tipsi.Sample(latt, site_set, pbc_func, nr_processes=nr_processes, read_from_file='system.hdf5')
    else:
        sp = tipsi.Sample(latt, site_set, pbc_func, nr_processes=nr_processes)
        hop_dict = hopdict(struct, elec_field)
        sp.add_hop_dict(hop_dict)
        sp.save()
    sp.rescale_H(rescale)
    return sp

