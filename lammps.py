from pymatgen.core.structure import Structure, IMolecule
from itertools import islice
import numpy as np
import os
import subprocess
from shutil import copyfile
import glob
import copy

def lattvec_comm2lmp(latt_vec):
    a,b,c = latt_vec
    xhi = a[0]
    xlo = 0.
    xy = b[0]
    yhi = b[1]
    ylo = 0.
    xz, yz, zhi = c
    zlo = 0
    box = [[xlo, xhi],[ylo, yhi],[zlo, zhi]]
    tilts = [xy, xz, yz]
    return box, tilts

def lattvec_lmp2comm(box, tilts):
    xlo, xhi = box[0]
    ylo, yhi = box[1]
    zlo, zhi = box[2]
    xy, xz, yz = tilts
    return np.array([[xhi-xlo, 0, 0],
                     [xy, yhi-ylo, 0],
                     [xz, yz, zhi-zlo]])

def write_lammps_datafile(box, atom_ids, coords, n_atom, n_atom_type, Masses, atom_type, \
                           atom_style='full', fname='data.struct', **kws):
        """
        Args:
            box: the box 
            coords: cartisian coords    
            n_atom: number of atoms
            n_atom_type: number of the atom type
            Masses: the mass of each atom type
            atom_type: the atomic type of atom in lammps Note: starting from 1
            atom_style: the atomic style in lammps 
            kws: {'tilts':[xy, xz, yz], 'qs': [charges of each atom], 'molecule-tag':[mulecule tag of each atom]}
                 qs: the charges of all atoms
                 molecule-tag: the molecule-tag of all atoms, which is for kc-full potential
                 Note:
                    when atom_style is full, qs and molecule-tag must be given. 
                    if tilts doesn't exist, tilts will not write into the data file, which means a orthogonal box (or molecule)
                      
        """
        xlo, xhi = box[0]
        ylo, yhi = box[1]
        zlo, zhi = box[2]
        with open(fname, 'w') as f:
            f.write(' lammps structure \n\n')
            f.write('    %s atoms\n\n' % n_atom)
            f.write('    %s atom types\n\n' % n_atom_type)
            f.write(' %.8f %.8f xlo xhi\n' % (xlo, xhi))
            f.write(' %.8f %.8f ylo yhi\n' % (ylo, yhi))
            f.write(' %.8f %.8f zlo zhi\n' % (zlo, zhi))
            try:
                xy, xz, yz = kws['tilts']
                f.write(' %.8f %.8f %.8f xy xz yz\n\n' % (xy, xz, yz))
            except: 
                f.write('\n')
            f.write(' Masses\n\n')
            for i in range(len(Masses)):
                f.write('    %s %s\n' % (i+1, Masses[i]))
            f.write('\n')
            f.write(' Atoms\n\n')
            if atom_style == 'atomic':
                for i in range(n_atom):
                    f.write('    %s %s  %.8f %.8f %.8f\n' % (i+1, atom_type[i], \
                                coords[i][0], coords[i][1], coords[i][2]))
            elif atom_style == 'full':
                for i in range(n_atom):
                    f.write('    %s %s %s %s %.8f %.8f %.8f\n' % (atom_ids[i], kws['molecule_tag'][i], atom_type[i], kws['qs'][i],\
                                coords[i][0], coords[i][1], coords[i][2]))

class LammpsStruct:
    def __init__(self):
        self.dtype = {'q': float, 'x': float, 'y':float, 'z':float, 'id':int, 'type':int, 'mol':int, 'mass':float,\
                 'xs': float, 'ys':float, 'zs': float, 'fx':float, 'fy':float, 'fz':float, 'element': str}

    def read_from_file(self, fname):
        """
        read relaxed structure by lammps with custom format 
        """
        with open(fname, 'r') as f:
            natom = int(float([i for i in islice(f, 3, 4)][0]))
        with open(fname, 'r') as f:
            data = [i for i in islice(f, 9+natom)]
            data0 = copy.deepcopy(data)
            while True: 
                data_tmp = [i for i in islice(f, 9+natom)]
                if len(data_tmp):
                    data = data_tmp
                else:
                    break
        box = data[5:8]
        box = np.array([i.split() for i in box], dtype=float)
        xlo_, xhi_, xy = box[0]
        ylo_, yhi_, xz = box[1]
        zlo_, zhi_, yz = box[2]
        xlo = xlo_ - min(0.0,xy,xz,xy+xz)
        xhi = xhi_ - max(0.0,xy,xz,xy+xz)
        ylo = ylo_ - min(0.0,yz)
        yhi = yhi_ - max(0.0,yz)
        zlo = zlo_
        zhi = zhi_
            
        self.box = [[xlo, xhi],[ylo, yhi],[zlo, zhi]]
        self.tilts = [xy, xz, yz]
    
        props = data[8].split()[2:]
        
        sites = data[9:]
        coords = np.array([i.split() for i in sites])
        ids = dict(zip(props, range(len(props))))
        
        atom_ids = np.array(coords[:,ids['id']], dtype=int)
        n_atom = len(atom_ids)
        ids_sort = np.argsort(atom_ids)
        coords = coords[ids_sort]
        self.natom = natom
        self.props = props
        data = {}
        for i in ids:
            data[i] = np.array(coords[:,ids[i]], dtype = self.dtype[i])
        self.data = data

    @property
    def latt_vec(self):
        return lattvec_lmp2comm(self.box, self.tilts)

    @property
    def cart_coords(self):
        coords = np.concatenate([[self.data['x']],[self.data['y']],[self.data['z']]], axis=0).T
        return coords

    def write_datafile(self, atom_style='full'):
        """
        fname: the file saving the relaxation (atom or custom format)
        obtain the relaxed structure from the lammps output file and write it into the
        lammps input file for relax further
        """
    
        atom_type_ = dict(zip(self.data['type'], range(len(self.data['type']))))
        n_atom_type = len(atom_type_)
        Masses = [self.data['mass'][atom_type_[i]] for i in range(1,n_atom_type+1)]
        coords = np.concatenate([[self.data['x']],[self.data['y']],[self.data['z']]], axis=0).T
        write_lammps_datafile(self.box, self.data['id'], coords, self.natom, n_atom_type, Masses, self.data['type'], \
                               atom_style='full', tilts=self.tilts, molecule_tag=self.data['mol'], qs=self.data['q'])    

    def write_POSCAR(self, fname='POSCAR'):
        latt_vec = latt_lmp2comm(self.box, self.tilts)
        coords = np.concatenate([[self.data['x']],[self.data['y']],[self.data['z']]], axis=0).T
        z_coords = coords[:,-1]
        min_z = np.min(z_coords)
        coords[:,-1] = coords[:,-1] - min_z + 1
        coord_str = np.array(coords, dtype=str)
        coord_str = '\n'.join([' '.join(i) for i in coord_str])
        with open(fname, 'w') as f:
            f.write('poscar\n')
            f.write('   1.0\n')
            f.write(' %s %s %s\n' % (latt_vec[0][0], latt_vec[0][1], latt_vec[0][2]))
            f.write(' %s %s %s\n' % (latt_vec[1][0], latt_vec[1][1], latt_vec[1][2]))
            f.write(' %s %s %s\n' % (latt_vec[2][0], latt_vec[2][1], latt_vec[2][2]))
            f.write(' C\n')
            f.write(' %s\n' % self.natom)
            f.write('Cartesian\n')
            f.write(coord_str)

    def to_POSCAR_molecule(self, fname='POSCAR'):
        coords = np.concatenate([[self.data['x']],[self.data['y']],[self.data['z']]], axis=0).T
        atom_types = self.data['type']
        n_type = len(set(atom_types))
        elems = []
        num_eles = []
        for i in range(n_type):
            ele = 'C' if i in [0, 1] else 'H'
            elems.append(ele)
            num = len(np.where(atom_types==i+1)[0])
            num_eles.append(num)
        x_min, y_min, z_min = np.min(coords, axis=0)
        x_max, y_max, z_max = np.max(coords, axis=0)
        latt_vec = np.array([[x_max-x_min+6, 0, 0], [0, y_max-y_min+6, 0], [0, 0, z_max-z_min+6]])
        coords = coords - [x_min, y_min, z_min] +[3, 3, 3]
        coord_str = np.array(coords, dtype=str)
        coord_str = '\n'.join([' '.join(i) for i in coord_str])
        with open(fname, 'w') as f:
            f.write('poscar\n')
            f.write('   1.0\n')
            f.write(' %s %s %s\n' % (latt_vec[0][0], latt_vec[0][1], latt_vec[0][2]))
            f.write(' %s %s %s\n' % (latt_vec[1][0], latt_vec[1][1], latt_vec[1][2]))
            f.write(' %s %s %s\n' % (latt_vec[2][0], latt_vec[2][1], latt_vec[2][2]))
            f.write(' %s\n' % ' '.join(elems))
            f.write(' %s\n' % ' '.join([str(i) for i in num_eles]))
            f.write('Cartesian\n')
            f.write(coord_str)


class Log:
    def __init__(self, log_file):
        self._log_file = log_file
        self._parse() 

    def _parse(self):
        """
        parse the total energy from lammps log file
        """
        with open(self._log_file,'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if 'reading atoms' in lines[i]:
                    self.n_atom = int(lines[i+1].split()[0]) 
                elif 'Energy initial, next-to-last, final' in lines[i]:
                    self.e_tot = float(lines[i+1].split()[-1])

