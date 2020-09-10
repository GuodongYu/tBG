from tBG.quantum_dot import QuantumDot, QuantumDotQC, QuantumDotAB
from monty.json import jsanitize
import numpy as np
import scipy.io
import glob
import os
import json
import re
import copy
import time
import multiprocessing as mp

class PathName:
    """
    functions for path name
    """
    title = 'quantum_dot'

    @staticmethod
    def _get_shape(n):
        if n==3:
            return 'triangle'
        elif n==6:
            return 'hexagon'
        elif n==12:
            return 'decagon'

    @classmethod
    def tBG_regular(cls, n, overlap, orient):
        shape = PathName._get_shape(n)
        typ = '%s_%s' % (overlap, orient)
        return os.path.join(cls.title, 'tBG', shape, typ)

    @classmethod
    def tBG_rectangle(cls):
        return os.path.join(cls.title, 'tBG', 'rectangle', 'side_center')

    @classmethod
    def QC_regular(cls, n):
        shape = PathName._get_shape(n)
        return os.path.join(cls.title, 'QC', shape, 'hole')

    @classmethod
    def AB_regular(cls, n, overlap):
        shape = PathName._get_shape(n)
        return os.path.join(cls.title, 'AB', shape, overlap)

def is_done(fold):
    return os.path.isfile(fold+'/data.npz') and os.path.isfile(fold+'/POSCAR')

class ScanFolder:
    """
    functions to derive values (including Rs, values, WHs) by scanning folders
    """
    @staticmethod
    def _collect_values(fold):
        fs_abs = glob.glob(fold+'/*')
        fs = [os.path.split(i)[-1] for i in fs_abs]
        label = fs[0].split('_')[0]
        if 'R' in label or 'theta' in label:
            vs = np.array([re.findall('\d*\.?\d+',i)[0] for i in fs], dtype=np.float64)
        elif label[0] == 'W':
            vs = np.array([re.findall('\d*\.?\d+',i)[0:2] for i in fs], dtype=np.float64)
            vs = [tuple(i) for i in vs]
        return vs, fs_abs
    
    @staticmethod
    def Rs_thetas(fold):
        """
        for tBG type
        """
        Rs, Rs_fold = ScanFolder._collect_values(fold)
        R_thetas = {}
        for i in range(len(Rs)):
            R = Rs[i]
            R_fold = Rs_fold[i]
            thetas, thetas_fold = ScanFolder._collect_values(R_fold)
            R_thetas[R] = sorted(np.array(thetas))
        return R_thetas
            
    @staticmethod
    def Rs(fold):
        """
        for quasicrystal and AB type
        """
        Rs, Rs_fold = ScanFolder._collect_values(fold)
        return sorted(Rs)

class UnfinishedJobs:
    @staticmethod
    def Rs_thetas(finished, Rs, thetas):
        Rs = np.array(Rs, dtype=np.float64)
        if Rs.ndim==2:
            Rs = [tuple(i) for i in Rs]
        thetas = np.array(thetas, dtype=np.float64)
        undo = {}
        for R in Rs:
            if R not in finished:
                undo[R] = thetas
            else:
                undo[R] =  np.array(list( set(thetas) - (set(thetas)&set(finished[R])) ))
        print(undo)
        undo = {k:v for k,v in undo.items() if len(v)}
        return undo

    @staticmethod
    def Rs(finished, Rs):
        return np.array(list( set(Rs) - (set(Rs)&set(finished) ) ))


class Geometry:
    """
    functions for geometry, output in units of nm
    """

    a = 0.246 # nm
    s0 = 0.5*np.sqrt(3)/2*a*a
    @classmethod
    def side_length(cls, n, R):
        """
        n: the number of sides
        R: the radius (orgin to vertex in units of a=2.46 angstrom)
        """
        R_nm = R*cls.a
        if n == 3:
            return 4/np.sqrt(3) * R_nm
        elif n==6:
            return R_nm
        elif n==12:
            return 2*R_nm*np.cos(5*np.pi/12)

    @classmethod
    def square_regular(cls, n, R):
        R_nm = R*cls.a
        if n==3:
            return 0.5*(3/2*R_nm)*(np.sqrt(3)*R_nm)
        elif n==6:
            return 1.5*np.sqrt(3)*R_nm**2
        elif n==12:
            lside = Geometry.side_length(12, R)
            return 3*(2+np.sqrt(3))*lside**2

    @classmethod    
    def square_rectangle(cls, W, H):
        w = W*cls.a
        h = H*cls.a
        return w*h

    @classmethod
    def natom2R(cls, n, natom):
        """
        evaluate R (in units of a) from natom, NOTE: not exact but nearby
        """
        if n==3:
            up = 2*natom*cls.s0
            down = 3*np.sqrt(3)
            return np.sqrt(up/down)/cls.a
        elif n==6:
            up = natom*cls.s0
            down = 3*np.sqrt(3)
            return np.sqrt(up/down)/cls.a
        elif n==12:
            c = 1/(2*np.cos(5*np.pi/12))
            up = natom*cls.s0
            down = 6*(2+np.sqrt(3))
            return c*np.sqrt(up/down)/cls.a

    @classmethod
    def natom2WH(cls, ratio_W2H, natom):
        H = np.sqrt(natom*cls.s0/(2*ratio_W2H))/cls.a
        return np.array([ratio_W2H*H, H]).T
        
class OutputQD:
    @staticmethod
    def _bilayer_output(qd, fold, split=False):
        pmg_st = qd.pmg_struct()
        pmg_st.to('poscar',os.path.join(fold, 'POSCAR'))
        del pmg_st
        t0 =time.time()
        qd.add_hopping_pz(max_dist=5.0, g0=2.8, a0=1.42, g1=0.48, h0=3.35, rc=6.14, lc=0.265, q_dist_scale=2.218, nr_processes=1)
        t1 = time.time()
        print('time for adding hopping: %s s' % (t1-t0))
        if split:
            H = qd.get_Hamiltonian()
            np.savez_compressed(os.path.join(fold, 'H_mat'), H_mat=H)
            del H
            J = qd.get_current_mat()
            np.savez_compressed(os.path.join(fold, 'J_mat'), J_mat=J)
            del J
            ops = qd.symmetry_operations()
            ops_np = {i:ops[i] for i in ops}
            del ops
            np.savez_compressed(os.path.join(fold, 'ops'), **ops_np)
        else:
            t0 = time.time()
            H = qd.get_Hamiltonian()
            t1 = time.time()
            print('time for getting H %s s' % (t1-t0))
            J = qd.get_current_mat()
            t2 = time.time()
            print('time for getting J %s s' % (t2-t1))
            
            ops = qd.symmetry_operations()
            ops_np = {i:ops[i] for i in ops}
            ops_np['H_mat'] = H
            ops_np['J_mat'] = J
            t3 = time.time()
            print('time for getting ops %s s' % (t3-t2))
    
            np.savez_compressed(os.path.join(fold, 'data'), **ops_np)
            t4 = time.time()
            print('time for output %s s' % (t4-t3))

    @staticmethod
    def tBG_regular(n, R, theta, overlap, orient, rm_single_bond=True, fold='.', new_cut_style=False):
        qd = QuantumDot()
        qd.regular_polygon(n, R, theta, overlap=overlap, orient=orient, rm_single_bond=rm_single_bond, new_cut_style=new_cut_style)
        l_side = Geometry.side_length(n, R)
        S = Geometry.square_regular(n, R) 
        fold = fold+'_'+qd.point_group+'_nsite%i' % len(qd.coords)+'_lside%.2fnm' % l_side +'_S%.2fnm2' % S
        if not os.path.isdir(fold):
            os.makedirs(fold)
        OutputQD._bilayer_output(qd, fold)

    @staticmethod
    def tBG_rectangle(w, h, theta, rm_single_bond=True, fold='.', new_cut_style=False):
        qd = QuantumDot()
        qd.rectangle(w, h, theta, overlap='side_center', rm_single_bond=rm_single_bond, new_cut_style=new_cut_style)
        pg = qd.point_group
        fold = fold + '_' + pg + '_nsites%i' % len(qd.coords)
        if not os.path.isdir(fold):
            os.makedirs(fold)
        OutputQD._bilayer_output(qd, fold)

    @staticmethod
    def QC_regular(n, R, rm_single_bond=True, fold='.'):
        t0 = time.time()
        qd = QuantumDotQC()
        qd.regular_polygon(n, R, rm_single_bond=rm_single_bond)
        l_side = Geometry.side_length(n, R) 
        S = Geometry.square_regular(n, R)
        fold = fold+'_'+qd.point_group+'_nsite%i' % len(qd.coords)+'_lside%.2fnm' % l_side +"_S%.2fnm2" % S
        if not os.path.isdir(fold):
            os.makedirs(fold)
        t1 = time.time()
        print('time for preparing %s s' % (t1-t0))
        OutputQD._bilayer_output(qd, fold)

    @staticmethod
    def AB_regular(n, R, overlap, rm_single_bond=True, fold='.'):
        qd = QuantumDotAB()
        qd.regular_polygon(n, R, overlap=overlap, rm_single_bond=rm_single_bond)
        l_side = Geometry.side_length(n, R) 
        S = Geometry.square_regular(n, R)
        fold = fold+'_'+qd.point_group+'_nsite%i' % len(qd.coords)+'_lside%.2fnm' % l_side +"_S%.2fnm2" % S
        if not os.path.isdir(fold):
            os.makedirs(fold)
        OutputQD._bilayer_output(qd, fold)

class JobStatus:
    
    def __init__(self, prefix='.'):
        self.prefix = prefix
        self.root = os.path.join(prefix, PathName.title)
        self.jobs_status_file = os.path.join(self.root, 'finished_jobs.json')

    def _scan_folder_tree(self, check=False):
        """
        scan the whole folder tree in self.root including ignoring whether finished
        """
        out = {}
        folds = [i[0] for i in os.walk(self.root) if i[-2] and i[-2][0][0] in ['R','W']]
        grps_name = [i.replace(self.root, '')[1:] for i in folds]
        for i in range(len(grps_name)):
            name = grps_name[i]
            if 'tBG' in name:
                scan_func = ScanFolder.Rs_thetas
            else:
                scan_func = ScanFolder.Rs
            out[name] = scan_func(folds[i])
        return out

    def _read_finished_jobs(self):
        """
        read finished jobs saved in self.jobs_status_file
        """
        if os.path.getsize(self.jobs_status_file):
            finished = json.load(open(self.jobs_status_file))
        else:
            finished = {}
        out = {}
        for grp in finished:
            if grp == 'tBG/rectangle/side_center':
                out[grp] = {}
                for WH_str in finished[grp]:
                    WH_tuple = tuple(np.array(WH_str[1:-1].split(','), dtype=np.float64))
                    out[grp][WH_tuple] = finished[grp][WH_str]
            elif 'tBG' in grp:
                out[grp] = {float(i):finished[grp][i] for i in finished[grp]}
            else:
                out[grp] = finished[grp]
        return out

    def _diff_scanned_and_saved(self):
        """
        get the jobs scanned but not saved in self.jobs_status_file 
        scanned jobs can include the unfinished, such as the case of a huge size system without data.npz output
        """
        out = {}
        scanned = self._scan_folder_tree()
        finished = self._read_finished_jobs()
        for grp in scanned:
            if grp not in finished:
                out[grp] = scanned[grp]
            else:
                if 'tBG' in grp:
                    out[grp] = {}
                    for R in scanned[grp]:
                        if R not in finished[grp]:
                            out[grp][R] = scanned[grp][R]
                        else:
                            out[grp][R] = list( set(scanned[grp][R]) - \
                                                ( set(scanned[grp][R]) & set(finished[grp][R]) ) )
                else:
                    out[grp] = list( set(scanned[grp]) - (set(scanned[grp]) & set(finished[grp])) )

        out = {k: v for k, v in out.items() if v} ## remove items with empty values 
        out_tBG = {k: v for k, v in out.items() if 'tBG' in k} ## tBG part
        out_other = {k: v for k, v in out.items() if 'tBG' not in k} ## non tBG part

        out_tBG = {k: {i:j for i,j in v.items() if j} for k, v in out.items() if 'tBG' in k} ## remove items with empty 
        out = {**out_tBG, **out_other}
        out = {k: v for k, v in out.items() if v} 
        
        return out

    def _diff_finished_and_saved(self):
        """
        get finished output but not saved in self.jobs_status_file
        Two steps are taken:
        1: get the diff between saved and scaned, namely the output of self._diff_scanned_and_saved
        2: check whether they are finished among the outpu by step 1, and output just the finished 
        """
        def get_folders(Rs, folder):
            ndim = np.array(Rs).ndim
            fs_all = glob.glob(folder+'/*')
            Rs = np.array(Rs, dtype=np.float64)
            if ndim == 1:
                Rs_str = np.array(list(map(str, Rs)))
                Rs_all = [os.path.split(f)[-1].split('_')[0] for f in fs_all]
                Rs_all = np.array([re.findall('\d*\.?\d+',i)[0] for i in Rs_all], dtype=np.float64)
                Rs_all_str = np.array(list(map(str, Rs_all)))
                
            elif ndim == 2:
                Rs_str = np.array(list(map(lambda x:'%s~%s' % (x[0],x[1]), Rs)))
                Rs_all = np.array([re.findall('\d*\.?\d+', os.path.split(f)[-1]) for f in fs_all], dtype=np.float64)
                Rs_all_str = np.array(list(map(lambda x:'%s~%s' % (x[0],x[1]), Rs_all)))
            sorter = np.argsort(Rs_all_str)
            indices = sorter[np.searchsorted(Rs_all_str, Rs_str, sorter=sorter)]
            return np.array(fs_all)[indices]

        diff = self._diff_scanned_and_saved()
        out = {}
        for grp in diff:
            if 'tBG' in grp:
                out[grp] = {}
                Rs = list(diff[grp].keys())
                fs_R = get_folders(Rs,os.path.join(self.root, grp))
                for i in range(len(diff[grp])):
                    R = Rs[i]
                    f_R = fs_R[i]
                    thetas = diff[grp][R]
                    fs_check = get_folders(thetas, f_R)
                    indices_done = np.array([is_done(f) for f in fs_check])
                    out[grp][R] = np.array(thetas)[indices_done]
            else:
                Rs = diff[grp]
                fs_check = get_folders(Rs,os.path.join(self.root, grp))
                indices_done = np.array([is_done(f) for f in fs_check])
                out[grp] = np.array(Rs)[indices_done]
        return out

    def _merge_saved_and_screened_diff(self, fast=True):
        """
        return a job dictionary which merges the saved and finished-but-not-saved jobs
        """
        if fast:
            finished = self._read_finished_jobs()
        else:
            finished = {}
        screened = self._diff_finished_and_saved()
        for grp in screened:
            if grp not in finished:
                finished[grp] = screened[grp]
            else:
                if 'tBG' in grp:
                    for R in screened[grp]:
                        if R not in finished[grp]:
                            finished[grp][R] = screened[grp][R]
                        else:
                            finished[grp][R] = np.append(finished[grp][R], screened[grp][R])
                else:
                    finished[grp] = np.append(finished[grp], screened[grp])             
        return finished

    def _del_unfinished_and_scanned(self):
        self.update()
        
               
    def update(self, fast=True):
        if not os.path.isdir(self.root):
            os.mkdir(self.root)
            with open(self.jobs_status_file, 'w') as f:
                json.dump({}, f)
        finished_all = self._merge_saved_and_screened_diff(fast=fast)
        with open(self.jobs_status_file, 'w') as f:
            json.dump(jsanitize(finished_all), f) 

def div_groups(ll, n_group):
    """
    divide a list into n_group groups
    
    example:
    if ll=[1,2,3,4,5,6,7,8] n_group= 3
    return [[1,4,7],[2,5,8],[3,6]]
    """
    ll = np.array(ll, dtype=np.float64)
    if ll.ndim == 1:
        ll = sorted(ll)
    elif ll.ndim == 2:
        if ll.shape[-1] == 2:
            ll = sorted(ll, key=lambda x:x[0])
        elif ll.shape[-1] == 3:
            ll = sorted(ll, key=lambda x:x[0]*x[1])
    ll = np.array(ll)
    if len(ll)<n_group:
        return [[i] for i in ll ] + [ [] ]*(n_group-len(ll)) 
    size = int(len(ll)/n_group)
    div = [[ll[n_group*i+j] for i in range(size)] for j in range(n_group)]
    n_left = len(ll) - size*n_group
    [div[i].append(ll[-n_left:][i]) for i in range(n_left)]
    return div

def convert_undo(undo):
    """
    convert job dictionary into list
    such as {1.2:[1,2], 2.7:[8,9]} -> [[1.2, 1], [1.2, 2], [2.7, 8], [2.7, 9]]
    """
    if not undo:
        return []
    one_key = np.array(list(undo.keys())[0])
    shape = np.array([one_key]).shape
    if shape == (1,2):
        return np.concatenate([[[k[0],k[1],v[i]] for i in range(len(v))] for k,v in undo.items()])
    elif shape == (1,):
        return np.concatenate([[[k,v[i]] for i in range(len(v))] for k,v in undo.items()])


class MassOutput:
    """
    mass output quantum dots
    """
    def __init__(self, prefix='.', update_job_info=True):
        self.prefix = prefix
        self.root = os.path.join(prefix, PathName.title)
        if update_job_info:
            JobStatus(prefix).update()

    def _read_finished_jobs(self, grp):
        finished = JobStatus(prefix=self.prefix)._read_finished_jobs()
        if grp in finished:
            return finished[grp]
        else:
            return {}

    def tBG_regular(self, n, Rs, thetas, overlap, orient, rm_single_bond=True, nr_processes=1, new_cut_style=False):
        grp = '/'.join(PathName.tBG_regular(n, overlap, orient).split('/')[1:])
        finished = self._read_finished_jobs(grp)
        path_title = PathName.tBG_regular(n, overlap, orient)
        path = os.path.join(self.prefix, path_title) ## get location 

        def output(Rs_thetas):
            for R, theta in Rs_thetas:
                where = os.path.join(path, 'R%s' % R, 'theta%s' % theta)
                OutputQD.tBG_regular(n, R, theta, overlap, orient, rm_single_bond=rm_single_bond, fold=where, new_cut_style=new_cut_style)       

        Rs = np.round(Rs, 2)
        thetas = np.round(thetas, 2)
        undo = UnfinishedJobs.Rs_thetas(finished, Rs, thetas)
        print('Running %s' % grp, undo)
        Rs_thetas = convert_undo(undo)
        if nr_processes == 1:
            output(Rs_thetas)
        else:
            print('parent pid: %s' % os.getpid())
            div = div_groups(Rs_thetas, nr_processes)
            processes = [None for i in range(nr_processes)]
            for i, tags in enumerate(div):
                processes[i] = mp.Process(target=output, args=(tags,))
                processes[i].start()
            for p in processes:
                p.join()

    def tBG_rectangle(self, sizes, thetas, rm_single_bond=True, nr_processes=1, new_cut_style=False):
        grp = '/'.join(PathName.tBG_rectangle().split('/')[1:])
        finished = self._read_finished_jobs(grp)

        sizes = np.array([np.round(i, 2) for i in sizes])
        thetas = np.round(thetas, 2)
        path_title = PathName.tBG_rectangle()
        path = os.path.join(self.prefix, path_title) ## get location

        def output(WHs_thetas): 
            for W,H,theta in WHs_thetas:
                where = os.path.join(path, 'W%s_H%s' % (W, H), 'theta%s' % theta)
                OutputQD.tBG_rectangle(W, H, theta, rm_single_bond=rm_single_bond, fold=where, new_cut_style=new_cut_style)
       
        undo = UnfinishedJobs.Rs_thetas(finished, sizes, thetas) 
        print('Running %s' % grp, undo)
        WHs_thetas = convert_undo(undo)
        if nr_processes==1:
            output(WHs_thetas)
        else:
            div = div_groups(WHs_thetas, nr_processes)
            processes = [None for i in range(nr_processes)]
            print('Groups for parallel: ', div)
            for i, tags in enumerate(div):
                processes[i] = mp.Process(target=output, args=(tags,))
                processes[i].start()
            for p in processes:
                p.join()
    
    def QC_regular(self, n, Rs, rm_single_bond=True, nr_processes=1): 
        grp = '/'.join(PathName.QC_regular(n).split('/')[1:])
        finished = self._read_finished_jobs(grp)

        Rs = np.round(Rs, 2)
        path_title = PathName.QC_regular(n)
        path = os.path.join(self.prefix, path_title)

        def output(Rs):
            for R in Rs:
                where = os.path.join(path, 'R%s' % R)
                OutputQD.QC_regular(n, R, rm_single_bond=rm_single_bond, fold=where)
        
        undo = UnfinishedJobs.Rs(finished, Rs)
        print('Running %s' % grp, undo)
        if nr_processes == 1:
            output(Rs)
        else:
            div = div_groups(Rs, nr_processes)
            processes = [None for i in range(nr_processes)]
            print('Groups for parallel: ', div)
            for i, tags in enumerate(div):
                processes[i] = mp.Process(target=output, args=(tags,))
                processes[i].start()
            for p in processes:
                p.join()

    def AB_regular(self, n, Rs, overlap, rm_single_bond=True, nr_processes=1): 
        grp = '/'.join(PathName.AB_regular(n, overlap).split('/')[1:])
        finished = self._read_finished_jobs(grp)

        Rs = np.round(Rs, 2)
        path_title = PathName.AB_regular(n, overlap)
        path = os.path.join(self.prefix, path_title)

        def output(Rs):
            for R in Rs:
                where = os.path.join(path, 'R%s' % R)
                OutputQD.AB_regular(n, R, overlap=overlap, rm_single_bond=rm_single_bond, fold=where)
        
        undo = UnfinishedJobs.Rs(finished, Rs)
        print('Running %s' % grp, undo)
        if nr_processes == 1:
            output(Rs)
        else:
            div = div_groups(Rs, nr_processes)
            print('Groups for parallel: ', div)
            processes = [None for i in range(nr_processes)]
            for i, tags in enumerate(div):
                processes[i] = mp.Process(target=output, args=(tags,))
                processes[i].start()
            for p in processes:
                p.join()

def main():
    a = 1
    b = a/np.sqrt(3)
    thetas = np.linspace(0.5, 30, 60)
    Rs = np.linspace(10/2.46, 70/2.46, 31)
    rm_single_bond = True
    thetas = [0, 10, 30]   

    mass_output = MassOutput()
    mass_output.QC_regular(12, [6],  rm_single_bond=True)
    mass_output.QC_regular(6, [6],  rm_single_bond=True)
    mass_output.QC_regular(3, [7.5],  rm_single_bond=True)
    mass_output.tBG_rectangle([[14.5*b, 8.1],[20*b, 9.1]], [0, 10, 30],  rm_single_bond=True)
    mass_output.tBG_regular(6, [7.1*b], [0,10, 30], 'hole', 'armchair', rm_single_bond=rm_single_bond)    
    mass_output.tBG_regular(6, [4], [0,10,30], 'hole', 'zigzag', rm_single_bond=rm_single_bond)    
    mass_output.tBG_regular(3, [5], [0,10,30], 'hole', 'zigzag', rm_single_bond=rm_single_bond)    
    mass_output.tBG_regular(3, [9.0*b], [0, 10, 30], 'atom1', 'armchair', rm_single_bond=rm_single_bond)    
            
if __name__ == '__main__':
    MassOutput.setup_prefix('.')
    MassOutput.tBG_rectangle([[21, 10.5],[15, 9.1]], [5, 20, 23],  rm_single_bond=True)
