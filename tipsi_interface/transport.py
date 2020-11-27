import tipsi
from tipsi import Sample
from tipsi import Lattice
from tipsi import SiteSet
import numpy as np
import json
from monty.json import jsanitize
from scipy import interpolate
import copy


class TipsiCalc(object):
    def __init__(self, tipsi_sample):
        self.sample = tipsi_sample
        self.norb = len(self.sample.site_x)
        
    def config(self, B, nr, nt=2048, mu=0., Temp=300, seed=1337, quasi_enes=[], **dckb):
        self.sample.set_magnetic_field(B)
        config = tipsi.Config(self.sample)
        config.generic['correct_spin'] = True
        config.generic['nr_random_samples']=nr
        config.generic['nr_time_steps'] = nt
        config.generic['beta'] = 11604.505/Temp
        config.generic['mu'] = mu
        config.generic['cuda'] = False
        config.generic['seed'] = seed
        if dckb:
            for i in dckb:
                config.dckb[i] = dckb[i]
        if len(quasi_enes):
            config.quasi_eigenstates['energies'] = quasi_enes
        config.save()
        return config

    @staticmethod
    def _DOS_norm0(dos, norb, nelec, prec=1.e-3):
        energies = np.array(dos['energies'])
        dos = np.array(dos['DOS'])
        idos = []
        de = energies[1]-energies[0]
        ne = len(energies)
        summ = 0.0
        energies_mid = []
        for i in range(ne-1):
            e_mid = (energies[i]+energies[i+1])/2.0
            energies_mid.append(e_mid)
            summ = summ + (dos[i]+dos[i+1])/2.0
            idos.append(summ)
        summ = summ * de
        C = norb/summ 
        idos = np.array(idos)*de*C*2.0
        #dos = dos * C

        if de>prec:
            tck = interpolate.splrep(energies_mid, idos)
            erange = energies_mid[-1] - energies_mid[0]
            energies_ = np.linspace(energies_mid[0], energies_mid[-1], int(erange/prec))
            idos_ = interpolate.splev(energies_, tck)
        else:
            idos_ = idos
            energies_ = energies
        idos_relative = np.abs(idos_ - nelec)
        idos_relative = idos_relative.tolist()
        ind = idos_relative.index(min(idos_relative))
        return {'energies':energies,'DOS':dos, 'efermi':energies_[ind]}

    @staticmethod
    def _DOS_norm(dos, norb, nelec):
        energies = np.array(dos['energies'])
        dos = np.array(dos['DOS'])
        idos = []
        de = energies[1]-energies[0]
        ne = len(energies)
        summ = 0.0
        for i in range(ne-1):
            summ = summ + dos[i]
            if i==0:
                idos.append(0.)
            else:
                idos.append(summ)
        summ = summ * de
        C = norb/summ 
        idos = np.array(idos)*de*C*2.0
        dos = dos * C

        idos_relative = np.abs(idos - nelec)
        idos_relative = idos_relative.tolist()
        ind = idos_relative.index(min(idos_relative))
        return {'energies':energies,'DOS':dos, 'efermi':energies[ind]}

    def DOS_calc(self, B, nr, nt=2048, save_to='DOS.json'):
        config = self.config(B, nr, nt)
        corr_DOS = tipsi.corr_DOS(self.sample, config)
        energies_DOS, DOS = tipsi.analyze_corr_DOS(config, corr_DOS)
        dos = {}
        dos['energies'] = energies_DOS
        dos['DOS'] = DOS
        dos = self._DOS_norm0(dos, self.norb, self.norb)
        if save_to.split('.')[-1] != 'json':
            raise ValueError('Error: only json file are accept!!')
        with open(save_to,'w') as f:
            f.write(json.dumps(jsanitize(dos)))

    def LDOS_calc(self, B, nr, nt=1024, ldos_sites = [], ldos_separated=True, save_to = 'LDOS.json'):
        """
        Args:
            T: the magnetic field
            nr: the number of random states
            nt: the steps of time evolution
            ldos_sites: the sites of the samples, on which the local density of states will be calculated
            save_to: a file name with json extension, and the ldos data will be saved to it. 
                    please note only the json file will be used here.
        """


        if ldos_separated:
            for i in ldos_sites:
                config = self.config(B, nr, nt, save=False)  
                LDOS_sites = []
                LDOS_sites.append(self.sample.tag_to_index[(0,0,0,i)])
                config.LDOS['site_indices'] = LDOS_sites
                config.save()
                corr_LDOS = tipsi.corr_LDOS(self.sample, config)
                energies_LDOS, LDOS = tipsi.analyze_corr_LDOS(config, corr_LDOS)
                ldos = {}
                ldos['energies'] = energies_LDOS
                ldos['LDOS'] = LDOS
                if save_to.split('.')[-1] != 'json':
                    raise ValueError('Error: only json file are accept!!')
                with open('site%s_' % i +save_to,'w') as f:
                    f.write(json.dumps(jsanitize(ldos)))
        else:
            config = self.config(B, nr, nt, save=False)  
            LDOS_sites = []
            for i in ldos_sites:
                LDOS_sites.append(self.sample.tag_to_index[(0,0,0,i)])
            config.LDOS['site_indices'] = LDOS_sites
            config.save()
            corr_LDOS = tipsi.corr_LDOS(self.sample, config)
            energies_LDOS, LDOS = tipsi.analyze_corr_LDOS(config, corr_LDOS)
            ldos = {}
            ldos['energies'] = energies_LDOS
            ldos['LDOS'] = LDOS
            if save_to.split('.')[-1] != 'json':
                raise ValueError('Error: only json file are accept!!')
            with open(save_to,'w') as f:
                f.write(json.dumps(jsanitize(ldos)))

    def AC_calc(self, B, nr, mu=0., nt=1024, seed=1337,save_to=''):
        config = self.config(B, nr, nt=nt, seed=seed, mu=mu)
        corr_AC = tipsi.corr_AC(self.sample, config)
        omegas, AC= tipsi.analysis.analyze_corr_AC(config, corr_AC)
        ac = {}
        ac['omegas'] = omegas
        ac['AC'] = AC
        if save_to:
            with open(save_to,'w') as f:
                f.write(json.dumps(jsanitize(ac)))
        return ac

    def DC_calc(self, B, nr, nt=1024, ispin=1, save_to='DC.json'):
        config = self.config(B, nr, nt)
        corr_DOS, corr_DC = tipsi.correlation.corr_DC(self.sample, config)
        energies, DC = tipsi.analysis.analyze_corr_DC(config, corr_DOS, corr_DC)
        dc = {}
        dc['energies'] = energies
        if ispin==1:
            dc['DC'] = DC[0]
        with open(save_to,'w') as f:
            f.write(json.dumps(jsanitize(dc)))
    
    def Hall_cond_calc(self, B, Temp=0.0001, n_kernel=8000, ne_integral=8000, direction=2, 
                        energy_range=[-3.3,2.6], ef=0.905, nr=1, nt=8192, save_to='Hall_cond.json'):
        energy_range = np.array(energy_range) + ef
        dckb = {}
        dckb['n_kernel'] = n_kernel
        dckb['direction']  = direction
        dckb['ne_integral']  = ne_integral
        dckb['energies'] = np.arange(energy_range[0],energy_range[1],0.001)
        config = self.config(B, nr, nt, Temp, **dckb)
        energies, mu, cond = tipsi.get_dckb(self.sample, config)
        out = {}
        out['energies'] = energies
        out['cond'] = cond
        with open(save_to,'w') as f:
            f.write(json.dumps(jsanitize(out)))

    def quasi_eigenstate(self, B=0., nr=1, nt=1024, energies=[-3.142, -2.28431, -2.06515, -1.9465, -1.81467, -1.46036, 1.87015, 2.48977]):
        energies = np.array(energies) + 0.905
        config = self.config(B, nr, nt=nt, quasi_enes=energies)
        quasi_eigenstates = tipsi.correlation.quasi_eigenstates(self.sample, config)
        return quasi_eigenstates
         
        for i in range(len(energies)):
            np.save('quasi_eigenstate%s' % i, quasi_eigenstates[i])
            fname = 'quasi_eigenstate%s.pdf' % i
            tipsi.output.plot_wf(quasi_eigenstates[i,:], sample, fname, site_size=10, colorbar=True)


