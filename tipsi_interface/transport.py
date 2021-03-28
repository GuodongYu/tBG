import tipsi
from tipsi import Sample
from tipsi import Lattice
from tipsi import SiteSet
import numpy as np
import json
from monty.json import jsanitize
from scipy import interpolate
import copy

config_dyn_pol = {'q_points':[[0,0,0]], 'background_dielectric_constant':1.0, 'coulomb_constant':1.4399644*2*np.pi}
config_generic = {'correct_spin':True, 'nr_random_samples':1, 'nr_time_steps':2048, 'beta':11604.505/300, \
                  'mu':0, 'cuda':False, 'seed':1337}

class TipsiCalc(object):
    def __init__(self, tipsi_sample):
        self.sample = tipsi_sample
        self.norb = len(self.sample.site_x)

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

    def DOS_calc(self, ene_cut=20, config_generic=config_generic, B=0, save_to='DOS.json'):
        """
        ene_cut: energy range [-ene_cut, ene_cut] will be calculated for DOS
        """
        if B:
            self.sample.set_magnetic_field(B)
        config = tipsi.Config(self.sample)
        for i in config_generic:
            config.generic[i] = config_generic[i]
        config.sample['energy_range'] = ene_cut
        config.save()
        corr_DOS = tipsi.corr_DOS(self.sample, config)
        energies_DOS, DOS = tipsi.analyze_corr_DOS(config, corr_DOS)
        dos = {}
        dos['energies'] = energies_DOS
        dos['DOS'] = DOS
        dos = self._DOS_norm0(dos, self.norb, self.norb)
        with open(save_to,'w') as f:
            f.write(json.dumps(jsanitize(dos)))

    def AC_calc(self, config_generic=config_generic, B=0, save_to='AC.json'):
        """
        energy_range: the omegas maximum 
        """
        if B:
            self.sample.set_magnetic_field(B)
        config = tipsi.Config(self.sample)
        for i in config_generic:
            config.generic[i] = config_generic[i]
        config.save()
        corr_AC = tipsi.corr_AC(self.sample, config)
        omegas, AC= tipsi.analysis.analyze_corr_AC(config, corr_AC)
        ac = {}
        ac['omegas'] = omegas
        ac['AC'] = AC
        with open(save_to,'w') as f:
            f.write(json.dumps(jsanitize(ac)))
        return ac

    def plasmon_calc(self, config_generic=config_generic, config_dyn_pol=config_dyn_pol, save_to='plasmon.json'):
        config = tipsi.Config(self.sample)
        for i in config_generic:
            config.generic[i] = config_generic[i]
        for i in config_dyn_pol:
            config.dyn_pol[i] = config_dyn_pol[i]
        config.save()
        corr_dyn_pol = tipsi.corr_dyn_pol(self.sample, config)
        qval, omegas, dyn_pol = tipsi.analyze_corr_dyn_pol(config, corr_dyn_pol)
        qval, omegas, epsilon = tipsi.get_dielectric_function(config, dyn_pol)
        out = {}
        out['q_points'] = qval
        out['omegas'] = omegas
        out['dyn_pol_real'] = dyn_pol.real
        out['dyn_pol_imag'] = dyn_pol.imag
        out['epsilon_real'] = epsilon.real
        out['epsilon_imag'] = epsilon.imag
        with open(save_to,'w') as f:
            f.write(json.dumps(jsanitize(out)))
        return out

    def LDOS_calc(self, B=0, n_sample=1, n_timestep=1024, ldos_sites = [], ldos_separated=True, save_to = 'LDOS.json'):
        """
        Args:
            T: the magnetic field
            nr: the number of random states
            nt: the steps of time evolution
            ldos_sites: the sites of the samples, on which the local density of states will be calculated
            save_to: a file name with json extension, and the ldos data will be saved to it. 
                    please note only the json file will be used here.
        """

        if B:
            self.sample.set_magnetic_field(B)
        if ldos_separated:
            for i in ldos_sites:
                config = self.config_generic(nr, nt, save=False)  
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
        config = self.config(B, nr, nt, Temp, **dckb)
        dckb = {}
        dckb['n_kernel'] = n_kernel
        dckb['direction']  = direction
        dckb['ne_integral']  = ne_integral
        dckb['energies'] = np.arange(energy_range[0],energy_range[1],0.001)
        for i in dckb:
            config.dckb[i] = dckb[i]
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


