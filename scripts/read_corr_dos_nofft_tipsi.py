"""read_corr_dos_nofft.py
Read graphene TBPM correlation functions from sim_data folder, get dos with more data (N*N_DOSfactor)

original written Zhen Zhan in 20181030
Improved by Guodong Yu 07-2020 by rewritting all for cycles
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import tipsi

# Hanning window
def window_Hanning(i, N):
    """ Hanning window.
    
    Parameters: 
        i: integer, summation index
        N: integer, total length of summation
    Return:
        Hanning window value
    """
    return 0.5 * (1 + np.cos(np.pi * i /N))

def dos_interplation_from_files(config_f, corr_DOS_f, N_DOSfactor=8):
    
    # read config file and correlation file
    config = tipsi.read_config(config_f)
    corr_DOS = tipsi.read_corr_DOS(corr_DOS_f)
    
    # get config parameters
    tnr = config.generic['nr_time_steps']  # Number of time steps
    en_range = config.sample['energy_range']
   
    Ntnr = N_DOSfactor * tnr  # the new number of energy points
    energies_DOS = np.array([0.5 * i * en_range / Ntnr - en_range / 2. for i in range(Ntnr * 2) ])
    t_step = np.pi / (1 * en_range)

    corr_DOS = window_Hanning(np.arange(tnr), tnr) * corr_DOS
    
    # get DOS
    DOS = np.zeros(Ntnr * 2)
    spectrum = np.ones(Ntnr * 2)

    new = np.zeros(2*Ntnr)
    
    def put_value(i):
        omega_j = np.pi * (i-Ntnr) / Ntnr * np.arange(1,tnr+1)
        new[i] = np.sum(2*( (np.real(corr_DOS)*np.cos(omega_j) -np.imag(corr_DOS)*np.sin(omega_j))))

    tmp = [put_value(i) for i in range(2*Ntnr)] 
    spectrum = spectrum + new 
    spectrum[spectrum<0] = 0
    
    if config.generic['correct_spin']:
        NSpin = 2
    else:
        NSpin = 1
     
    DOS = spectrum / np.pi * t_step * NSpin

    np.savetxt("DOS_new.txt", np.column_stack((energies_DOS, DOS)))


def dos_interplation(tnr, en_range, correct_spin, corr_DOS, N_DOSfactor=8):
   
    Ntnr = N_DOSfactor * tnr  # the new number of energy points
    energies_DOS = np.array([0.5 * i * en_range / Ntnr - en_range / 2. for i in range(Ntnr * 2) ])
    t_step = np.pi / (1 * en_range)

    corr_DOS = window_Hanning(np.arange(tnr), tnr) * corr_DOS
    
    # get DOS
    DOS = np.zeros(Ntnr * 2)
    spectrum = np.ones(Ntnr * 2)

    new = np.zeros(2*Ntnr)
    
    def put_value(i):
        omega_j = np.pi * (i-Ntnr) / Ntnr * np.arange(1,tnr+1)
        new[i] = np.sum(2*( (np.real(corr_DOS)*np.cos(omega_j) -np.imag(corr_DOS)*np.sin(omega_j))))

    tmp = [put_value(i) for i in range(2*Ntnr)] 
    spectrum = spectrum + new 
    spectrum[spectrum<0] = 0
    
    if correct_spin:
        NSpin = 2
    else:
        NSpin = 1
     
    DOS = spectrum / np.pi * t_step * NSpin

    np.savetxt("DOS_new.txt", np.column_stack((energies_DOS, DOS)))


    
