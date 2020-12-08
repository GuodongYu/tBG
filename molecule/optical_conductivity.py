import numpy as np
import pickle

def _parse(struct_f, eigen_f):
    with open(struct_f, 'rb') as f:
        qd = pickle.load(f)
    eigen = np.load(eigen_f, allow_pickle=True)
    vals = eigen['vals']
    vecs = eigen['vecs']
    return qd, vals, vecs


t = 2.8

def occup_0K(vals, spin=1):
    """
    for graphene system, one pz orbital contribute one electron
    """
    n_dim = len(vals)
    if spin == 1:
        ne = n_dim
        f_below = 2.0
    elif spin==2:
        ne = int(n_dim/2)
        f_below = 1.0
    ind_ef = int(n_dim/2) if n_dim%2 else int(n_dim/2)-1
    inds = [ind_ef]
    ef = vals[ind_ef]
    ind = ind_ef

    i = 1
    while True:
        e = vals[ind_ef-i]
        if ef - e > 1.e-5:
            break
        else:
            inds.append(ind_ef-i)
        i = i + 1 
    
    i = 1
    while True:
        e = vals[ind_ef+i]
        if e-ef > 1.e-5:
            break
        else:
            inds.append(ind_ef+i)
        i = i + 1 

    ind_ef_min = min(inds)
    ind_ef_max = max(inds)
    occup2 = [f_below]*ind_ef_min
    occup_ef = [(ne - np.sum(occup2))/len(inds)]*len(inds)
    occup0 = [0.]*(n_dim-ind_ef_max-1)
    return np.array(occup2+occup_ef+occup0)

def Fermi_func(e, mu, T):
    kb = 0.000086173324 # eV.K
    a = (e-mu)/(kb*T)
    return 1/(np.exp(a)+1)

def get_chem_pot(vals, n_e, T, de=0.1, spin=1, prec=1.e-5):
    vbm, cbm = get_vbm_cbm(vals, spin)
    mu0 = (vbm+cbm)/2.
    def occup_func(vals, mu, T):
        if spin==1:
            return 2*Fermi_func(vals, mu, T)
        elif spin==2:
            return Fermi_func(vals, mu, T)
    
    ## step 1
    n_e0 = np.sum(occup_func(vals, mu0, T))
    if abs(n_e0-n_e)<prec:
        return mu0
    if n_e0>n_e:
        de = -de
    else:
        de = de

    while True:
        n_e0 = np.sum(occup_func(vals, mu0, T))
        mu1 = mu0 + de
        n_e1 = np.sum(occup_func(vals, mu1, T))
        if (n_e0-n_e)*(n_e1-n_e)<0:
            break
        else:
            mu0 = mu1

    ## make sure mu0 < mu1
    mu0, mu1 = sorted([mu0, mu1])
    while True:
        mu_mid = (mu0+mu1)/2
        n_e_mid = np.sum(occup_func(vals, mu_mid, T))
        if abs(n_e_mid-n_e)<prec:
            return mu_mid
        else:
            if n_e_mid<n_e:
                mu0, mu1 = mu_mid, mu1
            else:
                mu0, mu1 = mu0, mu_mid
                
def occup_TK(vals, T=5, spin=1, de=0.1, prec=1.e-5):
    """
    Only for the system with one orbital contributing one electron

    vals: the sorted eigenvals 
    T: the temperature in units K
    spin: 1 (spin non-polorized), 2 (spin polorized)
    de: the energy step for calculating fermi level in eV
    prec: the precision for number of electrons
    """
    if spin==1:
        n_orb = len(vals)
    elif spin==2:
        n_orb = int(len(vals)/2)
    n_e = n_orb
    ef = get_chem_pot(vals, n_e, T, de, spin, prec)
    if spin==1:
        return 2*Fermi_func(vals, ef, T)
    elif spin==2:
        return Fermi_func(vals, ef, T)


def get_vbm_cbm(vals, spin=1):
    occup = occup_0K(vals, spin)
    inds_ef = np.intersect1d(np.where(occup>0)[0], np.where(occup<2./spin))
    if len(inds_ef):
        print('Catch a metal!')
        return vals[np.min(inds_ef)], vals[np.max(inds_ef)]
    else:
        ind_vbm = np.max(np.where(occup==2./spin)[0])
        ind_cbm = np.min(np.where(occup==0.)[0])
        return vals[ind_vbm], vals[ind_cbm]

def get_inds_band_edge(vals):
    occup = occup_0K(vals)
    inds_ef = np.intersect1d(np.where(occup>0)[0], np.where(occup<2))

    def get_inds_vbms_cbms(ind_init, step):
        inds = [ind_init]
        val = vals[ind_init]
        i = 1
        while True:
            ind_i = ind_init+step*i
            vali = vals[ind_i]
            if abs(vali-val)<1.e-8:
                inds.append(ind_i)
                i = i + 1
            else:
                break
        return inds
    ind_vbm = np.max(np.where(occup==2.)[0])
    ind_cbm = np.min(np.where(occup==0.)[0])
    inds_vbm = get_inds_vbms_cbms(ind_vbm, -1)
    inds_cbm = get_inds_vbms_cbms(ind_cbm, 1)
    return inds_vbm, inds_cbm, inds_ef
    
def pick_up_transition_pairs(vals, omega, e_win, occup):
    omega_minus = max(omega-e_win, 1.e-6)
    omega_plus = omega + e_win
    inds_ef = np.intersect1d(np.where(occup>0)[0], np.where(occup<2)[0])
    if len(inds_ef):
        ind_vbm = max(inds_ef)
        ind_cbm = min(inds_ef)
    else:
        ind_vbm = max(np.where(occup==2)[0])
        ind_cbm = min(np.where(occup==0)[0])
    vbm = vals[ind_vbm]
    cbm = vals[ind_cbm]
    
    e_bott = cbm - omega - e_win
    e_top = vbm + omega + e_win
    inds_shot = np.intersect1d(np.where(vals>=e_bott)[0], np.where(vals<=e_top)[0])

    inds_vb = np.arange(ind_vbm, inds_shot[0]-1, -1)
    inds_cb = np.arange(ind_cbm, inds_shot[-1]+1)

    def add_pair(ind_vb):
        e0 = vals[ind_vb]
        des = vals - e0
        inds_chosen = np.intersect1d(np.where(des>=omega_minus)[0], np.where(des<=omega_plus)[0])
        inds_chosen = np.intersect1d(inds_chosen, inds_cb)
        pairs_chosen = [[ind_vb, indi] for indi in inds_chosen]
        return pairs_chosen
    pairs = [add_pair(ind_vb) for ind_vb in inds_vb]
    pairs = [i for i in pairs if len(i)]
    if len(pairs):
        return np.concatenate(pairs)
    else:
        return []

def Re_optical_conductivity(J_mat, vals, vecs, omegas, gamma=0.05*t, e_win=5*0.05*t):
    """
    inputs:
        gamma: the energy width for Lorentzian function, which is used for simulate delta function
        e_win: the energy window to pick up energy level pairs for calculating optical transition, which means that for hbar*omega 
               two energy levels with hbar*omega -e_win <= deltaE <= hbar*omega+e_win are picked up for calculating
               the optical conductivity.
        omega_lim: the frequency range in units of eV (measured as hbar*omega)
        d_omega: the frequency resolution
    """
    e = 1 # electron charge
    hbar_eVs = 6.582119514 *10**(-16)
    h_eVs = 4.1356676969 * 10**(-15)
    sigma0 = (np.pi/2)*(e**2/h_eVs)
    occup = occup_0K(vals)
    def calc_sigma_mn_pair(indm, indn, omega):
        vecm = vecs[:,indm]
        vecn = vecs[:,[indn]]
        Jmn = np.dot(vecm.conj(), np.matmul(J_mat, vecn))
        fn = occup[indn]
        fm = occup[indm]
        de = vals[indn] - vals[indm]
        denominator = (omega-de)**2 + gamma**2
        denominator_ = (omega+de)**2 + gamma**2
        return np.linalg.norm(Jmn)**2*( (fm-fn)/denominator + (fn-fm)/denominator_)

    def calc_sigma_one_point(omega):
        a = 2.46
        A = len(vals)*(np.sqrt(3)/4*a**2)
        c = 2*gamma/(omega/hbar_eVs*A*10000)
        pairs = pick_up_transition_pairs(vals, omega, e_win, occup)   
        if not len(pairs):
            return 0.0
        sigmas_mn = [calc_sigma_mn_pair(pair[0], pair[1], omega) for pair in pairs]  
        return c*np.sum(sigmas_mn)*10000
    
    sigmas = [calc_sigma_one_point(omega) for omega in omegas]
    return np.array(sigmas)/sigma0

def calc_optical_conductivity(struct_f, eigen_f, omegas=np.arange(0.001, 5*t, 0.01*t), gamma=0.05, e_win=30*0.05*t, save_to='sigma.txt'):
    qd, vals, vecs = _parse(struct_f, eigen_f) 
    Jx, Jy = qd.get_current_mat()
    sigma_x = Re_optical_conductivity(Jx, vals, vecs, omegas, gamma=gamma, e_win=e_win) 
    sigma_y = Re_optical_conductivity(Jy, vals, vecs, omegas, gamma=gamma, e_win=e_win)
    sigma = np.column_stack([omegas, sigma_x, sigma_y])
    np.savetxt(save_to, sigma)
    return omegas, sigma_x, sigma_y 

def plot_optical_cond(t=2.8, sigma_f='sigma.txt', save_to='optical_cond.pdf'):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    sigma = np.loadtxt(sigma_f)
    ax.plot(sigma[:,0]/t, sigma[:,1], label='$\sigma_{xx}$')
    plt.legend()
    plt.savefig(save_to)
