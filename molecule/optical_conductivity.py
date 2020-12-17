import numpy as np

t = 2.8
    
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

def plot_optical_cond(sigma_f='sigma.txt', save_to='optical_cond.pdf'):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    sigma = np.loadtxt(sigma_f)
    ax.plot(sigma[:,0], sigma[:,1], label='$\sigma_{xx}$')
    plt.legend()
    plt.savefig(save_to)
