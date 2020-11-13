from tBG.scripts.quantum_dot.symmetry_analyzer import _parse
import numpy as np

t = 2.8
def occup_0K(vals):
    """
    for graphene system, one orbital offer one electron
    """
    ne = len(vals)
    ind_ef = int(ne/2) if ne%2 else int(ne/2)-1
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
    occup2 = [2.]*ind_ef_min
    occup_ef = [(ne - np.sum(occup2))/len(inds)]*len(inds)
    occup0 = [0.]*(ne-ind_ef_max-1)
    return np.array(occup2+occup_ef+occup0)

def get_vbm_cbm(vals):
    occup = occup_0K(vals)
    inds_ef = np.intersect1d(np.where(occup>0)[0], np.where(occup<2))
    if len(inds_ef):
        print('Catch a metal!')
        return vals[np.min(inds_ef)], vals[np.max(inds_ef)]
    else:
        ind_vbm = np.max(np.where(occup==2.)[0])
        ind_cbm = np.min(np.where(occup==0.)[0])
        return vals[ind_vbm], vals[ind_cbm]

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
        return np.linalg.norm(Jmn)**2*(fm-fn)/denominator

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

def calc_optical_conductivity(struct_f, eigen_f, gamma=0.02*t, e_win=10*0.05*t):
    omegas = np.arange(4*gamma, 3*t, 0.01*t)
    qd, vals, vecs = _parse(struct_f, eigen_f) 
    Jx, Jy, _ = qd.get_current_mat()
    sigma_x = Re_optical_conductivity(Jx, vals, vecs, omegas, gamma=gamma, e_win=e_win) 
    sigma_y = Re_optical_conductivity(Jy, vals, vecs, omegas, gamma=gamma, e_win=e_win)
    sigma = np.column_stack([omegas, sigma_x, sigma_y])
    np.savetxt('sigma.txt', sigma)
    return omegas, sigma_x, sigma_y 


