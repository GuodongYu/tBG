from tBG.TBG30_approximant.structure import  cart2frac, Structure
import numpy as np

#peaks = [-2.28, -2.06, -1.94, -1.81, -1.46]
peaks = [-2.28, -2.06, -1.81]

def coords_turn(struct):
    coords = struct.coords
    latt = struct.latt_compatible
    latt_2 = latt/2 
    coords_new = []
    for coord in coords:
        i, j = np.floor(cart2frac(coord[:-1], latt_2))
        if i == -1:
            i = 0
        if j == -1:
            j = 0
        if i == 2:
            i = 1
        if j == 2:
            j = 1
        shift = np.concatenate((-i*latt[0]-j*latt[1],[0.]), axis=0)
        coords_new.append(coord+shift)
    return np.array(coords_new)

def get_subsystems(st):
    argsort = np.argsort(st.layer_zcoords)
    subsys = [[argsort[0]]]
    for i in range(1,len(argsort)):
        lay_id = argsort[i]
        lay_id_before = argsort[i-1]
        tp = st.layer_types[lay_id]
        tp_before = st.layer_types[lay_id_before]
        if len(tp)==len(tp_before):
            subsys[-1].append(argsort[i])
        else:
            subsys.append([argsort[i]])
    return subsys

def plot_ratio_among_sub_systems(ax=None, eig_f='EIGEN.npz', xlim=None, ylim=None, Ef=0.0, subsys_plot=[0]):
    cs = ['gray','red','blue']
    data = np.load(eig_f)
    struct = data['struct'][0]
    subsys = get_subsystems(struct)
    layer_nsite = struct.layer_nsites
    sub_nsite = [np.sum([layer_nsite[j] for j in i]) for i in subsys]
    nsite = np.sum(layer_nsite)
    inds = struct._layer_inds()
    vals = data['vals']
    chgs = data['chgs']
    nsub = len(subsys)

    plot=False
    if not ax:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1,1)
        plot = True

    for ik in range(len(data['kpoints'])):
        es = data['vals'][ik] - Ef
        chg_subs = []
        for isub in range(len(subsys)):
            sub = subsys[isub]
            chg_sub = 0.
            for layi in sub:
                ind = inds[layi]
                chg_sub = chg_sub + np.sum(chgs[ik][:,ind[0]:ind[1]+1], axis=1)
            chg_subs.append(chg_sub/sub_nsite[isub]*nsite)
            #chg_subs.append(chg_sub)
        chg_subs = [i/np.sum(chg_subs, axis=0) for i in chg_subs]
        #for i in range(len(chg_subs)):
        k = 0
        for i in subsys_plot:
            ax.scatter(es, chg_subs[i], alpha=0.5, s=10, color=cs[k],facecolors='none')
            k = k + 1
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if plot:
        plt.xlabel('$\mathbf{Energy-E_f}$ (eV)')
        plt.ylabel('Occupation')
        plt.savefig('Occupation.pdf', bbox_inches='tight', pad_inches=0)
        
            

def plot_eigen_vec(eig_f, e_range=[-4, 3], size=[2,2], Ef=0.905, rotation_at_center=True, split=True):
    """
    rotation_at_center: whether shift the rotation centor to the middile of the unit cell
    split: if true, the wavefunction will plot also in different subsystems
    """

    data = np.load(eig_f)
    struct = data['struct'][0]
    subsys = get_subsystems(struct)
    struct.add_hopping(max_dist=2.0)
    scale = 40000
    #cs = ['blue','red','purple','green','yellow','gray']*10
    cs = ['black']*10
    coords_expand = []
    if rotation_at_center:
        print("shift rotation centor at middle of the unit cell, and only unit cell is plot")
        size = [1, 1]
        coords = coords_turn(struct)
    else:
        coords = struct.coords
    latt = struct.latt_compatible
    inds = struct._layer_inds()
    vals = data['vals']
    chgs = data['chgs']
    for ik in range(len(data['kpoints'])):
        es = vals[ik] - Ef
        for ie in range(len(es)):
            e = es[ie]
            if e<e_range[0]:
                continue   
            if e>e_range[1]:
                break
            fig_name = 'chg_k%s_e%s.png' % (ik, ie) 
            from matplotlib import pyplot as plt
            if not split:
                fig, ax = plt.subplots(1,1, figsize=[20,20])
            else:
                fig, ax_ = plt.subplots(2,2, figsize=[20,20])
                ax = ax_[0][0]
                axes = [ax_[0][1], ax_[1][0], ax_[1][1]]
            if not rotation_at_center:
                struct._plot(ax, size=size)
            for i in range(size[0]):
                for j in range(size[1]):
                    tran = latt[0]*i + latt[1]*j
                    O = tran
                    A = tran + latt[1]
                    B = tran + latt[0]
                    C = tran + latt[0] + latt[1]
                    if not rotation_at_center:
                        ax.plot([O[0], A[0], C[0], B[0], O[0]], \
                             [O[1], A[1], C[1], B[1], O[1]], color='black', linestyle='dashed', linewidth=0.1,clip_on=False)
                    coords_ij = tran+coords[:,0:-1]
                    #for layi in range(len(struct.layer_nsites)):
                    #    ind = inds[layi]
                    #    ax.scatter(coords_ij[:,0][ind[0]:ind[1]+1], coords_ij[:,1][ind[0]:ind[1]+1], \
                    #                s = chgs[ik][:,ie][ind[0]:ind[1]+1]*scale, color=cs[layi],clip_on=False)
                    for sub_id in range(len(subsys)):
                        sub = subsys[sub_id]
                        for layer_id in sub:
                            ind = inds[layer_id]
                            ax.scatter(coords_ij[:,0][ind[0]:ind[1]+1], coords_ij[:,1][ind[0]:ind[1]+1], \
                                    s = chgs[ik][:,ie][ind[0]:ind[1]+1]*scale, color=cs[sub_id],clip_on=False)
                            try:
                                axes[sub_id].scatter(coords_ij[:,0][ind[0]:ind[1]+1], coords_ij[:,1][ind[0]:ind[1]+1], \
                                    s = chgs[ik][:,ie][ind[0]:ind[1]+1]*scale, color=cs[sub_id],clip_on=False)
                            except:
                                pass
            ax.tick_params(axis="y",direction="in", pad=-22, labelsize=0.)
            ax.tick_params(axis="x",direction="in", pad=-15, labelsize=0.)
            ax.set_title('$E-E_f=%.2f eV$' % e) 
            try:
                for axi in axes:
                    axi.set_aspect('equal',adjustable='box')
                    axi.axis('off')
            except:
                pass
            ax.set_aspect('equal',adjustable='box')
            ax.axis('off')
            plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
            plt.clf()

def plot_eigen_vec_turn(eig_f, std_f, e_range=None, std_prec=0.01, max_prec=0.001, Ef=-0.905):
    """
    Args:
        eig_f: the EIGEN file
        std_f: the file saving the std and max occupation of all states
        e_range: [0,1] the states in this energy range will be ploted, the e_shift is alread added
        std_prec: only std less than std_prec, state will be ploted, to keep C12 symmetry
        max_prec: only max occupation more than max_prec, state will be ploted, to filter the uniform distribution state
        e_shift: the eigen values will be shift by e_shift ( make Fermi energy at 0)
    """
    es_c12=[]
    data = np.load(eig_f)
    info = np.load(std_f)
    stds = info['std']
    stds_all = info['std_all']
    maxs = info['maxs']
    struct = data['struct'][0]
    scale = 30000
    cs = ['blue','red','purple','green','yellow','gray']*10
    coords = coords_turn(struct)[:,:-1]
    latt = struct.latt_compatible
    inds = get_ind_range_in_each_layer(struct.layer_nsites)
    vals = data['vals']
    chgs = data['chgs']
    for ik in range(len(data['kpoints'])):
    #for ik in [k_th]:
        es = vals[ik] + e_shift
        for ie in range(len(data['vals'][0])):
            e = es[ie]
            if e_range is not None:
                if e <= e_range[0]:
                    continue
                if e >= e_range[1]:
                    break
            print(e)
            std = stds[ik][ie]
            std_all = stds_all[ik][ie]
            if not (std<std_prec and maxs[ik][ie]>max_prec):
                continue
            es_c12.append(e)
            fig_name = 'chg_k%s_e%s.png' % (ik, ie) 
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(1,1)
            O = [0,0]
            A = latt[1]
            B = latt[0]
            C = latt[0] + latt[1]
            #ax.plot([O[0], A[0], C[0], B[0], O[0]], \
            #         [O[1], A[1], C[1], B[1], O[1]], color='black', linestyle='dashed', linewidth=0.1,clip_on=False)
            for layi in range(len(struct.layer_nsites)):
                ind = inds[layi]
                ax.scatter(coords[:,0][ind[0]:ind[1]+1], coords[:,1][ind[0]:ind[1]+1], linewidths=0,alpha=0.2,\
                            s = chgs[ik][:,ie][ind[0]:ind[1]+1]*scale, color=cs[layi],clip_on=False)
            ax.tick_params(axis="y",direction="in", pad=-22, labelsize=0.)
            ax.tick_params(axis="x",direction="in", pad=-15, labelsize=0.)
            ax.set_title('%.2f eV, std:%.5f, max:%.5f' % (e, std, maxs[ik][ie])) 
            ax.set_xlim((-max([latt[0][0]/2, latt[1][0]/2]), max([latt[0][0]/2, latt[1][0]/2])))
            ax.set_ylim((-max([latt[0][1]/2, latt[1][1]/2]), max([latt[0][1]/2, latt[1][1]/2])))
            ax.set_aspect('equal',adjustable='box')
            ax.axis('off')
            plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
            plt.clf()
    np.save('vals_C12', es_c12)

