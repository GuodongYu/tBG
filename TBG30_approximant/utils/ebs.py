import time
from mpl_toolkits import axisartist
import os
import numpy as np
import json
from tBG.TBG30_approximant.structure import Structure
from tBG.TBG30_approximant.band_unfold import BandUnfold, unfolded_band_struct

def f2e(x):
    if x<1000:
        return(str(x))
    y = '%.1e' % x
    x, y = y.split('e')
    y = int(float(y))
    if float(x) == 1.0:
        return '10^%s' % y
    else:
        return '%s\\times10^{%s}' % (x, y)

class PMK(object):
    def __init__(self, M, h=3.461, g1=0.48, append=None, vac_con=0, Anderson_W=0, hop_dist=5.0, elec_field=0.0):
        s = Structure()
        s.make_structure(M, h=h)
        if append:
            s.append_layers(append)
        if vac_con:
            s.add_vacancy_disorder(vac_con)
        s.add_hopping(max_dist=hop_dist, g1=g1)
        self.buf = BandUnfold(s, elec_field=elec_field)
        if Anderson_W:
            self.buf.add_Ansderon_disorder(Anderson_W)

    def kpoints_line_mode(self, k_path=['K_bot','M_top','KR_bot','D_bot','G','D_top','KR_top','M_bot','K_top'], dk=0.01):
        kpts, k_info = self.buf.kpoints_line_mode(k_path=k_path,dk=dk)
        return kpts, k_info

    def kpoints_line_onek(self, k_D='K_top', dk=0.005, direction=[1,0], nstep=40):
        kpts, k_info = self.buf.kpoints_line_onek(k_D=k_D, dk=dk, direction=direction, nstep=nstep)
        return kpts, k_info

    def run(self, ks, k_info=None, fname='KPT_EIG_PMK'):
        self.buf.PMK_calc(ks_cart=ks, k_info=k_info, fname=fname)

def plot_unfolded_band(ax=None, ebs_f='EBS.npz', fname='ebs.png', title='', title_size=40,\
                      ylim=[-16, 8], scale=1, Ef=0.0, E0s=[-1.46, -1.59], inset=False, params=None):
    plot = False
    if ax is None:
        from matplotlib import pyplot as plt
        if params:
            plt.rcParams.update(params)
            plt_param = 'params'
        else:
            try:
                import tBG
                tBG.params['figure.figsize']=[8,10]
                plt.rcParams.update(tBG.params)
                plt_param = 'tBG_params'
            except:
                plt_param = 'default'
                pass
        fig = plt.figure(1)
        ax = axisartist.Subplot(fig, 111)
        fig.add_subplot(ax)
        plot = True

    label_change={'K_bot':'K', 'KR_bot':'$\mathbf{\widetilde{K}_1}$', 'D_bot':'$\mathbf{\widetilde{K}_2}$','M_bot':'M', \
                  'K_top':'$\mathbf{\widetilde{K}}$', 'KR_top':'$\mathbf{K_1}$', 'D_top':'$\mathbf{K_2}$', \
                  'M_top':'$\mathbf{\widetilde{M}}$', 'G':'$\mathbf{\Gamma}$', 'M_bot_orig':'$\mathbf{M_0}$',\
                  'M_top_end':'$\mathbf{\widetilde{M}_0}$'}

    data = np.load(ebs_f)
    es = data['energies'] - Ef
    As = data['EBS']*20
    kpts = data['kpoints']
    kpath = data['k_info'][0]
        

    def Xs():
        ks = np.array([0.0])
        dks = kpath['dks']
        inds = kpath['inds']
        for i in range(len(dks)):
            if kpath['type'] == 'EBS_path':
                ind0, ind1 = inds[i]+1, inds[i+1]
            elif kpath['type'] == 'EBS_onek':
                ind0, ind1 = 1, len(kpts)-1
            ks_i = np.array([j*dks[i] for j in range(1, ind1-ind0+2)]) + ks[-1]
            ks = np.concatenate((ks, ks_i), axis=0)
        return ks
    ks = Xs()

    XX, YY = np.meshgrid(ks, es)
    ax.scatter(XX, YY, s=As*scale, color='black',linewidth=0.)


    if inset:
        def circle_dirac(ax):
            diracs = ['K_bot', 'K_top', 'KR_bot', 'KR_top', 'D_bot', 'D_top']
            for label in diracs:
                i = dict(zip(k_info['labels'], ks_label))[label]
                ax.scatter(i, 0.0, facecolors='none', linewidths=0.5, edgecolors='blue', s=150, clip_on=False)
        #circle_dirac(ax)
        def get_kpt(label):
            for i in range(len(kpath['labels'])):
                if kpath['labels'][i]==label:
                    return ks[kpath['inds'][i]]
            
        def inset_plot(label, axinset, xlim_in, ylim_in, rect_plot=False, scale=1, hvlines=False, txt=''):
            def closest_ind(l, x):
                return np.argmin(np.abs(np.array(l)-x))
            def get_inds():
                ind_x0 = closest_ind(ks, xlim_in[0])
                ind_x1 = closest_ind(ks, xlim_in[1])
                ind_y0 = closest_ind(es, ylim_in[0])
                ind_y1 = closest_ind(es, ylim_in[1])
                return ind_x0, ind_x1, ind_y0, ind_y1
            ind_x0, ind_x1, ind_y0, ind_y1 = get_inds()
            axinset.set_adjustable('box')
            axinset.scatter(XX[ind_y0:ind_y1,ind_x0:ind_x1], YY[ind_y0:ind_y1,ind_x0:ind_x1],\
                            s=As[ind_y0:ind_y1,ind_x0:ind_x1]*scale, color='black',linewidth=0.)
            for i in E0s:
                axinset.axhline(i, linestyle='dashed', color='black', linewidth=0.5)
            k = get_kpt(label)
            axinset.set_xlim(xlim_in)
            axinset.set_ylim(ylim_in)
            axinset.set_xticklabels('')
            axinset.set_yticklabels('')
            axinset.set_xticks([])
            axinset.set_yticks([])
            if rect_plot:
                left, right = xlim_in
                bott, top = ylim_in
                ax.plot([left, left, right, right, left], [bott, top, top, bott, bott], \
                         linewidth=0.5, color='blue')
            if hvlines:
                for i in kpath['inds']:
                    axinset.axvline(ks[i], linestyle='dashed', color='black', linewidth=0.8)
                axinset.axhline(0.0, linestyle='dashed', color='black',linewidth=0.8)
            if txt:
                axinset.text(0.5, 0.47, txt, fontsize=9, transform=axinset.transAxes, \
                             verticalalignment='center', horizontalalignment='center')

        def get_zone_to(xlim_in, ylim_in):
            xlim = [ks[0], ks[-1]]
            left, right = xlim_in
            left_ratio = (left-xlim[0])/(xlim[1]-xlim[0])
            right_ratio = (right-xlim[0])/(xlim[1]-xlim[0])
            width = right_ratio - left_ratio
            
            bott, top = ylim_in
            bott_ratio = (bott-ylim[0])/(ylim[1]-ylim[0])
            top_ratio = (top-ylim[0])/(ylim[1]-ylim[0])
            height = top_ratio - bott_ratio
            return [left_ratio, bott_ratio, width, height]

        ## M_top
        #kpt_M_top = get_kpt('M_top')
        #xlim_in = [kpt_M_top-8*dk, kpt_M_top+8*dk]
        #ylim_in = [-2.0, -1.0]
        #axinset = ax.inset_axes([0.01, 0.01, 0.25, 0.25])
        #inset_plot('M_top', axinset, xlim_in, ylim_in, rect_plot=True, scale=1., txt='126 meV')

        ## M_bot
        #kpt_M_bot = get_kpt('M_bot')
        #xlim_in = [kpt_M_bot-8*dk, kpt_M_bot+8*dk]
        #ylim_in = [-2.0, -1.0]
        #axinset1 = ax.inset_axes([0.74, 0.01, 0.25, 0.25])
        #inset_plot('M_bot', axinset1, xlim_in, ylim_in, rect_plot=True, scale=1., txt='128 meV')

        #KR_bot
        kpt_KR_bot = get_kpt('KR_bot')
        dk = np.mean(kpath['dks'])
        xlim_in = [kpt_KR_bot-15*dk, kpt_KR_bot+15*dk]
        ylim_in = [-0.7, 0.7]
        zone_to = [xlim_in[0],ylim_in[0],xlim_in[1]-xlim_in[0],ylim_in[1]-ylim_in[0]]
        axinset2 = ax.inset_axes(zone_to, transform=ax.transData)
        scale_in = 300
        inset_plot('KR_bot', axinset2, xlim_in, ylim_in, rect_plot=True, scale=scale_in, hvlines=True)
        #circle_dirac(axinset2)
        ax.text(xlim_in[0]/2+xlim_in[1]/2, ylim_in[1]+0.1, '$\\times %s$' % f2e(scale_in), horizontalalignment='center')

        #KR_top
        kpt_KR_top = get_kpt('KR_top')
        xlim_in = [kpt_KR_top-15*dk, kpt_KR_top+15*dk]
        ylim_in = [-0.7, 0.7]
        zone_to = [xlim_in[0],ylim_in[0],xlim_in[1]-xlim_in[0],ylim_in[1]-ylim_in[0]]
        axinset3 = ax.inset_axes(zone_to, transform=ax.transData)
        scale_in = 300
        inset_plot('KR_top', axinset3, xlim_in, ylim_in, rect_plot=True, scale=scale_in, hvlines=True)
        #circle_dirac(axinset3)
        ax.text(xlim_in[0]/2+xlim_in[1]/2, ylim_in[1]+0.1, '$\\times %s$' % f2e(scale_in), horizontalalignment='center')

        ##Gamma
        #kpt_G = get_kpt('G')
        #xlim_in = [kpt_G-0.1*10.5, kpt_G+0.1*10.5]
        #ylim_in = [0.2-0.905, 1.6-0.905]
        #zone_to = get_zone_to(xlim_in, ylim_in) 
        #axinsetG = ax.inset_axes(zone_to)
        #scale_in = 1.e+6
        #inset_plot('G', axinsetG, xlim_in, ylim_in, rect_plot=True, scale=scale_in, hvlines=True)
        #circle_dirac(axinsetG)
        #ax.text(xlim_in[0]/2+xlim_in[1]/2, ylim_in[1], '$\\times %s$' % f2e(scale_in), fontsize=10, horizontalalignment='center')
        
        ##D_top
        #kpt_D_top = get_kpt('D_top')
        #xlim_in = [kpt_D_top-0.1, kpt_D_top+0.1]
        #ylim_in = [0.2, 1.6]
        #zone_to = get_zone_to(xlim_in, ylim_in) 
        #axinset4 = ax.inset_axes(zone_to)
        #inset_plot('D_top', axinset4, xlim_in, ylim_in, rect_plot=True, scale=1e+3, hvlines=True)
        #circle_dirac(axinset4)
    
        ##D_bot
        #kpt_D_bot = get_kpt('D_bot')
        #xlim_in = [kpt_D_bot-0.1*5, kpt_D_bot+0.1*5]
        #ylim_in = [0.2, 1.6]
        #zone_to = get_zone_to(xlim_in, ylim_in) 
        #axinset5 = ax.inset_axes(zone_to)
        #inset_plot('D_bot', axinset5, xlim_in, ylim_in, rect_plot=True, scale=1e+6, hvlines=True)
        #circle_dirac(axinset5)
    kpath['inds'] = [kpath['inds'][i] for i in range(len(kpath['inds'])) if i not in [3,5] ]
    for i in [ks[ind] for ind in kpath['inds']]:
        ax.axvline(i, linestyle='dashed', color='black', linewidth=0.8)
    ax.axhline(0.0, linestyle='dashed', color='black',linewidth=0.8)
    if kpath['type']=='EBS_path':
        kpath['labels'].remove('D_top')
        kpath['labels'].remove('D_bot')
        kpath['dks'] = [kpath['dks'][i] for i in range(len(kpath['dks'])) if i not in [2,3,4]]
    ax.set_xlim((ks[0], ks[-1]))
    ax.set_xticks([ks[i] for i in kpath['inds']])
    ax.set_yticks(range(-20, 20, 2))
    labels = [label_change[i] for i in kpath['labels']]
    ax.set_xticklabels(labels)
    ax.text(0.45, 0.06, title, transform=ax.transAxes,bbox=dict(boxstyle="round",
                   fc=(1, 1, 1),ec=(0, 0., 0.)),fontsize=title_size, verticalalignment='bottom')
    if ylim:
        ax.set_ylim(ylim)
    if scale != 1.0:
        ax.text(0.24, 0.88, '$\\times%s$' % scale, transform=ax.transAxes)
    ax.set_ylabel('$\mathbf{Energy-E_f}$ (eV)')
    if plot:
        plt.savefig(fname, bbox_inches='tight',pad_inches=0, dpi=600)
        plt.clf()

def plot_unfolded_band2(ax=None, ebs_f='EBS.npz', fname='ebs.png', title='', title_size=40,\
                      ylim=[-16, 8], scale=1, Ef=0.0, E0s=[-1.46, -1.59], inset=False, params=None):
    plot = False
    if ax is None:
        from matplotlib import pyplot as plt
        if params:
            plt.rcParams.update(params)
            plt_param = 'params'
        else:
            try:
                import tBG
                tBG.params['figure.figsize']=[8,10]
                plt.rcParams.update(tBG.params)
                plt_param = 'tBG_params'
            except:
                plt_param = 'default'
                pass
        fig = plt.figure(1)
        ax = axisartist.Subplot(fig, 111)
        fig.add_subplot(ax)
        plot = True

    label_change={'K_bot':'K', 'KR_bot':'$\mathbf{\widetilde{K}_1}$', 'D_bot':'$\mathbf{\widetilde{K}_2}$','M_bot':'M', \
                  'K_top':'$\mathbf{\widetilde{K}}$', 'KR_top':'$\mathbf{K_1}$', 'D_top':'$\mathbf{K_2}$', \
                  'M_top':'$\mathbf{\widetilde{M}}$', 'G':'$\mathbf{\Gamma}$', 'M_bot_orig':'$\mathbf{M_0}$',\
                  'M_top_end':'$\mathbf{\widetilde{M}_0}$'}

    data = np.load(ebs_f)
    es = data['energies'] - Ef
    As = data['EBS']*20
    kpts = data['kpoints']
    kpath = data['k_info'][0]
        

    def Xs():
        ks = np.array([0.0])
        dks = kpath['dks']
        inds = kpath['inds']
        for i in range(len(dks)):
            if kpath['type'] == 'EBS_path':
                ind0, ind1 = inds[i]+1, inds[i+1]
            elif kpath['type'] == 'EBS_onek':
                ind0, ind1 = 1, len(kpts)-1
            ks_i = np.array([j*dks[i] for j in range(1, ind1-ind0+2)]) + ks[-1]
            ks = np.concatenate((ks, ks_i), axis=0)
        return ks
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import Normalize
    ks = Xs()
    xmin = min(ks)
    xmax = max(ks)
    ymin = min(es)
    ymax = max(es)
    im = ax.imshow(As,origin='lower',aspect='auto',extent=[xmin,xmax,ymin,ymax],interpolation='bilinear',\
                cmap = 'jet',norm=Normalize(vmin=0,vmax=1.0),alpha=0.9)
    #fig.colorbar(im, cax=cax, orientation='horizontal')

    #kpath['inds'] = [kpath['inds'][i] for i in range(len(kpath['inds'])) if i not in [3,5] ]
    for i in [ks[ind] for ind in kpath['inds']]:
        ax.axvline(i, linestyle='dashed', color='black', linewidth=0.8)
    ax.axhline(0.0, linestyle='dashed', color='black',linewidth=0.8)
    if kpath['type']=='EBS_path':
        kpath['labels'].remove('D_top')
        kpath['labels'].remove('D_bot')
        kpath['dks'] = [kpath['dks'][i] for i in range(len(kpath['dks'])) if i not in [2,3,4]]
    ax.set_xticks([ks[i] for i in kpath['inds']])
    labels = [label_change[i] for i in kpath['labels']]
    ax.set_xticklabels(labels)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_ylabel('$\mathbf{Energy-E_f}$ (eV)')
    if plot:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%',pad=0.2)
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.tight_layout(pad=0.1, h_pad=0.5, w_pad=None, rect=None)
        plt.savefig(fname, bbox_inches='tight',pad_inches=0, dpi=600)
        plt.clf()
    else:
        return xmin, xmax, kpath['inds']

def plot_unfolded_band2_old():
    unfold_band_f = 'UNFOLDED_BAND'
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib.colors import BoundaryNorm
    fig, (ax) = plt.subplots(nrows=1)
    cmap = plt.get_cmap('RdYlBu')
    with open(unfold_band_f, 'r') as f:
        x = f.readline().split()
        nk = int(x[1])
        ne = int(x[-1])
        skip = f.readline()
        e0, kx0, ky0, a0 = [float(i) for i in f.readline().split()]
        e1, kx1, ky1, a1 = [float(i) for i in f.readline().split()]
    data = np.loadtxt(unfold_band_f)
    dk = np.linalg.norm(np.array([kx0, ky0]) - np.array([kx1, ky1]))
    ks = [dk*i for i in range(nk)]*int(len(data)/nk)
    ne = int(len(ks)/nk)
    es = data[:,0]
    #As = np.float16(np.abs(data[:,-1]))
    As = np.float16(data[:,-1])
    X = np.reshape(ks, (nk, ne))
    Y = np.reshape(es, (nk, ne))
    C = np.reshape(As, (nk, ne))
    levels = MaxNLocator(nbins=100).tick_values(C.min(), C.max())
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    im = ax.imshow(C, cmap=cmap, norm=norm)
    if colorbar:
        fig.colorbar(im, ax=ax)
    plt.savefig('unfolded_band_structure_mesh.png')
    plt.clf()


def ebs_path_calc(M=None, h=3.461, g1=0.48, append=None, vac_con=0., Anderson_W=0, elec_field=0.0, elim=[-15,8], de=0.01, \
             pmk_calc=False, ebs_calc=False,\
             k_path=['M_bot_orig','K_bot','M_top','KR_bot','D_bot','G','D_top','KR_top','M_bot','K_top','M_top_end'], dk=0.01, \
             kpt_eig_pmk_f='KPT_EIG_PMK_ebs_path', ebs_f='EBS_path'):

    if pmk_calc:
        if M is None:
            raise ValueError('M must be given.')
        pmk = PMK(M, h=h, g1=g1, append=append, vac_con=vac_con, Anderson_W=Anderson_W, elec_field=elec_field)
        kpts, k_info = pmk.kpoints_line_mode(k_path=k_path, dk=dk)
        pmk.run(kpts, k_info, fname=kpt_eig_pmk_f)
    if ebs_calc:
        unfolded_band_struct(elim=elim, de=de, kpt_eig_pmk_f=kpt_eig_pmk_f+'.npz',fname=ebs_f)

def direct_label(direct, fmt=True):
    if direct == [1,0]:
        s = '$\mathbf{\\rightarrow x}$' if fmt else 'x'
    elif direct == [0,1]:
        s = '$\mathbf{\\rightarrow y}$' if fmt else 'y'
    elif direct == [1,1]:
        s = '$\mathbf{\\rightarrow x+y}$' if fmt else 'x+y'
    elif direct == [1,-1]:
        s = '$\mathbf{\\rightarrow x-y}$' if fmt else 'x-y'
    else:
        raise ValueError('direct is not accepted')
    return s

def get_f_name(Dirac, direct):
    label = Dirac+'_'+str(direct[0])+str(direct[1])
    kpt_eig_pmk_f= 'KPT_EIG_PMK_ebs_'+label
    ebs_f = 'EBS_'+label
    png_f = 'ebs_'+label+'.png'
    return kpt_eig_pmk_f, ebs_f, png_f
    
def ebs_onek_calc(M, append=None, elec_field=0.0, elim=[-15,8], de=0.01, dk=0.002, nstep=40,\
                  Diracs = ['K_bot','KR_bot','K_top','KR_top'], \
                  directs = [[1,0],[0,1],[1,1],[1,-1]]):

    pmk = PMK(M, append, elec_field=elec_field)
    for i in range(len(Diracs)):
        D = Diracs[i]
        for j in range(len(directs)):
            direct = directs[j]
            kpt_eig_pmk_f, ebs_f, fname = get_f_name(D, direct)
            print('%s %s' % (D, direct_label(direct, fmt=False)))
            if not os.path.isfile(kpt_eig_pmk_f+'.npz'):
                print('   PMK running...')
                kpts, k_info = pmk.kpoints_line_onek(k_D=D, dk=dk, direction=direct, nstep=nstep)
                pmk.run(kpts, k_info, fname=kpt_eig_pmk_f)
                print('   PMK done')
            else:
                print('   PMK file existed, not run again!')
            if not os.path.isfile(ebs_f+'.npz'):
                print('   EBS running...')
                unfolded_band_struct(elim=elim, de=de, kpt_eig_pmk_f=kpt_eig_pmk_f+'.npz',fname=ebs_f)
                print('   EBS done\n')
            else:
                print('   EBS file existed, not run again\n')

def ebs_onek_plot(ylim=[-0.6,0.6], one_fig=True, Ef=0.0, with_D=False, title='', title_size=40, params=None,\
                  Diracs = ['K_bot','KR_bot','K_top','KR_top'],\
                  directs = [[1,0],[0,1],[1,1],[1,-1]], path='.', fname='ebs_ks.png'):
    scales = {'K_bot':1,'K_top':1,'KR_bot':300,'KR_top':300}
    if with_D:
        Diracs.append('D_bot')
        Diracs.append('D_top')
        scales['D_bot'] = 5e+6
        scales['D_top'] = 5e+6
    if one_fig:
        from matplotlib import pyplot as plt
        if params:
            plt.rcParams.update(params)
        else:
            try:
                import tBG
                plt.rcParams.update(tBG.params)
            except:
                pass
        fig = plt.figure(1)
        k = 1
        for i in range(len(Diracs)):
            D = Diracs[i]
            for j in range(len(directs)):
                ax = axisartist.Subplot(fig, len(Diracs),len(directs),k)
                fig.add_subplot(ax)
                direct = directs[j]
                kpt_eig_pmk_f, ebs_f, f_name = get_f_name(D, direct)
                plot_unfolded_band(ax=ax, ebs_f=os.path.join(path, ebs_f+'.npz'), ylim=ylim, scale=scales[D], Ef=Ef)
                ax.set_ylabel('')
                if not i:
                    ax.set_title(direct_label(direct))
                if j:
                    ax.set_yticks([])
                else:
                    ax.set_ylabel('$\mathbf{Energy-E_f}$ (eV)')
                    ax.set_yticks([-0.6, -0.3, 0.0, 0.3, 0.6])
                k = k + 1
        fig.suptitle(title, x=0.5, y=1.01, fontsize=title_size)
        plt.tight_layout(w_pad=0.01, h_pad=0.01)
        plt.subplots_adjust(wspace=0, hspace=0.2)
        plt.savefig(fname, bbox_inches='tight',pad_inches=0.2)
        plt.clf()
    else:
        for i in range(len(Diracs)):
            D = Diracs[i]
            for j in range(len(directs)):
                direct = directs[j]
                kpt_eig_pmk_f, ebs_f, fname = get_f_name(D, direct)
                plot_unfolded_band(fname=fname, ebs_f=os.path.join(path, ebs_f+'.npz'), ylim=ylim, scale=scales[D])

if __name__ == '__main__':
    ebs_onek_calc(15, append={'A':[-1,2]}, Diracs=['K_bot'], directs=[[1,0]])
    ebs_onek_plot(Diracs=['K_bot'], directs=[[1,0]])
