from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
import tBG
tBG.params['font.size']=25
tBG.params['lines.linewidth']=4
tBG.params['figure.figsize']=[10,7.5]
plt.rcParams.update(tBG.params)



def with_AAtld():
    fig = plt.figure()
    gs = GridSpec(6, 3)
    gs.update(top=1, bottom=0., hspace=0.0, wspace=0.2)
    ps = {'A-At': gs[:2,0], 'A-A-At':gs[2:4,0], 'B-A-At':gs[4:,0], 'A-A-At-At':gs[0:3,1],\
         'B-A-At-Bt':gs[3:, 1], 'A-A-At-Bt':gs[0:3, 2], 'A-A-At-A-B':gs[3:, 2]}
    
    ts = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)']
    
    j = 0
    for stack in ps:
        if stack in ['A-At', 'A-A-At', 'B-A-At']:
            scale = 1.5
        else:
            scale = 1.
        scale = scale*0.35
        ax = plt.subplot(ps[stack])
        lys = stack.split('-')
        for i in range(len(lys)):
            label = '$'+lys[i]+'$' if 't' not in lys[i] else '$\widetilde{'+lys[i][0]+'}$'
            ax.plot([0,0.5],[i*scale,i*scale], color='black', linewidth=2.0)
            ax.text(0.52, i*scale, label, verticalalignment='center', fontsize=16)
        ax.set_ylim(0,3)
        ax.set_axis_off()
        ax.text(0.2, (i+1)*scale, ts[j], verticalalignment='center', fontsize=16)
        ax.tick_params(axis="y",direction="in", pad=-22, labelsize=0.)
        ax.tick_params(axis="x",direction="in", pad=-15, labelsize=0.)
        j = j + 1
    plt.show()
    #plt.savefig('multilayer.pdf', bbox_inches='tight',pad_inches=0)

def without_AAtld():
    fig = plt.figure()
    gs = GridSpec(2, 3)
    gs.update(top=1, bottom=0., hspace=0.0, wspace=0.2)
    ps = {'A-A-At':gs[0,0], 'B-A-At':gs[1,0], 'A-A-At-At':gs[0,1],\
         'B-A-At-Bt':gs[1, 1], 'A-A-At-Bt':gs[0, 2], 'A-A-At-A-B':gs[1, 2]}
    
    ts = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)']
    
    j = 0
    cs = {'$A$':'black','$B$':'grey','$\widetilde{A}$':'red','$\widetilde{B}$':'orange'}
    for stack in ps:
        scale = 1.
        scale = scale*0.25
        ax = plt.subplot(ps[stack])
        lys = stack.split('-')
        tit = ''
        for i in range(len(lys)):
            label = '$'+lys[i]+'$' if 't' not in lys[i] else '$\widetilde{'+lys[i][0]+'}$'
            tit=tit+label
            ax.plot([0,0.5],[i*scale,i*scale], color=cs[label])
            ax.text(0.52, i*scale, label, verticalalignment='center',color=cs[label])
        
        ax.set_ylim(-0.1,2.3)
        ax.set_axis_off()
        #ax.text(0.2, (i+1)*scale, ts[j], verticalalignment='center')
        ax.text(0.13, (i+1)*scale, tit, verticalalignment='center', horizontalalignment='left')
        ax.tick_params(axis="y",direction="in", pad=-22, labelsize=0.)
        ax.tick_params(axis="x",direction="in", pad=-15, labelsize=0.)
        j = j + 1
    plt.show()
    #plt.savefig('multilayer.pdf', bbox_inches='tight',pad_inches=0.1)
if __name__=='__main__':
    without_AAtld()
