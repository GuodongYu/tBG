from tBG.TBG30_approximant.structure  import Structure

def plot_atop_vs_M():
    import tBG
    tBG.params['figure.figsize']=[12,8]
    tBG.params['font.size']=20
    tBG.params['lines.linewidth']=4
    at = []
    ms = []
    ns = []
    for i in range(3, 100):
        try:
            s = Structure()
            s.make_structure(i)
            at.append(s.a_top)
            ms.append(s.n_bottom)
            ns.append(s.n_top)
        except:
            pass
    from matplotlib import pyplot as plt
    plt.rcParams.update(tBG.params)
    plt.scatter(ms, at, clip_on=False)
    for i in range(len(ms)):
        plt.text(ms[i], at[i], '%s/%s' % (ms[i],ns[i]))
    plt.axhline(2.456, ls='dashed', color='blue')
    plt.xlabel('M')
    plt.xticks(range(0,100,10))
    plt.ylabel(r'$\mathrm{a_t(\AA)}$')
    plt.ylim(min(at)-0.004, max(at)+0.004)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('atop_M.pdf', bbox_inches='tight', pad_inches=0.02)

if __name__=='__main__':
    plot_atop_vs_M()
