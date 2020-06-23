scale = 8
params = {'backend': 'ps',
          'axes.labelsize': 4*scale,
          'axes.linewidth': 0.3*scale,
          'font.size': 3*scale,
          'font.weight': 'normal',
          'legend.fontsize': 3*scale,
          'xtick.labelsize': 4*scale,
          'ytick.labelsize': 4*scale,
          'xtick.major.pad': 8,
          'ytick.major.pad': 8,
          'axes.labelpad': 8,
          'text.usetex': True,
          'figure.figsize': [12,8],
          'lines.markersize': 1*scale,
          'lines.linewidth':  0.3*scale,
          'font.family' : 'Times New Roman',
          'mathtext.fontset': 'stix'
          }

# the six stacks studied in project multilayer graphene quasicrystal
stacks = ['AAAt','BAAt', 'AAAtAt', 'BAAtBt','AAAtBt','AAAtAB']
# R value of the round disk consisting of around 10 million sites
Rs = [678, 678, 589, 589, 589, 526]
# sample with supercell size of the 15/26 approximant of 30TBG contaning 10 million sites
SCs = [50, 50, 43, 43, 43, 39]
