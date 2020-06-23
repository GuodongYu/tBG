import numpy as np
from tBG.periodic_bilayer_graphene import Structure, BandUnfold


def plot_BZs(M, hop_dist=2.0):
    s = Structure(M)
    s.add_hopping(max_dist=hop_dist)
    bduf = BandUnfold(s)
    bduf._plot_BZs(path=True, text=True, dots=True, a_quarter=False, D_show=True)


if __name__=='__main__':
    plot_BZs(41)
