from tBG.crystal.diagonalize import diag_k_path
from tBG.crystal.structures import  CommensuStruct
from tBG.crystal.diagonalize import diag_k_path, plot_band

cs = CommensuStruct()
cs.make_structure(1,2) ## 22, 23 for angle 1.47 degree
cs.add_hopping_pz()
# cs.hoppings 
diag_k_path(cs)
plot_band(show=True)
