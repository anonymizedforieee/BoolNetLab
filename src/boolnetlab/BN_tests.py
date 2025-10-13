import os
import random
import networkx as nx
import utils
from utils import *
from BN import *
from csPBN import *

PATH = os.path.join("OneDrive - IDEAS NCBR sp. z.o.o","Research_projects",
                    "DeepRL-4-ReCell","Analysis","PBN_Analysis_Tool")

# Simple BN
list_of_nodes = ['x0','x1','x2','x3','x4']
BN_functions = ['x0|(!x1)', '(!x2)&(x1|x0)', '(!x3)&x4', '(x0)&x4', 'x2|(x3)']

bn = BN(list_of_nodes, BN_functions)

G_struct_BN = bn.getStructureGraph()
D_struct_BN = nx.drawing.nx_agraph.to_agraph(G_struct_BN)
pos = D_struct_BN.layout('dot')

folder = os.path.join(PATH, 'Structure_graphs')
filename = 'BN_struct.png'

if not os.path.exists(folder):
    os.makedirs(folder)
# D_struct_BN is a instant of the PyGraphviz.AGraph class
D_struct_BN.draw(os.path.join(folder, filename))

print("BN strong articulation nodes:")
print(bn.getStrongArticulationPoints())

print("BN attractors:")
print(bn.getAttractors())

print("Attractors without updating node x1:")
print(bn.getAttractors(excludedNodes=['x1']))

seed_states = set()
#seed_states.add(bin2state("01000"))
#seed_states.add(bin2state("11011"))
seed_states.add(bin2state("11111"))

print(bn.getComplementaryStates({"10000"}, "x1"))

attr_x1excl = list(bn.getAttractors(excludedNodes=['x1']).values())
attr_states = set()
for attr in attr_x1excl:
    for s in attr:
        attr_states.add(s)
print(attr_states)

cstates_x1 = bn.getComplementaryStates(attr_states, 'x1')
print(cstates_x1)

print(bn.getAttractorsFromStates(cstates_x1))

# Gbn = bn._generatePartialStateTransitionGraph({'00101'})
# Dbn = nx.drawing.nx_agraph.to_agraph(Gbn)
# pos = Dbn.layout('dot')
# Dbn.draw('stg.png')

# print(bn.getAttractorsFromStates({'00101'}))