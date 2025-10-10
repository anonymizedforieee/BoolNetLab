import os
import random
import networkx as nx
import utils
from utils import *
from BN import *
from csPBN import *
from PBN import *

import matplotlib.pyplot as plt
import numpy as np

PATH = os.path.join("OneDrive - IDEAS NCBR sp. z.o.o","Research_projects",
                    "DeepRL-4-ReCell","Analysis","PBN_Analysis_Tool")

# === NETWORK DEFINITIONS ================================================

# ---- Bladder BN model ----

bladder_bn_nodes_GBDQ = [
    'v_AKT', 'v_ATM_b1', 'v_ATM_b2', 'v_Apoptosis_b1', 'v_Apoptosis_b2', 'v_CDC25A', 'v_CHEK1_2_b1', 'v_CHEK1_2_b2', 'v_CyclinA',
    'v_CyclinD1', 'v_CyclinE1', 'v_DNAdamage', 'v_E2F1_b1', 'v_E2F1_b2', 'v_E2F3_b1', 'v_E2F3_b2', 'v_EGFR', 'v_EGFR_stimulus',
    'v_FGFR3', 'v_FGFR3_stimulus', 'v_GRB2', 'v_GrowthInhibitors', 'v_Growth_Arrest', 'v_MDM2', 'v_PI3K', 'v_PTEN', 'v_Proliferation',
    'v_RAS', 'v_RB1', 'v_RBL2', 'v_SPRY', 'v_TP53', 'v_p14ARF', 'v_p16INK4a', 'v_p21CIP'
]

bladder_bn_nodes_CABEAN = [
   'v_AKT',
   'v_ATM_b1',
   'v_ATM_b2',
   'v_Apoptosis_b1',
   'v_Apoptosis_b2',
   'v_CDC25A',
   'v_CHEK1_2_b1',
   'v_CHEK1_2_b2',
   'v_CyclinA',
   'v_CyclinD1',
   'v_CyclinE1',
   'v_DNAdamage',
   'v_E2F1_b1',
   'v_E2F1_b2',
   'v_E2F3_b1',
   'v_E2F3_b2',
   'v_EGFR',
   'v_EGFR_stimulus',
   'v_FGFR3',
   'v_FGFR3_stimulus',
   'v_GRB2',
   'v_GrowthInhibitors',
   'v_Growth_Arrest',
   'v_MDM2',
   'v_PI3K',
   'v_PTEN',
   'v_Proliferation',
   'v_RAS',
   'v_RB1',
   'v_RBL2',
   'v_SPRY',
   'v_TP53',
   'v_p14ARF',
   'v_p16INK4a',
   'v_p21CIP',
]

bladder_bn_funs_CABEAN = [
   'v_PI3K', # v_AKT
   '(((~v_DNAdamage&v_ATM_b1)&v_ATM_b2)|v_DNAdamage)', # v_ATM_b1
   '((v_DNAdamage&v_E2F1_b1)&v_ATM_b1)', # v_ATM_b2
   '((((((((~v_Apoptosis_b1&~v_E2F1_b1)&v_TP53)|(((~v_Apoptosis_b1&v_E2F1_b1)&~v_E2F1_b2)&v_TP53))|((~v_Apoptosis_b1&v_E2F1_b1)&v_E2F1_b2))|(((v_Apoptosis_b1&~v_Apoptosis_b2)&~v_E2F1_b1)&v_TP53))|((((v_Apoptosis_b1&~v_Apoptosis_b2)&v_E2F1_b1)&~v_E2F1_b2)&v_TP53))|(((v_Apoptosis_b1&~v_Apoptosis_b2)&v_E2F1_b1)&v_E2F1_b2))|(v_Apoptosis_b1&v_Apoptosis_b2))', # v_Apoptosis_b1
   '((v_Apoptosis_b1&v_E2F1_b1)&v_E2F1_b2)', # v_Apoptosis_b2
   '((((~v_E2F1_b1&v_E2F3_b1)&~v_RBL2)&~v_CHEK1_2_b1)|((v_E2F1_b1&~v_RBL2)&~v_CHEK1_2_b1))', # v_CDC25A
   '(((~v_ATM_b1&v_CHEK1_2_b1)&v_CHEK1_2_b2)|v_ATM_b1)', # v_CHEK1_2_b1
   '((v_E2F1_b1&v_ATM_b1)&v_CHEK1_2_b1)', # v_CHEK1_2_b2
   '(((((~v_E2F1_b1&v_E2F3_b1)&v_CDC25A)&~v_RBL2)&~v_p21CIP)|(((v_E2F1_b1&v_CDC25A)&~v_RBL2)&~v_p21CIP))', # v_CyclinA
   '((((~v_RAS&~v_p16INK4a)&~v_p21CIP)&v_AKT)|((v_RAS&~v_p16INK4a)&~v_p21CIP))', # v_CyclinD1
   '(((((~v_E2F1_b1&v_E2F3_b1)&v_CDC25A)&~v_RBL2)&~v_p21CIP)|(((v_E2F1_b1&v_CDC25A)&~v_RBL2)&~v_p21CIP))', # v_CyclinE1
   'v_DNAdamage', # v_DNAdamage
   '(((((((((~v_RAS&~v_E2F1_b1)&v_E2F3_b1)&~v_RB1)&~v_RBL2)|(((((~v_RAS&v_E2F1_b1)&~v_E2F1_b2)&v_E2F3_b1)&~v_RB1)&~v_RBL2))|((~v_RAS&v_E2F1_b1)&v_E2F1_b2))|(((v_RAS&~v_E2F1_b1)&~v_RB1)&~v_RBL2))|((((v_RAS&v_E2F1_b1)&~v_E2F1_b2)&~v_RB1)&~v_RBL2))|((v_RAS&v_E2F1_b1)&v_E2F1_b2))', # v_E2F1_b1
   '((((((((((~v_RAS&v_E2F1_b1)&v_E2F3_b1)&v_E2F3_b2)&~v_RB1)&~v_RBL2)&v_ATM_b1)&v_ATM_b2)&v_CHEK1_2_b1)&v_CHEK1_2_b2)|(((((((v_RAS&v_E2F1_b1)&~v_RB1)&~v_RBL2)&v_ATM_b1)&v_ATM_b2)&v_CHEK1_2_b1)&v_CHEK1_2_b2))', # v_E2F1_b2
   '(((((~v_RAS&v_E2F3_b1)&v_E2F3_b2)|((v_RAS&~v_E2F3_b1)&~v_RB1))|(((v_RAS&v_E2F3_b1)&~v_E2F3_b2)&~v_RB1))|((v_RAS&v_E2F3_b1)&v_E2F3_b2))', # v_E2F3_b1
   '((((v_RAS&v_E2F3_b1)&~v_RB1)&v_CHEK1_2_b1)&v_CHEK1_2_b2)', # v_E2F3_b2
   '((((~v_EGFR_stimulus&~v_FGFR3)&~v_GRB2)&v_SPRY)|((v_EGFR_stimulus&~v_FGFR3)&~v_GRB2))', # v_EGFR
   'v_EGFR_stimulus', # v_EGFR_stimulus
   '((v_FGFR3_stimulus&~v_EGFR)&~v_GRB2)', # v_FGFR3
   'v_FGFR3_stimulus', # v_FGFR3_stimulus
   '((((~v_EGFR&v_FGFR3)&~v_GRB2)&~v_SPRY)|v_EGFR)', # v_GRB2
   'v_GrowthInhibitors', # v_GrowthInhibitors
   '((((~v_RB1&~v_RBL2)&v_p21CIP)|(~v_RB1&v_RBL2))|v_RB1)', # v_Growth_Arrest
   '(((((~v_RB1&~v_ATM_b1)&~v_TP53)&~v_p14ARF)&v_AKT)|(((~v_RB1&~v_ATM_b1)&v_TP53)&~v_p14ARF))', # v_MDM2
   '((v_RAS&~v_PTEN)&v_GRB2)', # v_PI3K
   'v_TP53', # v_PTEN
   '((~v_CyclinE1&v_CyclinA)|v_CyclinE1)', # v_Proliferation
   '((((~v_EGFR&~v_FGFR3)&v_GRB2)|(~v_EGFR&v_FGFR3))|v_EGFR)', # v_RAS
   '(((~v_CyclinD1&~v_CyclinE1)&~v_CyclinA)&~v_p16INK4a)', # v_RB1
   '(~v_CyclinD1&~v_CyclinE1)', # v_RBL2
   'v_RAS', # v_SPRY
   '(((((~v_E2F1_b1&v_ATM_b1)&v_CHEK1_2_b1)&~v_MDM2)|((((v_E2F1_b1&~v_E2F1_b2)&v_ATM_b1)&v_CHEK1_2_b1)&~v_MDM2))|((v_E2F1_b1&v_E2F1_b2)&~v_MDM2))', # v_TP53
   'v_E2F1_b1', # v_p14ARF
   '(v_GrowthInhibitors&~v_RB1)', # v_p16INK4a
   '((((~v_GrowthInhibitors&~v_CyclinE1)&v_TP53)&~v_AKT)|((v_GrowthInhibitors&~v_CyclinE1)&~v_AKT))', # v_p21CIP
]

def gbdqState2cabeanState(state):
    
    return tuple([state[bladder_bn_nodes_GBDQ.index(node)] for node in bladder_bn_nodes_CABEAN])


def gbdqState2assapbnState(state):
    
    return tuple([state[bladder_bn_nodes_GBDQ.index(node)] for node in bladder_bn_nodes_assapbn])


# === MODEL ANALYSIS =====================================================
      
print("--- BN model of the bladder network ---")
bladder_bn = BN(bladder_bn_nodes_CABEAN, bladder_bn_funs_CABEAN)

input_nodes = bladder_bn.getInputNodeNames()
num_input_nodes = len(input_nodes)
print(f"Input nodes: {input_nodes}")
print(f"Number of input nodes: {num_input_nodes}")

# Pseudo-attractor states found by PASIP (in the order of GBDQ)
s_gbdq = [
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1),
    (0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1),
    (1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0),
    (0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1),
    (0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0),
    (0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1)
]

print(f"Number of pseudo-attractor states found by PASIP in GBDQ: {len(s_gbdq)}")

s_gbdq_set = set(s_gbdq)
print(f"Number of unique pseudo-attractor states found by PASIP in GBDQ: {len(s_gbdq_set)}")

if (len(s_gbdq) != len(s_gbdq_set)):
    print(f"\033[93mWARNING: The pseudo-attractor states found by PASIP of GBDQ are not unique!\033[0m")
    s_gbdq_aux = []
    for el in s_gbdq:
        if el not in s_gbdq_aux:
            s_gbdq_aux.append(el)
    s_gbdq = s_gbdq_aux

# for i in range(len(s)):
#     if i not in [5,9,24,26,33]:
#         print(gbdqState2assapbnState(s[i]))

# i = bladder_bn_nodes_CABEAN.index(bladder_bn_nodes_assapbn[23])
# print(bladder_bn_funs_CABEAN[i])
# i_cycD1 = bladder_bn_nodes_assapbn.index('v_CyclinD1')
# i_cycE1 = bladder_bn_nodes_assapbn.index('v_CyclinE1')
# print((gbdqState2assapbnState(s[0])[i_cycD1], gbdqState2assapbnState(s[0])[i_cycE1]))

# print(bladder_bn_funs_CABEAN[29])
# print(gbdqState2cabeanState(s[0]))
# print((bladder_bn_nodes_CABEAN[9],bladder_bn_nodes_CABEAN[10]))

# print(gbdqState2cabeanState(s[0])[9])
# print(gbdqState2cabeanState(s[0])[10])

fixpoints = []
non_fixpoints = []

for i in range(len(s_gbdq)):
    s_cabean = gbdqState2cabeanState(s_gbdq[i])
    if not bladder_bn.isFixPointAttractor(s_cabean):
        print(f"State of index {i} is not a fixpoint state.")
        non_fixpoints.append(i)
    else:
        fixpoints.append(i)

print(f"Fixpoint pseudo-attractor states: {fixpoints}\n")

### === Computing attractors reachable from the non-fixpoint pseudo-attractor states found by PASIP ===

non_fixpoint_states = [state2bin(gbdqState2cabeanState(s_gbdq[i])) for i in non_fixpoints]

complex_attractors = []

# for i, non_fixpoint_state in zip(non_fixpoints, non_fixpoint_states):
    
#     print(f"Checking attractor(s) reachable from non-fixpoint state at index {i}: ({non_fixpoint_state})_CABEAN ...")
#     attrs = bladder_bn.getAttractorsFromStates([non_fixpoint_state])
#     assert(len(attrs) == 1)
#     print(f"Size of the reachable attractor: {len(attrs[0])}")
#     assert(non_fixpoint_state in attrs[0])
#     print(f"The attractor contains the following set of non-fixpoint states: {set(non_fixpoint_states).intersection(attrs[0])}\n")

#     complex_attractors.append(attrs)


### === Checking consistency of PASIP found pseudo-attractor states with the specified initial condition ===

# Checking which of the pseudo-attractor states found by PASIP are consistent with the initial condition
# specified in Research_projects\DeepRL-4-ReCell\GraphBDQ_paper\models\models_my_init_settings\bladder_FGFR3_stimulus_true.ispl

FGFR3_stimulus_ind = bladder_bn_nodes_CABEAN.index('v_FGFR3_stimulus')

attractor_states_with_FGFR3_stimulus_active = list(filter(lambda i: gbdqState2cabeanState(s_gbdq[i])[FGFR3_stimulus_ind] == 1, range(len(s_gbdq))))

#attractor_states_with_FGFR3_stimulus_active = []
#for i in range(len(s_gbdq)):
#    if gbdqState2cabeanState(s_gbdq[i])[FGFR3_stimulus_ind] == 1:
#        attractor_states_with_FGFR3_stimulus_active.append(i)

print(f"Number of pseudo-attractor states found by PASIP with FGFR3_stimulus=true: {len(attractor_states_with_FGFR3_stimulus_active)}")
print(f"Pseudo-attractor states found by PASIP with FGFR3_stimulus=true: {attractor_states_with_FGFR3_stimulus_active}")


# attrs = bladder_bn.getAttractorsFromStates([state2bin(s[24])])
# assert(len(attrs) == 1)
# print(f"Size of complex attractor: {len(attrs[0])}")
# assert(state2bin(s[24]) in attrs[0])
# print(not_fixpoint_states.intersection(attrs[0]))

# attrs = bladder_bn.getAttractorsFromStates([state2bin(s[26])])
# assert(len(attrs) == 1)
# print(f"Size of complex attractor: {len(attrs[0])}")
# assert(state2bin(s[26]) in attrs[0])
# print(not_fixpoint_states.intersection(attrs[0]))

# attrs = bladder_bn.getAttractorsFromStates([state2bin(s[33])])
# assert(len(attrs) == 1)
# print(f"Size of complex attractor: {len(attrs[0])}")
# assert(state2bin(s[33]) in attrs[0])
# print(not_fixpoint_states.intersection(attrs[0]))

# attrs = bladder_bn.getAttractorsFromStates([state2bin(s[9])])
# assert(len(attrs) == 1)
# print(f"Size of complex attractor: {len(attrs[0])}")
# assert(state2bin(s[9]) in attrs[0])
# print(not_fixpoint_states.intersection(attrs[0]))


# attrs = bn_cd4.getAttractorsFromStates([state2bin(s)])
# if len(attrs) != 1:
#     print("The given state is not a fix-point attractor!")
# else:
#     attr = attrs[0]
#     if len(attr) != 1:
#         print("The given state is not a fix-point attractor!")
#     else:
#         attr_state_str = attr.pop()
#         print(attr_state_str)
#         print(state2bin(s))
#         assert(attr_state_str == state2bin(s))

#attr = bladder_bn.getAttractors()
#print(attr)


### ====================================================================================

### Result log ###

# --- BN model of the bladder network ---
# Number of pseudo-attractor states found by PASIP in GBDQ: 34
# Number of pseudo-attractor states found by PASIP in GBDQ: 25
# WARNING: The pseudo-attractor states found by PASIP of GBDQ are not unique!
# State of index 5 is not a fixpoint state.
# State of index 9 is not a fixpoint state.
# State of index 15 is not a fixpoint state.
# State of index 17 is not a fixpoint state.
# State of index 24 is not a fixpoint state.
# Fix points: [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23]

# Checking attractor(s) reachable from non-fixpoint state at index 5: (10000000000000100100111110010110010)_CABEAN ...
# Size of the reachable attractor: 512
# The attractor contains the following set of non-fixpoint states: {'10000000000000100100111110010110010'}

# Checking attractor(s) reachable from non-fixpoint state at index 9: (10000100111010100100100010110010100)_CABEAN ...
# Size of the reachable attractor: 184320
# The attractor contains the following set of non-fixpoint states: {'10000100111010100100100010110010100'}

# Checking attractor(s) reachable from non-fixpoint state at index 15: (01010010000100000100101001011111001)_CABEAN ...
# Size of the reachable attractor: 16
# The attractor contains the following set of non-fixpoint states: {'01010010000100000100101001011111001'}

# Checking attractor(s) reachable from non-fixpoint state at index 17: (01010010000100100100111001010111011)_CABEAN ...
# Size of the reachable attractor: 32
# The attractor contains the following set of non-fixpoint states: {'01010010000100100100111001010111011'}

# Checking attractor(s) reachable from non-fixpoint state at index 24: (01010010000100000100111001011111001)_CABEAN ...
# Size of the reachable attractor: 16
# The attractor contains the following set of non-fixpoint states: {'01010010000100000100111001011111001'}

# Number of pseudo-attractor states found by PASIP with FGFR3_stimulus=true: 14
# Pseudo-attractor states found by PASIP with FGFR3_stimulus=true: [1, 3, 6, 7, 8, 10, 11, 12, 13, 14, 18, 19, 21, 23]

# Fixpoint attractors found by CABEAN
CABEAN_results = [
    '0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-1-0-0-0-0-0-1-1-0-0-0-0-0',
    '0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-1-1-0-0-0-0-0-0-1-0-0-0-1-1',
    '0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-1-1-0-0-0-0-0-1-1-0-0-0-0-1',
    '0-0-0-0-0-1-0-0-1-1-1-0-1-0-1-0-0-0-1-1-0-0-0-0-0-0-1-1-0-0-1-0-1-0-0',
    '0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-1-1-0-1-1-0-0-0-0-1-1-1-1-0-0-0-1',
    '0-0-0-0-0-0-0-0-0-0-0-0-0-0-1-0-0-0-1-1-0-1-1-0-0-0-0-1-0-1-1-0-0-1-1',
    '0-0-0-0-0-1-0-0-1-0-1-0-1-0-1-0-0-0-1-1-0-1-0-0-0-0-1-1-0-0-1-0-1-1-0',
    '0-0-0-0-0-1-0-0-1-1-1-0-1-0-1-0-0-1-1-1-0-0-0-0-0-0-1-1-0-0-1-0-1-0-0',
    '0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-1-1-1-0-1-1-0-0-0-0-1-1-1-1-0-0-0-1',
    '0-0-0-0-0-0-0-0-0-0-0-0-0-0-1-0-0-1-1-1-0-1-1-0-0-0-0-1-0-1-1-0-0-1-1',
    '0-0-0-0-0-1-0-0-1-0-1-0-1-0-1-0-0-1-1-1-0-1-0-0-0-0-1-1-0-0-1-0-1-1-0',
    '0-1-0-1-0-0-1-0-0-0-0-1-0-0-0-0-0-0-0-0-0-0-1-0-0-1-0-0-1-1-0-1-0-0-1',
    '0-1-0-1-0-0-1-0-0-0-0-1-0-0-0-0-0-0-0-0-0-1-1-0-0-1-0-0-0-1-0-1-0-1-1',
    '0-1-0-1-0-0-1-0-0-0-0-1-0-0-0-0-0-0-0-0-0-1-1-0-0-1-0-0-1-1-0-1-0-0-1',
    '0-1-0-1-0-0-1-0-0-0-0-1-0-0-0-0-0-0-1-1-0-0-1-0-0-1-0-1-1-1-1-1-0-0-1',
    '0-1-0-1-0-0-1-0-0-0-0-1-0-0-0-0-0-0-1-1-0-1-1-0-0-1-0-1-1-1-1-1-0-0-1',
    '0-1-0-1-0-0-1-0-0-0-0-1-0-0-1-0-0-0-1-1-0-1-1-0-0-1-0-1-0-1-1-1-0-1-1',    
    '0-1-0-1-0-0-1-0-0-0-0-1-0-0-0-0-0-1-1-1-0-0-1-0-0-1-0-1-1-1-1-1-0-0-1',
    '0-1-0-1-0-0-1-0-0-0-0-1-0-0-0-0-0-1-1-1-0-1-1-0-0-1-0-1-1-1-1-1-0-0-1',
    '0-1-0-1-0-0-1-0-0-0-0-1-0-0-1-0-0-1-1-1-0-1-1-0-0-1-0-1-0-1-1-1-0-1-1'
]

CABEAN_results_set = set(CABEAN_results)

assert len(CABEAN_results) == len(CABEAN_results_set), "\033[31mCABEAN returns dublicates!!!\033[0m"

gbdq_fixpoint_states = []
for fp in fixpoints:
    s_cabean = gbdqState2cabeanState(s_gbdq[fp])

    # Convert a PBN_analysis_tool state to a string representation in the CABEAN output format
    s_cabean_str = str(s_cabean[0])
    for i in range(1,len(s_cabean)):
        s_cabean_str = s_cabean_str + '-' + str(s_cabean[i])

    gbdq_fixpoint_states.append(s_cabean_str)

#if set(gbdq_fixpoint_states) == CABEAN_results_set:
#    print(f"\tPASIP and CABEAN find the same set of {len(CABEAN_results)} fixpoint attractor states.")
#else:
#    print(f"\t\033[31mPASIP and CABEAN fixpoint attractor states differ!\033[0m")

print(f"Number of fixpoint pseudo-attractor states found by PASIP: {len(gbdq_fixpoint_states)}")
if set(gbdq_fixpoint_states).issubset(CABEAN_results_set):
    print(f"All fixpoint pseudo-attractor states found by PASIP are among CABEAN found fixpoint attractors number of which is {len(CABEAN_results)}.")


def __state_matches_template(state, template):
    
    if len(state) != len(template):
        return False
    
    for i, value in enumerate(state):
        if i % 2 == 0 and template[i] != '-':
            if value != template[i]:
                return False

    return True


CABEAN_complex_attractor_states = [
    #======================== find attractor #1 : 512 states ========================
    #: 27 nodes 1 leaves 512 minterms
    '--0-0-0-0-0-0-0-0-0-0-0-0-0---0---1-0-0---1-1-----0-0---0-1---0-0-1--',
    '0-1-0-1-0-0-1-0-0-0-0-1-0-0-0-0---1-0-0---0-1-0-0-1-0---1-1---1-0-0-1',
    '0-1-0-1-0-0-1-0-0-0-0-1-0-0-0-0---1-0-0---1-1-0-0-1-0---1-1---1-0-0-1',
    '0-1-0-1-0-0-1-0-0-0-0-1-0-0---0---1-0-0---1-1-0-0-1-0---0-1---1-0-1-1'
]

non_fixpoint_pa_states = []
for pa_state_ind in range(len(s_gbdq)):

    if pa_state_ind not in fixpoints:
        s_cabean = gbdqState2cabeanState(s_gbdq[pa_state_ind])

        # Convert a PBN_analysis_tool state to a string representation in the CABEAN output format
        s_cabean_str = str(s_cabean[0])
        for i in range(1,len(s_cabean)):
            s_cabean_str = s_cabean_str + '-' + str(s_cabean[i])

        non_fixpoint_pa_states.append((pa_state_ind, s_cabean_str))


complex_attractor_pa_states_indices = []
for nfpas_ind, nfpas in non_fixpoint_pa_states:
    for ccas in CABEAN_complex_attractor_states:
        if __state_matches_template(nfpas, ccas):
            complex_attractor_pa_states_indices.append(nfpas_ind)

print(f"Indices of pseudo-attractor states that belong to a complex attractor found by CABEAN: {complex_attractor_pa_states_indices}")

spurious_pseudo_attractor_states = set(range(len(s_gbdq))) - set(fixpoints) - set(complex_attractor_pa_states_indices)

print(f"Set of spurious pseudo-attractor states indices: {spurious_pseudo_attractor_states}")

# CABEAN failed to find attractors for the environmental condition EGFR_stimulus = true and the ramining input nodes set to false
# 
# for spas in spurious_pseudo_attractor_states:
#     print(f"Checking whether pseudo-attractor state of index {spas} is spurious ...")
    
#     # Getting the environmental condition
#     env_cond = {}
#     for input_node in input_nodes:
#         env_cond[input_node] = s_gbdq[spas][bladder_bn_nodes_GBDQ.index(input_node)]
#     print(f"\tEnvironmental condition for the pseudo-attractor state of index {spas}: {env_cond}")
    
#     spas_cabean = gbdqState2cabeanState(s_gbdq[spas])
#     attrs = bladder_bn.getAttractorsFromStates([spas_cabean])
#     print(f"\tNumber of attractors: {len(attrs)}")
#     for i, attr in enumerate(attrs):
#         print(f"\tSize of attractor {i}: {len(attr)}")
#         if state2bin(spas_cabean) in attr:
#             print(f"\tPseudo-attractor state of index {spas} is not spurious: attractor {i} contains it.")

# Output
# Number of fixpoint pseudo-attractor states found by PASIP: 20
# All fixpoint pseudo-attractor states found by PASIP are among CABEAN found fixpoint attractors number of which is 20.
# Indices of pseudo-attractor states that belong to a complex attractor found by CABEAN: [5, 15, 17, 24]
# Set of spurious pseudo-attractor states indices: {9}
# Checking whether pseudo-attractor state of index 9 is spurious ...
#         Number of attractors: 1
#         Size of attractor 0: 184320
#         Pseudo-attractor state of index 9 is not spurious: attractor 0 contains it.

# Create histogram

extra_pa_state = s_gbdq[list(spurious_pseudo_attractor_states)[0]]
#traj_gen = bladder_bn.trajectory_statistics(extra_pa_state, 10000)

home_path = os.environ.get("HOME")
bladder_bn.simulate(init_state=extra_pa_state,
                    num_steps=20,
                    pert_prob=0.01,
                    file_path=os.path.join(home_path, "Boolean-networks", "PBN_tool", "trajectory"))

# for _ in range(3):
#     statistics = next(traj_gen)
#     num_non_zero_counts = len(statistics)

#     cut_off = 1000

#     print(f"Number of attractor states with non-zero revisit count: {num_non_zero_counts}")
#     print(f"Max revisits: {np.max(list(statistics.values()))}")

#     extra_pa_state_value = statistics.pop(extra_pa_state)

#     values = np.array(sorted(list(statistics.values()), reverse=True))
#     values = values[values >= cut_off]
#     values = np.concatenate([np.array([extra_pa_state_value]), values])

#     fig, ax = plt.subplots(figsize=(10, 4))
#     x = list(range(len(values)))
#     ax.bar(x, values, width=0.8)
#     ax.set_ylabel('Number of revisits')
#     ax.set_xlabel('State index')
#     xticks = list(range(0,len(statistics),1000))
#     #ax.set_xticks(xticks)
#     #ax.set_xticklabels(xticks)

#     #plt.savefig('bladder_attractor_hist.pdf')

#     plt.show()
