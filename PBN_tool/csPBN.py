import networkx as nx
import matplotlib.colors as mcolors
import random
import os
import boolean.boolean as bool
from BN import *


class csPBN():

    def __init__(self,list_of_nodes,consitutent_BN_functions,constituent_BN_edge_colors=None):

        self.numConsitutentBNs = len(consitutent_BN_functions)
        
        self.bool_algebra = bool.BooleanAlgebra()

        if constituent_BN_edge_colors is None:
            color_names = random.choices(list(mcolors.CSS4_COLORS.keys()),k=self.numConsitutentBNs)
            self.constituent_BN_edge_colors = [mcolors.CSS4_COLORS[color] for color in color_names]
        else:
            self.constituent_BN_edge_colors = constituent_BN_edge_colors

        self.constituentBNs = []
        for consitutent_BN_funcs in consitutent_BN_functions:
            self.constituentBNs.append(BN(list_of_nodes,consitutent_BN_funcs))

        self.list_of_nodes = []
        for node in list_of_nodes:
            self.list_of_nodes.append(self.bool_algebra.Symbol(node))

        self.num_nodes = len(self.list_of_nodes)


    def getNeighborStates(self,state):
        neighborStates = []

        for bn in self.constituentBNs:
            neighborStates.append(bn.getNeighborStates(state))

        return neighborStates
    

    def generateStateTransitionGraph(self):

        G = nx.MultiDiGraph()

        for initial_state_int in range(2**self.num_nodes):
            initial_state = bin2state(int2bin(initial_state_int,self.num_nodes))

            list_of_neighbor_states = self.getNeighborStates(initial_state)
            
            for i,neighbor_states in enumerate(list_of_neighbor_states):
                edges_aux = []
                for ns in neighbor_states:
                    edges_aux.append((state2bin(initial_state), state2bin(ns)))
                G.add_edges_from(edges_aux,color=self.constituent_BN_edge_colors[i])

        return G
    

    def getAttractors(self):
        Gpbn = self.generateStateTransitionGraph()

        return nx.attracting_components(Gpbn)


    def getAttractorStates(self):
        
        attractor_states_bin = set()
        
        Gpbn_bsccs = self.getAttractors()

        for bscc in Gpbn_bsccs:
            attractor_states_bin.update(bscc)

        return attractor_states_bin


    def getConstituentAttractors(self):
        
        attractors = dict()

        for i,bn in enumerate(self.constituentBNs):
            G = nx.DiGraph()
            edges_aux = []

            for source_state_int in range(2**self.num_nodes):
                source_state = bin2state(int2bin(source_state_int,self.num_nodes))

                neighbor_states = bn.getNeighborStates(source_state)

                for ns in neighbor_states:
                    edges_aux.append((state2bin(source_state), state2bin(ns)))
                
                G.add_edges_from(edges_aux)

                bsccs = list(nx.attracting_components(G))

                attractors['BN'+str(i)] = bsccs

        return attractors
    
    
    def getConstituentAttractorStates(self):

        attractor_states_bin = set()

        attractors = self.getConstituentAttractors()

        for attractor_list in attractors.values():
            for attr in attractor_list:
                attractor_states_bin.update(attr)

        return attractor_states_bin
    
    
    def getReachableAttractors(self,states,cBN_index,withTransientStates=False):
        return self.constituentBNs[cBN_index].getReachableAttractors(states,withTransientStates)


    def generateConstituentAttractorsGraph(self):

        G = nx.DiGraph()

        constituentBNsAttractors = []

        for bn in self.constituentBNs:
            constituentBNsAttractors.append(bn.getAttractors())

        cBNsAttractorsDict = dict()
        print("Attractors of the individual constituent BNs:")
        for bn_ind,cBNAttr in enumerate(constituentBNsAttractors):
            for key in cBNAttr:
                cBNsAttractorsDict[f"BN{bn_ind}_{key}"] = cBNAttr[key]
                print(f"BN{bn_ind}_{key}: {cBNAttr[key]}")

        for source_bn_ind in range(self.numConsitutentBNs):
            for target_bn_ind in range(self.numConsitutentBNs):
                if source_bn_ind != target_bn_ind:
                    edges_aux = []

                    target_bn = self.constituentBNs[target_bn_ind]
                    
                    for attr_key in constituentBNsAttractors[source_bn_ind].keys():
                        source_attractor_name = "BN" + str(source_bn_ind) + '_' + attr_key

                        source_attractor = constituentBNsAttractors[source_bn_ind][attr_key]
                        source_attractor_states = set([bin2state(state_bin) for state_bin in source_attractor])

                        reachable_target_bn_attractors = target_bn.getReachableAttractors(source_attractor_states)
                        for reachable_attr in reachable_target_bn_attractors:
                            target_attractor_name = "BN" + str(target_bn_ind) + '_' + reachable_attr
                            edges_aux.append((source_attractor_name,target_attractor_name))
                    
                    G.add_edges_from(edges_aux,color=self.constituent_BN_edge_colors[target_bn_ind])

        return G, cBNsAttractorsDict
    

    def drawStateTransitionGraph(self, folder, filename, highlightConstitutentAttractors=False):

        Gpbn = self.generateStateTransitionGraph()
        # Dpbn is an instant of the PyGraphviz.AGraph class
        Dpbn = nx.drawing.nx_agraph.to_agraph(Gpbn)

        if highlightConstitutentAttractors:

            constituent_attractors = self.getConstituentAttractors()

            colors = mcolors.CSS4_COLORS
            by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name) for
                            name, color in colors.items())
            color_names = [name for hsv, name in by_hsv]
            color_start_ind = color_names.index('palegreen')
        
            # Modify node fillcolor and edge color.
            Dpbn.node_attr.update(color='darkblue', style='filled', fillcolor='chartreuse')
            Dpbn.edge_attr.update(arrowsize=1)
            color_ind = color_start_ind

            for BN_key in constituent_attractors.keys():
                color_ind = random.choice(range(color_start_ind,len(color_names)))
                for bscc in constituent_attractors[BN_key]:
                    for state in bscc:
                        n = Dpbn.get_node(state)
                        n.attr['fillcolor']=colors[color_names[color_ind]]
                #color_ind = (color_ind + 1) % len(color_names)

        pos = Dpbn.layout('dot')
        if not os.path.exists(folder):
            os.makedirs(folder)
        # Dpbn is an instant of the PyGraphviz.AGraph class
        Dpbn.draw(os.path.join(folder, filename))


    def drawConstituentAttractorsGraph(self, folder, filename):
        G_PBN_attr, pbnAttrDict = self.generateConstituentAttractorsGraph()
        D_PBN_attr = nx.drawing.nx_agraph.to_agraph(G_PBN_attr)
        D_PBN_attr.node_attr.update(color='darkblue', style='filled', fillcolor='yellow')
        D_PBN_attr.edge_attr.update(arrowsize=1)
        pos = D_PBN_attr.layout('dot')

        if not os.path.exists(folder):
            os.makedirs(folder)
        D_PBN_attr.draw(os.path.join(folder, filename))