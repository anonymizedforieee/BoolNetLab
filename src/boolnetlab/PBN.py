import networkx as nx
import matplotlib.colors as mcolors
import random
import os
import boolean.boolean as bool
from utils import *
import copy
import itertools


class PBN():

    def __init__(self,structure_dict,mode='asynchronous'):
        
        self.bool_algebra = bool.BooleanAlgebra()

        if mode not in ['asynchronous','synchronous']:
            raise ValueError(f"Wrong update scheme: {self.mode}")
            
        self.mode=mode

        list_of_nodes = structure_dict.keys()
        self.num_nodes = len(list_of_nodes)

        self.node_names = list_of_nodes

        self.list_of_nodes = []
        for node in list_of_nodes:
            self.list_of_nodes.append(self.bool_algebra.Symbol(node))

        self.functions = []
        for node in list_of_nodes:
            bool_funs = structure_dict[node]

            aux_list = []

            for (bool_fun, prob) in bool_funs:
                aux_list.append((self.bool_algebra.parse(bool_fun,simplify=True), prob))

            self.functions.append(aux_list)


    def getNeighborStates(self, state, excludedNodes = []):
        neighborStates = []

        excludedNodesIndices = [self.node_names.index(exclNode) for exclNode in excludedNodes]

        substitutions = dict()
        for i,node_value in enumerate(state):
            substitutions[self.list_of_nodes[i]] = self.bool_algebra.TRUE if node_value == 1 else self.bool_algebra.FALSE

        if self.mode == 'asynchronous':
            # Asynchronous update scheme
            for node_index,bool_funs in enumerate(self.functions):

                if node_index not in excludedNodesIndices:
                
                    for (bf, pr) in bool_funs:
                        
                        new_node_value = 1 if bf.subs(substitutions,simplify=True) == self.bool_algebra.TRUE else 0

                        new_state = list(state)
                        new_state[node_index] = new_node_value

                        neighborStates.append(tuple(new_state))

        elif self.mode == 'synchronous':
            # Synchronous update scheme
            possible_new_node_values = []

            for node_index,bool_funs in enumerate(self.functions):

                if node_index not in excludedNodesIndices:
                
                    possible_new_values = set()

                    for (bf, pr) in bool_funs:

                        possible_new_values.add(1 if bf.subs(substitutions,simplify=True) == self.bool_algebra.TRUE else 0)
                
                else:

                    possible_new_values = set(state[node_index])

                possible_new_node_values.append(possible_new_values)

            neighborStates = list(itertools.product(*possible_new_node_values))

        else:
            raise ValueError(f"Wrong update scheme: {self.mode}")
        
        return set(neighborState for neighborState in neighborStates)


    def generateStateTransitionGraph(self, excludedNodes = []):
        G = nx.DiGraph()

        for initial_state_int in range(2**self.num_nodes):
            initial_state = bin2state(int2bin(initial_state_int, self.num_nodes))

            neighbor_states = self.getNeighborStates(initial_state, excludedNodes)
            
            edges_aux = []
            for ns in neighbor_states:
                edges_aux.append((state2bin(initial_state), state2bin(ns)))
            G.add_edges_from(edges_aux)

        return G
    

    def _generatePartialStateTransitionGraph(self, initial_states_bin, excludedNodes = []):
        G = nx.DiGraph()

        states_to_process = set()
        processed_states = set()

        for initial_state_bin in initial_states_bin:
            states_to_process.add(bin2state(initial_state_bin))

        while len(states_to_process) > 0:
            source_state = states_to_process.pop()

            processed_states.add(source_state)

            neighbor_states = self.getNeighborStates(source_state, excludedNodes)
            
            edges_aux = []
            for ns in neighbor_states:
                edges_aux.append((state2bin(source_state), state2bin(ns)))
            G.add_edges_from(edges_aux)

            states_to_process = states_to_process.union(neighbor_states.difference(processed_states))

        return G


    def drawStateTransitionGraph(self, folder, filename, highlightAttractors=True):

        print("Drawing the state transition graph ...")

        Gpbn = self.generateStateTransitionGraph()
        # Dpbn is a instant of the PyGraphviz.AGraph class
        Dpbn = nx.drawing.nx_agraph.to_agraph(Gpbn)

        if highlightAttractors:
            attractors = self.getAttractors()

            colors = mcolors.CSS4_COLORS
            by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name) for
                            name, color in colors.items())
            color_names = [name for hsv, name in by_hsv]
            color_start_ind = color_names.index('palegreen')

            # Modify node fillcolor and edge color.
            Dpbn.node_attr.update(color='darkblue', style='filled', fillcolor='chartreuse')
            Dpbn.edge_attr.update(arrowsize=1)
            color_ind = color_start_ind

            for BN_key in attractors.keys():
                color_ind = random.choice(range(color_start_ind,len(color_names)))
                for state in attractors[BN_key]:
                    n = Dpbn.get_node(state)
                    n.attr['fillcolor']=colors[color_names[color_ind]]
                #color_ind = (color_ind + 1) % len(color_names)

        pos = Dpbn.layout('dot')
        if not os.path.exists(folder):
            os.makedirs(folder)
        # Dpbn is a instant of the PyGraphviz.AGraph class
        Dpbn.draw(os.path.join(folder, filename))

        print(f"Done. The state transition graph is saved to {os.path.join(folder, filename)}")


    def getAttractors(self, excludedNodes=[]):
        GBn = self.generateStateTransitionGraph(excludedNodes)

        attractors = dict()
        for attractor_index,attractor in enumerate(nx.attracting_components(GBn)):
            attractors['A'+str(attractor_index)] = attractor

        return attractors
    

    def getAttractorsFromStates(self, initial_states_bin, excludedNodes=[]):

        GBn = self._generatePartialStateTransitionGraph(initial_states_bin, excludedNodes)

        return list(nx.attracting_components(GBn))
    

    # Get the set of attractors reachable from the given set 
    # of states.
    def getReachableAttractors(self, states, withTransientStates=False):
        attractors = self.getAttractors()

        attractor_states = set()
        state2attr_dict = dict()
        
        for attractor_key in attractors.keys():
            attractor = attractors[attractor_key]
            attractor_states.update(attractor)

            for attrState in attractor:
                state2attr_dict[attrState] = attractor_key
        
        reachableAttractors = set()
        current_states = set(states)
        processed_states = set()
        while len(current_states) != 0:
            next_states = set()
            
            for state in current_states:
                next_states.update(self.getNeighborStates(state))
                processed_states.add(state)
                
            next_states_bin = set([state2bin(s) for s in next_states])
            for state_bin in next_states_bin.intersection(attractor_states):
                reachableAttractors.add(state2attr_dict[state_bin])
                next_states_bin.remove(state_bin)


            current_states = set([bin2state(s) for s in next_states_bin]).difference(processed_states)

        if withTransientStates:
            transient_states_bin = set([state2bin(s) for s in processed_states]).difference(attractor_states)
            return reachableAttractors, transient_states_bin 
        
        return reachableAttractors
    

    # def getStructureGraph(self):

    #     G = nx.DiGraph()

    #     edges_aux = []
    #     for i, f in enumerate(self.functions):
    #         for s in f.symbols:
    #             edges_aux.append((s, self.list_of_nodes[i]))
    #     G.add_edges_from(edges_aux)

    #     return G
    
    
    # def getStrongArticulationPoints(self):
    #     # Inefficient implementation

    #     G = self.getStructureGraph()

    #     num_sccs = nx.number_strongly_connected_components(G)

    #     strong_articulation_points = []
    #     for i, node in enumerate(self.list_of_nodes):
    #         G = self.getStructureGraph()
    #         G.remove_node(node)
    #         if nx.number_strongly_connected_components(G) > num_sccs:
    #             strong_articulation_points.append(self.node_names[i])

    #     return strong_articulation_points


    # def getComplementaryStates(self, states_bin : set, node: str) -> set:
    #     complementary_states = set()

    #     # Get the index of the node
    #     node_index = self.node_names.index(node)

    #     # Get the Boolean function for the node
    #     fun = self.functions[node_index]

    #     for state_bin in states_bin:

    #         state = bin2state(state_bin)
            
    #         substitutions = dict()
    #         for i, node_value in enumerate(state):
    #             substitutions[self.list_of_nodes[i]] = self.bool_algebra.TRUE if node_value == 1 else self.bool_algebra.FALSE

    #         new_node_value = 1 if fun.subs(substitutions,simplify=True) == self.bool_algebra.TRUE else 0

    #         new_state = list(state)
    #         new_state[node_index] = new_node_value

    #         complementary_states.add(state2bin(tuple(new_state)))

    #     return complementary_states
