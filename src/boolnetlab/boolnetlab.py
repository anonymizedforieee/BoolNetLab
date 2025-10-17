import networkx as nx
import matplotlib.colors as mcolors
from absl.logging import flush
from markdown_it.common.html_blocks import block_names
from matplotlib import pyplot as plt
from pyvis.network import Network
import boolean as bool
from collections import deque, defaultdict
from sympy.codegen.ast import continue_
from sympy.logic.boolalg import truth_table

import bang

import utils
from utils import *
import copy
import os
import random
import sympy
import time
from functools import cache

try:
    import dd.cudd as bdd
    CUDD_LOADED = True
except ImportError:
    print("""WARNING: The dd.cudd module of the dd package, which provides Cython bindings to the CUDD library in C,
         is not compiled and cannot be loaded. Please refer to https://github.com/tulip-control/dd for
         installation instructions. Falling back to dd.autoref, which wraps the pure-Python Binary
         Decision Diagrams implementation dd.bdd."""
    )
    import dd.autoref as bdd
    CUDD_LOADED = False

from dd.autoref import Function
from tqdm import tqdm
import sys

class BN_Realisation:
    __bool_algebra = bool.BooleanAlgebra()
    __bdd = bdd.BDD()
    _V_INDEX = 0
    _V_LOWLINK_INDEX = 1
    _V_ON_STACK_INDEX = 2
    _FIRST_RULE = 0
    _RECURSION_LIMIT = 1000000
    _GET_SCC = 0
    _GET_CONTROL_NODES = 1
    _CONVERTED_TRUE_IN_BOOLEAN_PY_TO_STRING = "1"
    _CONVERTED_FALSE_IN_BOOLEAN_PY_TO_STRING = "0"
    _FIRST_VARIABLE_NAME_IN_ISPL_FILE = 0

    @staticmethod
    def __generateRandomBooleanFunction(parents):
        is_fixed = False

        # Random truth_table.
        num_of_ones = random.randint(0, 2 ** len(parents))
        minterms = random.sample(range(2 ** len(parents)), num_of_ones)

        sym = sympy.symbols(parents)
        bool_formula = sympy.logic.SOPform(sym, minterms, [])
        fun = str(bool_formula)
        if fun == 'True' or fun == 'False':
            fun = fun.upper()
            is_fixed = True
            num_ps = 0
        else:
            num_ps = len(bool_formula.atoms())

        return fun, is_fixed, num_ps


    def getInputNodeNames(self):

        input_nodes = []

        for i, fun in enumerate(self.functions):
            syms = fun.symbols

            # node = TRUE / FALSE
            # if (len(syms) == 0):
            #    input_nodes.append(self.node_names[i])

            # A node depends only on itself.
            if (len(syms) == 1):
                if str(syms.pop()) == self.node_names[i]:
                    input_nodes.append(self.node_names[i])

        return input_nodes

    """
    Dodać description (Andrzej).
    """
    def save_ispl(self, filepath: str, initial_condition: str = None, verbose: bool = True) -> None:

        if verbose:
            print(f"Exporting the Boolean network to the {filepath} file in the .ispl format ... ", end='')

        # Creating folders if necessary
        if filepath[-1] == '\\' or filepath[-1] == '/':
            filepath = filepath[:-1]

        folder = os.path.dirname(filepath)
        filename = os.path.basename(filepath)

        if folder != '' and not os.path.exists(folder):
            os.makedirs(folder)
            if verbose:
                print(f"Created folder {folder}.")

        with open(filepath, 'w') as file:

            file.write('Agent M\n')

            file.write('\tVars:\n')
            for node_name in self.node_names:
                file.write('\t\t' + node_name + ' : boolean;\n')
            file.write('\tend Vars\n')

            file.write('\tActions = {none};\n')
            file.write('\tProtocol:\n')
            file.write('\t\tOther: {none};\n')
            file.write('\tend Protocol\n')

            file.write('\tEvolution:\n')
            for node_name, fun in zip(self.node_names, self.functions):
                if str(fun) == '1':
                    fun_str = '(' + node_name + '|~' + node_name + ')'
                elif str(fun) == '0':
                    fun_str = '(' + node_name + '&~' + node_name + ')'
                else:
                    fun_str = str(fun)
                file.write('\t\t' + node_name + '=true if ' + fun_str + '=true;\n')
                file.write('\t\t' + node_name + '=false if ' + fun_str + '=false;\n')
            file.write('\tend Evolution\n')

            file.write('end Agent\n')
            file.write('\n')

            file.write('InitStates\n')
            if initial_condition is None:
                input_node_names = self.getInputNodeNames()
                if len(input_node_names) == 0:
                    # Any value for the first node
                    file.write(f'\t\tM.{self.node_names[0]}=false or M.{self.node_names[0]}=true;\n')
                else:
                    initial_condition = ''
                    for input_node_name in input_node_names:
                        initial_condition += 'M.' + input_node_name + "=false and "
                    if len(initial_condition) > 0:
                        initial_condition = initial_condition[:-5]
                    file.write('\t\t' + initial_condition + ';\n')
            else:
                file.write(initial_condition + '\n')

            file.write('end InitStates\n')

        if verbose:
            print('done.')


    """
     This function reads variables, update functions from the ispl file.

     Args:
        path_to_ispl_file (str): string representing path to the ispl BN model.
     Returns:
        BN_Realisation object which has nodes and functions from the ispl file.
    """
    @classmethod
    def load_ispl(cls, path_to_ispl_file: str):
        if not os.path.isfile(path_to_ispl_file):
            raise FileNotFoundError(path_to_ispl_file)

        BN_variables = []
        BN_functions = []

        with open(path_to_ispl_file, "r") as ispl_file:
            line = ispl_file.readline()
            while line:
                while line.strip() != "Vars:":
                    line = ispl_file.readline()

                line = ispl_file.readline()

                while line.strip() != "end Vars":
                    line = line.strip()
                    gene_name = line.split(':')[0]
                    BN_variables.append(gene_name.strip())
                    line = ispl_file.readline()

                while line.strip() != "Evolution:":
                    line = ispl_file.readline()

                line = ispl_file.readline()

                while line.strip() != "end Evolution":
                    line = ispl_file.readline()
                    line = line.strip()
                    line = line.split(" if ")[1]
                    line = line.split("=")[0]
                    BN_functions.append(line.strip())
                    line = ispl_file.readline()

                while line.strip() != "InitStates":
                    line = ispl_file.readline()
                break

        assert len(BN_variables) == len(BN_functions), "The number of nodes does not match the number of Boolean functions."

        print(f"Loaded a Boolean network of {len(BN_variables)} nodes.")

        return BN_Realisation(BN_variables, BN_functions)


    @classmethod
    def generate_random_bn(cls, num_nodes: int, max_parent_nodes: int, min_parent_nodes: int = 1,
                           allow_self_loops: bool = True, allow_input_nodes: bool = True, mode: str = 'asynchronous'):

        assert num_nodes > 0, "Number of nodes must be at least 1!"

        if max_parent_nodes > num_nodes:
            if allow_self_loops:
                print(
                    "WARNING: Maximum number of parent nodes cannot be greater than the total number of nodes. Setting max_parent_nodes to the total number of nodes.")
                max_parent_nodes = num_nodes
            else:
                print(
                    "WARNING: Maximum number of parent nodes cannot be greater than the total number of nodes. Additionally, self loops are not allowed. Setting max_parent_nodes to one less than the total number of nodes.")
                max_parent_nodes = num_nodes - 1

        print(f"Generating a random Boolean network with {num_nodes} nodes ... ", end='')

        if not allow_input_nodes:
            if min_parent_nodes == 0:
                print(
                    f"WARNING: Input nodes are not allowed since allow_input_nodes is set to False. Thus, each node must have at least one parent node. Setting min_parent_nodes to 1.")
                min_parent_nodes = 1

        assert max_parent_nodes >= min_parent_nodes

        list_of_nodes = ['x' + str(i) for i in range(num_nodes)]
        input_node_indices = set()
        bn_functions = []

        for i in range(num_nodes):
            num_parents = random.randint(min_parent_nodes, max_parent_nodes)

            if allow_self_loops:
                potential_parent_nodes = list_of_nodes
            else:
                potential_parent_nodes = copy.copy(list_of_nodes)
                potential_parent_nodes.pop(i)

            parents = sorted(random.sample(potential_parent_nodes, num_parents))

            if len(parents) == 0:

                fun = 'x' + str(i)  # This will be an input node, so its initial should remain unchanged
                input_node_indices.add(i)

            else:

                fun, is_constant, num_ps = cls.__generateRandomBooleanFunction(parents)
                # Make sure that the node truly depends on at least the specified minimum number of parent nodes
                # and, if required, make sure that there are no input nodes.
                # ToDo: INEFFICIENT implementation!!!
                incorrect_function = True
                while incorrect_function:
                    fun, is_constant, num_ps = cls.__generateRandomBooleanFunction(parents)
                    incorrect_function = ((not allow_input_nodes) and fun == 'x' + str(i)) or (
                            num_ps < min_parent_nodes)

                if fun == 'x' + str(i):
                    # We have x_i(t+1) = x_i(t), so this is an input node
                    input_node_indices.add(i)

            bn_functions.append(fun)

        print("done.")
        print("=============================================================================================")
        print(f"A Boolean network with {num_nodes} nodes is generated. The Boolean functions are as follows:")
        for i, f in enumerate(bn_functions):
            print(f"{list_of_nodes[i]} = {f}")
        print(f"Number of fixed-value nodes: {len(input_node_indices)}")
        print("=============================================================================================")

        bn = BN_Realisation(list_of_nodes, bn_functions, mode)

        return bn


    """
    Generates a state transition graph for the Boolean network.

     Args:
        excludeNodes (list[str], optional): A list of node names whose updates will be excluded 
        during the construction of the state transition graph.

     Returns:
        networkx.Graph: A NetworkX graph object representing the state transition graph.
    """
    def generateStateTransitionGraph(self, excludedNodes: list[str] = []) -> nx.Graph:
        G = nx.DiGraph()

        for initial_state_int in range(2 ** self.num_nodes):
            initial_state = bin2state(int2bin(initial_state_int, self.num_nodes))

            neighbor_states = self.getNeighborStates(initial_state, excludedNodes)

            edges_aux = []
            for ns in neighbor_states:
                edges_aux.append((state2bin(initial_state), state2bin(ns)))
            G.add_edges_from(edges_aux)

        return G
    

    """
        Returns the set of neighboring states for a given state under the current update mode of the Boolean network.

        Args:
            
            state (tuple[int, ...]): The state for which the neighboring states will be computed.
            
            excludeNodes (list[str], optional): A list of node names whose updates will be excluded 
                when computing the neighboring states.

        Returns:
            set[tuple[int, ...]]: A set of neighboring states.
    """
    def getNeighborStates(self, state: tuple[int, ...], excludedNodes: list[str] = []) -> set[tuple[int, ...]]:
        neighborStates = []

        excludedNodesIndices = [self.node_names.index(exclNode) for exclNode in excludedNodes]

        substitutions = dict()
        for i, node_value in enumerate(state):
            substitutions[
                self.list_of_nodes[i]] = self.__bool_algebra.TRUE if node_value == 1 else self.__bool_algebra.FALSE

        if self.mode == 'asynchronous':
            # Asynchronous update scheme
            for node_index, fun in enumerate(self.functions):
                if node_index not in excludedNodesIndices:
                    new_node_value = 1 if fun.subs(substitutions, simplify=True) == self.__bool_algebra.TRUE else 0

                    new_state = list(state)
                    new_state[node_index] = new_node_value

                    neighborStates.append(new_state)

        elif self.mode == 'synchronous':
            # Synchronous update scheme
            new_state = list(state)

            for node_index, fun in enumerate(self.functions):
                if node_index not in excludedNodesIndices:
                    new_node_value = 1 if fun.subs(substitutions, simplify=True) == self.__bool_algebra.TRUE else 0
                    new_state[node_index] = new_node_value

            neighborStates.append(new_state)

        return set(tuple(neighborState) for neighborState in neighborStates)


    """
    This method extracts attractors from their compressed BDD representations and converts them into an explicit form.

     Args:
        
        all_attractors: A list of BDDs representing the individual attractors. The BDDs encode only attractor states
                        without transitions between them.
     
     Returns:
        A dictionary where each key represents an attractor and each value is a set of its states. The keys are strings
        of the form 'Ai', where i is a consecutive number starting from 0, identifying each attractor.
    
    """
    def _enumerate_attractors(self, all_attractors: list[Function]) -> dict[str, set[str]]:

        attractors = dict()

        for i, at in enumerate(all_attractors):
            attractor = set()
            for models in self.__bdd.pick_iter(at, care_vars=self.node_names):
                state_str = ''
                for node_name in self.node_names:
                    state_str += '1' if models[node_name] else '0'
                attractor.add(state_str)

            attractors['A' + str(i)] = attractor

        return attractors


    """
        Parameters
        ----------
        filepath: Path to the file where the state transition graph image is saved. The file extension defines the format.
        Available formats are: 'canon', 'cmap', 'cmapx', 'cmapx_np', 'dia', 'dot', 'fig', 'gd', 'gd2',
        'gif', 'hpgl', 'imap', 'imap_np', 'ismap', 'jpe', 'jpeg', 'jpg', 'mif', 'mp', 'pcl', 'pdf', 'pic',
        'plain', 'plain-ext', 'png', 'ps', 'ps2', 'svg', 'svgz', 'vml', 'vmlz', 'vrml', 'vtx', 'wbmp',
        'xdot', 'xlib' (note that not all may be available on every system depending on how Graphviz was built).
        
        layout: Layout to be used for visualisation of the state transition graph. Available layouts are:
        'neato', 'dot' (default), 'twopi', 'circo', 'fdp', and 'nop'.
    
        highlight_attractors: Indicates whether the attractors should be highlighted with colors. If set to True,
            the non-attractor states are colored by default with the chartreuse color from the mcolors.CSS4_COLORS
            palette (this can be changed with setting the transient_state_color option), while states of an attractor
            are colored by a randomly selected color from the palette or by a color from the provided color_names list.
    
        use_bdds: If True (default), the attractors are computed with symbolic operations on Binary Decision Diagrams.
            Otherwise, NetworkX attracting_components function is used. This option is ignored if highlight_attractors
            is False.
    
        color_names: An optional list of color names from the mcolors.CSS4_COLORS palette to be used for coloring
            attractors. Colors are used one after another to color the attractors. If the number of color names is
            smaller than the number of attractors, more than one attractor will have the same color. At least one color
            name must be provided. This option is ignored if highlight_attractors is False.
    
        transient_state_color: Name of the color in the mcolors.CSS4_COLORS palette to be used for coloring non-attractor
            states. The default value is 'chartreuse'. This option is ignored if highlight_attractors is False.

        selected_state_groups (list[list[tuple[int, ...]]]): A list of groups of states to be highlighted in the graph.
            Each group is a list of states.

        selected_group_colors (list[str]): Color names for each group of selected states. The number of colors must be
            same as the number of groups in selected_state_groups.
    
        Returns
        -------
        None
    """
    def draw_state_transition_graph(self,
                                    filepath: str,
                                    layout: str = 'dot',
                                    highlight_attractors: bool = True,
                                    use_bdds: bool = True,
                                    color_names: list[str] = None,
                                    transient_state_color: str = 'chartreuse',
                                    selected_state_groups: list[list[tuple[int, ...]]] = [],
                                    selected_group_colors: list[str] = [],
                                    ) -> None:
        RANDOM_COLORS = color_names is None

        if not RANDOM_COLORS:
            assert len(color_names) > 0, "The number of provided color names must be at least 1!"

        print("Drawing the state transition graph ...")

        Gbn = self.generateStateTransitionGraph()
        # Dpbn is a instant of the PyGraphviz.AGraph class
        Dbn = nx.drawing.nx_agraph.to_agraph(Gbn)

        if highlight_attractors:

            if use_bdds:
                attractors = self._enumerate_attractors(self.find_all_attractors())
            else:
                attractors = self.getAttractors()

            colors = mcolors.CSS4_COLORS

            # Modify node fillcolor and edge color.
            Dbn.node_attr.update(color='darkblue', style='filled', fillcolor=transient_state_color)
            Dbn.edge_attr.update(arrowsize=1)

            if RANDOM_COLORS:
                by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name) for
                                name, color in colors.items())
                color_names = [name for hsv, name in by_hsv]
                color_start_ind = color_names.index('palegreen')

            for i, BN_key in enumerate(attractors.keys()):
                # Select a color for the next attractor
                if RANDOM_COLORS:
                    color_ind = random.choice(range(color_start_ind, len(color_names)))
                else:
                    color_ind = i % len(color_names)

                color = colors[color_names[color_ind]]

                # Color the attractor states
                for state in attractors[BN_key]:
                    n = Dbn.get_node(state)
                    n.attr['fillcolor'] = color

            if len(selected_state_groups) > 0:

                assert len(selected_state_groups) == len(selected_group_colors), "Wrong number of colors in selected_group_colors!"

                for color, selected_group in zip(selected_group_colors, selected_state_groups):
                    for selected_state in [utils.state2bin(selected_state) for selected_state in selected_group]:
                        n = Dbn.get_node(selected_state)
                        n.attr['fillcolor'] = color
        
        Dbn.layout(layout)

        # Creating folders if necessary
        if filepath[-1] == '\\' or filepath[-1] == '/':
            filepath = filepath[:-1]

        folder = os.path.dirname(filepath)
        filename = os.path.basename(filepath)

        if folder != '' and not os.path.exists(folder):
            os.makedirs(folder)

        # Dpbn is a instant of the PyGraphviz.AGraph class
        Dbn.draw(os.path.join(folder, filename))

        print(f"Done. The state transition graph is saved to {os.path.join(folder, filename)}")


    """
        This method constructs an interaction graph for the Boolean network, where each edge represents
        a dependence of the target node on the source node. The dependency is determined syntactically
        — that is, by checking whether the Boolean function of the target node includes the variable
        corresponding to the source node — and not semantically. In other words, the method does not
        verify whether the source node is an essential variable of the target node's Boolean function.
        
        Returns:
         A NetworkX graph object representing the state transition graph.

    """
    @cache
    def getStructureGraph(self):

        G = nx.DiGraph()

        for node in self.node_names:
            G.add_node(node)

        edges_aux = []
        for i, f in enumerate(self.functions):
            for s in f.symbols:
                # edges_aux.append((s, self.list_of_nodes[i]))
                edges_aux.append((str(s), self.node_names[i]))

        G.add_edges_from(edges_aux)

        return G


    """
        Plots structure graph of the BN

        Parameters
        ----------
        layout: Layout of the nodes to be used. Possible values: 'spring' (default), 'circular',
        'forceatlas', 'planar', 'random', 'shell', and 'spectral'.

        Returns
        -------
        None
    """
    def plot_structure_graph(self, layout: str = 'spring') -> None:
        struct_graph = self.getStructureGraph()

        match layout:
            case 'spring':
                pos = nx.spring_layout(struct_graph)
            case 'circular':
                pos = nx.circular_layout(struct_graph)
            case 'forceatlas':
                pos = nx.forceatlas2_layout(struct_graph)
            case 'planar':
                pos = nx.planar_layout(struct_graph)
            case 'random':
                pos = nx.random_layout(struct_graph)
            case 'shell':
                pos = nx.shell_layout(struct_graph)
            case 'spectral':
                pos = nx.spectral_layout(struct_graph)
            case _:
                print("Unknown layout specified. Using the default 'spring' layout.")
                pos = nx.spring_layout(struct_graph)

        nx.draw_networkx(struct_graph, pos=pos)

        # ax = plt.axes()
        # nx.drawing.nx_pylab.draw(struct_graph,ax=ax)

        plt.show()


    """
        Plots the structure graph of a Boolean network using PyGraphviz and save the image in a file of specified format provided by the file extension.
    
        Parameters
        ----------
        filepath: Path to the file where the structure graph image is saved. The file extension defines the format.
        Available formats are: 'canon', 'cmap', 'cmapx', 'cmapx_np', 'dia', 'dot', 'fig', 'gd', 'gd2',
        'gif', 'hpgl', 'imap', 'imap_np', 'ismap', 'jpe', 'jpeg', 'jpg', 'mif', 'mp', 'pcl', 'pdf', 'pic',
        'plain', 'plain-ext', 'png', 'ps', 'ps2', 'svg', 'svgz', 'vml', 'vmlz', 'vrml', 'vtx', 'wbmp',
        'xdot', 'xlib' (note that not all may be available on every system depending on how Graphviz was built).
    
        layout: Layout to be used for visualisation of the structure graph. Available layouts are:
        'neato', 'dot' (default), 'twopi', 'circo', 'fdp', and 'nop'.
    
        Returns
        -------
        None
    """
    def plot_structure_graph_pgv(self, filepath: str, layout: str = 'dot') -> None:
        struct_graph = nx.drawing.nx_agraph.to_agraph(self.getStructureGraph())
        struct_graph.layout(layout)

        # Creating folders if necessary
        if filepath[-1] == '\\' or filepath[-1] == '/':
            filepath = filepath[:-1]

        folder = os.path.dirname(filepath)
        filename = os.path.basename(filepath)

        if folder != '' and not os.path.exists(folder):
            os.makedirs(folder)

        # struct_graph is an instant of the PyGraphviz.AGraph class
        struct_graph.draw(os.path.join(folder, filename))


    """
        Makes an interactive graph of the BN structure in HTML format.

        Parameters
        ----------
        filepath: Path to the file where the structure graph image is saved. The file is in HTML format, hence the filename extension must be .html.
        node_size: Size of graph nodes (default 20).
        height: Height of the graph in pixels (default 500).
        width: Width of the graph in pixels (default 500).

        Returns
        -------
        None
    """
    def plot_structure_graph_interactive(self, filepath: str, node_size: int = 20, height: int = 500, width: int = 500) -> None:

        g = Network(height=height, width=width, directed=True, notebook=True, cdn_resources='in_line')
        g.toggle_hide_edges_on_drag(False)
        g.toggle_physics(True)  # Triggering the toggle_physics() method allows for more fluid graph interactions
        #g.barnes_hut()
        g.repulsion()
        # g.show_buttons(filter_=['nodes', 'edges', 'physics'])
        g.show_buttons(filter_=['physics'])

        # structure_graph = self.getStructureGraph()
        # g.from_nx(structure_graph, default_node_size=node_size, )

        for node_name in self.node_names:
            g.add_node(node_name, label=node_name, labelHighlightBold=False, shape='circle')

        for i, f in enumerate(self.functions):
            for s in f.symbols:
                g.add_edge(str(s), self.node_names[i], physics=False, title=str(f))

        # Creating folders if necessary
        if filepath[-1] == '\\' or filepath[-1] == '/':
            filepath = filepath[:-1]

        folder = os.path.dirname(filepath)
        filename = os.path.basename(filepath)

        # The filename extension must be .html
        name, extension = os.path.splitext(filename)
        if extension != '.html':
            if extension == '':
                print('Warning: Filename without extension. The required .html extension is added.')
            else:
                print('Warning: Inappropriate filename extension. Replacing it with .html')
            filename = name + '.html'

        if folder != '' and not os.path.exists(folder):
            os.makedirs(folder)

        path_str = os.path.join(folder, filename)
        g.show(path_str)

        print(f"Interactive structure graph saved in HTML format to {path_str}")


    """
        This function creates the BDD representation of the transition matrix T of the given block.

        Description:
        Assume the block has n variables (x_1, ..., x_n). Assume moreover two different
        boolean states: x = (x_1,...,x_n) and y = (y_1,...,y_n) of the BN. 
        Then T(x, y) = True if and only if you can go from x to y using BN in one move. 
        So T is the function of 2n variables. To understand the way how it is done,
        assume three nodes BN with the following functions:

        f1 = bdd.add_expr('(x1 & ~x2) | x3')
        f2 = bdd.add_expr('x2')
        f3 = bdd.add_expr('x1 & x2')

        In asynchronous mode we have in fact three different possibilities: 
        the node x_1 is updated or the node x_2 is updated or the node x_3 is updated.
        Assume that the node x_1 has to be updated. Then it means that x1_next = f_1, x2_next = x_2, x3_next = x_3.
        Thus for the node x_1 we create a boolean expression:            

        R_1(x1, x2, x3, x1_next, x2_next, x3_next) <=> ( ~(xor(x1_next, f1)) & ~(xor(x2_next, x2)) & ~(xor(x3_next, x3)) ).
        The expression ~xor(a,b) == True <=> a == b, so this is exactly what we modeled above.
        In the same way we define R_2, R_3 - boolean expression which describes the situation when only x_2, x_3 is
        changed. After that, it is clear that T is of the following form 

                                T = R_1 | R_2 | R_3

        Args: 
           nodes (list[str]): the list of block variables.
           nodes_next (list[str]): converted list above with "_next" ending.
           functions (list[Function]): list of the update functions of the block nodes.

        Returns: The BDD representation of the local transition matrix of the given block. 
    """
    def __create_bdd_from_boolean_network(self,
                                          nodes: list,
                                          nodes_next: list,
                                          functions: list[Function]):
        eq = lambda a, b: ~self.__bdd.apply('xor', a, b)

        # T represents the transition matrix of BN.
        T_loc = self.__bdd.false

        #print("Constructing BDD encoded transition system of the BN ...", end='', flush=True)
        #for i in tqdm(range(len(nodes))):
        for i in range(len(nodes)):
            # Initial conditions for R_i.
            R_i = self.__bdd.true

            for j in range(len(nodes)):
                if i != j:
                    # The case: xj_next = xj for j != i.
                    R_i &= eq(self.__bdd.var(nodes_next[j]), self.__bdd.var(nodes[j]))
                else:
                    # Here R_i represents the logic: xi_next = f_i.
                    R_i &= eq(self.__bdd.var(nodes_next[j]), functions[j])

            T_loc |= R_i
            
            if not CUDD_LOADED:
                self.__bdd.collect_garbage()

        #print('done.', flush=True)

        return T_loc


    """
        This function finds all SCC of the BN and prepare all technical containers needed in 
        find_all_attractors method.
        
        Returns:
            all_blocks (dict): dict holding the block_number -> (SCC_nodes : set, block_control_nodes : set).
    """
    def __create_blocks(self):
        G = self.getStructureGraph()
        SCC_generator = nx.strongly_connected_components(G)
        SCC = list(SCC_generator)
        self.number_of_blocks = len(SCC)
        blocks = dict()

        # Dict holding "node" -> "scc_numb" where it belongs.
        self.node_block_number = dict()
        for block_nr, scc in enumerate(SCC):
            # Technical.
            block_dict = dict([
                (node, block_nr) for node in scc
            ])

            # Node -> number of block.
            self.node_block_number.update(block_dict)

            # Creating block.
            control_nodes_for_scc = set()

            for scc_node in scc:
                parent_set = set(G.predecessors(scc_node))
                control_nodes_for_scc |= parent_set

            control_nodes_for_scc -= scc
            blocks[block_nr] = (SCC[block_nr], control_nodes_for_scc)

        self.block_parents = defaultdict(set)
        self.block_children = defaultdict(set)
        for block_nr, (_, control_nodes) in blocks.items():
            for control_node in control_nodes:
                # Update block_parents.
                parent_block = self.node_block_number[control_node]
                self.block_parents[block_nr].add(parent_block)

                self.block_children[parent_block].add(block_nr)

        print(f"After SCC we have the following blocks of the structure graph: {blocks}")

        return blocks


    """
        Computes the attractors by explicitly constructing the state transition graph.

        This method was originally introduced for testing the implementation of the symbolic, 
        BDD-based attractor computation method. It should be used only for small networks due to 
        its explicit and potentially memory-intensive nature.

        Args:
        
            excludeNodes (list[str], optional): A list of node names whose updates will be excluded 
                during the construction of the state transition graph.
     
        Returns:

            dict[str, list[tuple[int, ...]]]: A dictionary where each key represents an attractor and 
                each value is a list of its states. The keys are strings of the form 'Ai', where *i* is 
                a consecutive number starting from 0, identifying each attractor.
    """
    def getAttractors(self, excludedNodes: list[str] = []) -> dict[str, list[tuple[int, ...]]]:
        GBn = self.generateStateTransitionGraph(excludedNodes)

        attractors = dict()
        for attractor_index, attractor in enumerate(nx.attracting_components(GBn)):
            attractors['A' + str(attractor_index)] = attractor

        return attractors


    # @staticmethod
    # def getAttractorsMonteCarlo(file):
    #     pbn = bang.load_from_file(file, "assa")
    #     pbn._n_parallel = min(max(77, pbn.n_nodes * 10), 2 ** pbn.n_nodes - 1)
    #     pbn.device = "gpu"

    #     attractors = pbn.monte_carlo_detect_attractors(trajectory_length=1100, attractor_length=1300)
    #     print(attractors)


    def getAttractorsMonteCarlo(self, n_parallel=-1, burn_in_len=1100, history_len=1300, print_result=False):
        """
        This function uses MonteCarlo simulations to detect pseudoattractors.

        Args:
            n_parallel: number of parallel simulations.
            burn_in_len: lenght of trajectories that will be discarded as random and non relevant in detection.
            history_len: length of trajectories that will be used to classify states into pseudoattractors.

        Returns:
            list of pseudoattractor states
        """
        if n_parallel == -1:
            n_parallel = min(max(77, self.num_nodes * 10), 2 ** self.num_nodes - 1)
        var_indices = {var: i for i, var in enumerate(self.node_names)}
        parent_variables = [sorted([var_indices[var.__str__()] for var in f.symbols]) for f in self.functions]
        truth_tables = [[y for _, y in
                         truth_table(self.functions_str[i], [x.__str__() for x in self.functions[i].symbols])] for
                        i in range(self.num_nodes)]

        pbn = bang.PBN(self.num_nodes,
                       [1 for _ in range(self.num_nodes)],
                       [len(f.symbols) for f in self.functions],
                       truth_tables,
                       parent_variables,
                       [[1.] for _ in range(self.num_nodes)],
                       0.,
                       [],
                       n_parallel=n_parallel)
        pbn.device = "gpu"

        attractors = pbn.monte_carlo_detect_attractors(trajectory_length=burn_in_len, attractor_length=history_len)
        if print_result:
            print(attractors)

        attractors_dict = dict()
        for i, attractor in enumerate(attractors):
            attractors_dict['A'+str(i)] = set()
            for state in attractor:
                state_str = ''.join(['1' if val else '0' for val in state])
                attractors_dict['A'+str(i)].add(state_str)
        
        return attractors_dict
        

    """
    Constructor for the Boolean Network (BN_Realisation) class.

    :param list[str] list_of_nodes: A list of node names in the Boolean network.
    :param list[str] list_of_functions: A list of Boolean functions in the network. The order of the functions 
        must correspond to the order of the nodes.
    :param str mode: The update scheme for the Boolean network. Must be either 'asynchronous' or 'synchronous'.
    """
    def __init__(self, list_of_nodes: list[str],
                 list_of_functions: list[str], mode: str = "asynchronous"):
        if mode not in ['asynchronous', 'synchronous']:
            raise ValueError(f"Wrong update scheme: {mode}")

        # Mode of the model.
        self.mode = mode

        # Container holding names of genes and theirs scc block number they belong.
        self.node_block_number = None

        # Number of SCC's in the Boolean Network structure.
        self.number_of_blocks = None

        # Number of nodes.
        self.num_nodes = len(list_of_nodes)

        # Names of nodes.
        self.node_names = list_of_nodes

        # String representing the update rules.
        self.functions_str = list_of_functions

        # Holds the original update function before the edge_removing function.
        self.original_function = None

        # Holds the index of modified update function.
        self.old_index = -1

        # Holds bool_algebra.Symbols of nodes.
        self.list_of_nodes = []

        for node_name in list_of_nodes:
            node = self.__bool_algebra.Symbol(node_name)
            self.list_of_nodes.append(node)

        self.functions = []
        for fun in list_of_functions:
            self.functions.append(self.__bool_algebra.parse(fun, simplify=True))

        # Declare Boolean variables in the BDD manager
        # (current and next state variables for the Boolean Network)
        for node_name in list_of_nodes:
            self.__bdd.declare(node_name)
            self.__bdd.declare(node_name + "_next")

        # Adding BDD expressions representing boolean update functions.
        self.bdd_expressions = []

        # Creating a dict containing relation: (node_name : str -> update_function : bdd_Function).
        # This dict will be used for searching purpose.
        self.node_name_to_bdd_update_function = dict()

        for index, fun in enumerate(list_of_functions):
            self.bdd_expressions.append(self.__bdd.add_expr(fun))
            self.node_name_to_bdd_update_function[self.node_names[index]] = self.bdd_expressions[-1]

        self.next_variables = [
            var + "_next" for var in self.node_names
        ]

        # For the hybrid Tarjan algorithm.
        self.visited_map = dict()

        # Dicts for renaming in searching for the images.
        self.x_to_x_next = dict(zip(self.node_names, self.next_variables))
        self.x_next_to_x = dict(zip(self.next_variables, self.node_names))

        # Block decomposition of the BN.
        self.number_of_blocks = None
        self.node_block_number = None
        self.block_parents = None
        self.block_children = None

        # Blocks is a dict container. It holds the relation:
        # block_number -> (SCC : set of name_node, control_nodes : set).
        self.blocks = self.__create_blocks()


    # Technical function checking if the block is an elementary block,
    # i.e. if it does not have any parent blocks.
    def __is_elementary_block(self, all_parents):
        return True if all_parents == set() else False

 
    """
      For a given dictionary holding "node_name": boolean value
      this function will create a bdd logical expression wich corresponds to that state. For example 
      (1) models = {"x1": 1, "x2": 0, "x3": 0} the output will be bdd of the expressions: (x1 & ~x2 & ~x3)
      (2) models = {"x1": 1, "x2": 1, "x3": 1} the output will be bdd of the expression: (x1 & x2 & x3). 
      This function is needed in method "push_forward_with_BDD" to create initial
      set of bdd's corresponding to the states given by the user for further propagation 
      through the BDD describing whole Boolean Network. 

      Args:
          models (dict): dictionary representing the state which needs to be converted into bdd.

      Returns:
          Function: the bdd expressions corresponding to the states given in models.
    """
    def __bdd_representation_of_boolean_state(self, models: dict):
        # Initial value for the rule representing models dictionary.
        rule = self.__bdd.true

        # Go through dict representing a boolean state of some genes and add !x or x
        # basing on the value of x.
        for variable, value in models.items():
            if not value:
                rule = self.__bdd.apply("and", rule, ~self.__bdd.var(variable))
            else:
                rule = self.__bdd.apply("and", rule, self.__bdd.var(variable))
        return rule


    """
    For a given list of initial binary states this function will create to each of them 
    a bdd logical expression wich corresponds to that state. For example 
    Initial_Set = [[1, 0, 0], [1,1,1]]. Then the output will be the list 
    of bdd's of the expressions: (x1 & ~x2 & ~x3), (x1 & x2 & x3), i.e. array  
    [bdd(x1 & ~x2 & ~x3), bdd(x1 & x2 & x3)]. 
    This function is needed in method "push_forward_with_BDD" to create initial
    set of bdd's corresponding to the states given by the user for further propagation 
    through the BDD describing whole Boolean Network. 
 
    Args:
        Initial_Set (list[list[int], ...]): list of binary lists of the same length representing some Boolean states
                                            of the Boolean Network.
        direction (string):                 technical string which is "forward" if the output is served for 
                                            searching image of some state and "backward" for the pre image of 
                                            some Boolean state.
    Returns:
        list: list of bdd's expressions corresponding to the states given in 
              Initial_Set.
    """
    def __bdd_representation_of_boolean_states(self,
                                               Initial_Set: list[list[int]],
                                               direction="forward") -> list:
        if direction == "forward":
            variables = self.node_names
        elif direction == "backward":
            variables = self.next_variables
        else:
            raise ValueError("The direction argument must be 'forward' or 'backward'")

        expressions = []

        for state in Initial_Set:
            # Initial condition for the rule which will correspond to
            # the fixed state in Initial_Set.
            rule = self.__bdd.true

            # Go through bits and if the bit is 1 then add to the rule operand "& var_name".
            # If bit is zero then add to the rule operand "&~var_name.
            # Note: dd library allows us to use different method for that, but it is much safer
            # to do it with bdd.apply method as shown below. This method allows one to use for
            # example 'xor' operators while other methods in dd library does not.
            for i, bit in enumerate(state):
                if bit == 0:
                    rule = self.__bdd.apply('and', rule, ~self.__bdd.var(variables[i]))
                else:
                    rule = self.__bdd.apply('and', rule, self.__bdd.var(variables[i]))
            expressions.append(rule)

        return expressions


    """
        This function will find some attractor state reachable from the node v. This is the 
        main method for the "find_some_attractor_state_from_fixed_state" method below. 
        This is in fact the standard Tarjan algorithm with the following modifications:
        (1) We find the first SCC which turns out to be BSCC (bottom strongly connected coponent).
        In other words - the first SCC found by Tarjan DFS algorithm is an attractor of BN.
        (2) We work on BDD's here. Thus the current state v is not a boolean vector but its BDD
        representation.     

        Important: The BDD representation does not improve the Tarjan algorithm here! The only
        improvment is that one need just to find first SCC and then the algorithm is straightforward
        with BDD (see find_all_attractors method for further details).

        Args: 
            v (Function): the BDD representation of current state.
            T_loc: the BDD representing of transition matrix of the block.
            index (list[int]): is the one-element list which holds the index in 
                               standard Tarjan algorithm. The list simulates C pointers.
            attractor_node (list[Function]): Once an attractor state is found this list will keep it.
                                             Until an attractor state is unknown - this list remains
                                             to be empty.
            all_values: list of nodes representing nodes which currently are in the processed block. 
    """
    def __hybrid_tarjan_recursive(self, v: Function, T_loc: Function, index: list, attractor_node: list,
                                  all_values: list):
        if attractor_node:
            return

        v_index = index[0]
        v_lowlink = index[0]
        v_onStack = True
        self.visited_map[v] = [v_index, v_lowlink, v_onStack]
        index[0] += 1

        # Find all pairs (x,x_next) satisfying pre logic.
        # It means that x has to be v and x_next has to be
        # connected with x in BN.
        pre = v & T_loc

        # Once all pairs are found use the exist operator for
        # taking x_next from the pair (x, x_next)
        post = self.__bdd.exist(all_values, pre)

        # x_next is the BDD represented in the x_next variables
        # and so one need to rename them for the future usage.
        rename_vars = dict(zip([val + "_next" for val in all_values], all_values))
        reach = self.__bdd.let(rename_vars, post)

        # Visit all neighbours of v.
        for state in self.__bdd.pick_iter(reach, care_vars=all_values):
            state_bdd = self.__bdd_representation_of_boolean_state(state)

            if state_bdd not in self.visited_map:  # state_bdd is not yet visited.
                self.__hybrid_tarjan_recursive(state_bdd, T_loc, index, attractor_node, all_values)

                #  Checking if the DFS found attractor node already.
                if attractor_node:
                    self.visited_map.clear()

                    return

                self.visited_map[v][self._V_LOWLINK_INDEX] = min(self.visited_map[v][self._V_LOWLINK_INDEX],
                                                                 self.visited_map[state_bdd][self._V_LOWLINK_INDEX])
            elif self.visited_map[state_bdd][self._V_ON_STACK_INDEX] is True:
                self.visited_map[v][self._V_LOWLINK_INDEX] = min(self.visited_map[v][self._V_LOWLINK_INDEX],
                                                                 self.visited_map[state_bdd][self._V_INDEX])

        if self.visited_map[v][self._V_LOWLINK_INDEX] == self.visited_map[v][self._V_INDEX]:
            # Append node which lies in bscc.
            attractor_node.append(v)

            return


    """
        For a give boolean state this function will find some attractor state reachable from 
        it. It uses the Hybrid Tarjan algorithm (see __hybrid_tyrjan_recursive method above).

        Args:
            bdd_of_init_state (list[int]): the boolean vector representing the state.
            T_loc (Function): bdd representing the transition matrix of the processing block. 
                                The word "loc" means here the local transition matrix, i.e. we create
                                the local transition matrix for a given block (or some union of blocks). 
            all_values (list[string]): list containing names of the block nodes.

        Returns:
            A bdd representation of the attractor node reachable from the given state.
    """
    def find_some_attractor_state_from_fixed_state(self, bdd_of_init_state, T_loc, all_values):
        index = [0]
        attractor_node = []
        self.__hybrid_tarjan_recursive(bdd_of_init_state, T_loc, index, attractor_node, all_values)

        return attractor_node[0]


    """
        For a given BDD representation of the BN state (or states) this function
        will compute all states reachable from this state (or states) in one step. The output 
        will be also of the BDD representation. 

        Args:
            states (Function): the BDD representation of state or multiple states.  
        Returns:
            Function: the BDD representation of all states reachable from this state in one step 
                      through the BN. 
    """
    def __one_step_image_monolithic(self, states: Function) -> Function:
        #if direction == "forward":
        #    nodes = self.node_names # x
        #    rename_vars = self.x_next_to_x # x_next -> x
        #elif direction == "backward":
        #    nodes = self.next_variables # x_next
        #    rename_vars = self.x_to_x_next # x -> x_next
        #else:
        #    raise ValueError(f"Wrong direction: {direction}")

        #T = self.bdd_transition_matrix
        T_global = self.__create_bdd_from_boolean_network(self.node_names, self.next_variables, self.bdd_expressions)
        transition_image = T_global & states
        image = self.__bdd.exist(self.node_names, transition_image)
        image_renamed = self.__bdd.let(self.x_next_to_x, image)

        return image_renamed


    """
        For a given BDD representation of the BN state (or states) this function
        will compute all states reachable from this state (or states) in one step with respect to the local
        transition matrix, i.e. transition matrix restricted to the block nodes. The output 
        will be also of the BDD representation. We can compute the forward image or backward image (preimage).

        Args:
            state (Function): the BDD representation of state or multiple states.
            T_loc (Function): the BDD representing the transition matrix.
            all_block_variables (list(string)): list of all names in the processing block.
            forward_vars, backward_vars (dict): technical dictionaries for renaming the variables in the result
                                                image bdd.
            direction (str): string representing direction of the image.
        Returns:
            Function: the BDD representation of all states reachable from this state in one step 
                      through the BN. 
        """
    def _one_step_image(self,
                        T_loc: Function,
                        all_block_variables: list,
                        state: Function,
                        forward_vars,
                        backward_vars,
                        direction: str) -> Function:
        if direction == "forward":
            nodes = all_block_variables  # x
            rename_vars = forward_vars  # x_next -> x
        elif direction == "backward":
            nodes = [val + "_next" for val in all_block_variables]  # x_next
            rename_vars = backward_vars  # x -> x_next
        else:
            raise ValueError(f"Wrong direction: {direction}")

        transition_image = state & T_loc
        image = self.__bdd.exist(nodes, transition_image)
        image_renamed = self.__bdd.let(rename_vars, image)

        return image_renamed

    """
        This function will find all attractors of the BN block reachable from the all_space BDD. The algorithm works as follows:
        Each time (until the space of all valid states is not empty), using the hybrid_tarjan DFS
        algorithm, we find the attractor state. After that the situation is straightforward:
        by finding the forward and backward image we find whole attractor and the set of states to delete. 
        The complexity of the problem highly depends on the structure of BDD of the attractor and images.

        Args:
            all_space (Function): the BDD representing the set of all states where attractors have to be found.
                                  all_space is None by the default - it works for elementary blocks. Once one has 
                                  to find attractors of a non-elementary block this parameter will be set to the 
                                  parent attractor.
            T_loc (Function): the BDD representing the transition matrix of block / unit of blocks.
            all_block_variables (list): the list of nodes in the block / unit of blocks.

        Returns: 
            list[Function]: list of BDD's representations of all attractors. 
    """
    def __find_all_attractors_in_scc_block(self, T_loc,
                                           all_block_variables: list,
                                           all_space=None):
        all_block_attractors = []
        if not all_space:
            all_space = self.__bdd.true

        while all_space != self.__bdd.false:
            model = next(self.__bdd.pick_iter(all_space, care_vars=all_block_variables), None)
            bdd_rule_for_model = self.__bdd_representation_of_boolean_state(model)
            attractor_node = self.find_some_attractor_state_from_fixed_state(bdd_rule_for_model, T_loc,
                                                                             all_block_variables)

            whole_attractor = attractor_node
            remember_last_image = attractor_node

            x_next_to_x = dict(zip([val + "_next" for val in all_block_variables], all_block_variables))
            x_to_x_next = dict(zip(all_block_variables, [val + "_next" for val in all_block_variables]))

            while True:
                # Find one-step image.
                image = self._one_step_image(T_loc,
                                             all_block_variables,
                                             remember_last_image,
                                             x_next_to_x,
                                             x_to_x_next,
                                             direction="forward")

                # Update all_space by deleting the new-found image.
                all_space = all_space & (~image)

                # Add image to the temper attractor holder.
                whole_attractor_constructor = whole_attractor | image

                # Check if the image updated. If not then we already
                # have the attractor.
                if whole_attractor_constructor == whole_attractor:
                    # print("Whole attractor was found.", flush=True)

                    break

                # Update whole-attractor container and last found image.
                whole_attractor = whole_attractor_constructor
                remember_last_image = image

            # Add new-found attractor to the list.
            all_block_attractors.append(whole_attractor)

            # Prepare variables for backward image.
            rename_variable = x_to_x_next  # Creating a renaming dict.
            attractor_node_rename = self.__bdd.let(rename_variable, attractor_node)  # Rename vars in bdd.
            remember_last_image = attractor_node_rename  # Create temper container for last found image.
            whole_backward_image = attractor_node_rename  # Create container for the whole backward image.

            # print("Computing backward image")
            while True:
                # Find backward image of the set "remember_last_image"
                back_image = self._one_step_image(T_loc,
                                                  all_block_variables,
                                                  remember_last_image,
                                                  x_next_to_x,
                                                  x_to_x_next,
                                                  direction="backward")

                # The result of above command is again in x_next for the future while-loop
                # calling and so to update all_space we need to rename them again.
                rename_variables_back = x_next_to_x  # Rename back_image back in terms of x.
                back_image_renamed = self.__bdd.let(rename_variables_back, back_image)
                all_space &= (~back_image_renamed)  # Update all_space by removing back_image.

                # Update the temper backward image container.
                whole_backward_image_constructor = whole_backward_image | back_image

                # Check if we have found everything.
                if whole_backward_image_constructor == whole_backward_image:
                    break

                # Update containers.
                whole_backward_image = whole_backward_image_constructor
                remember_last_image = back_image

        return all_block_attractors

    """
        For a given set of nodes it will return the list of bdd rules which prevents this variables to change. 
        This simulates the asynchronous mode. 
    """
    def __create_non_movers_conditions_for_variables_set(self, variables_set: set):
        eq = lambda a, b: ~self.__bdd.apply('xor', a, b)
        R = []

        for var in variables_set:
            var_next = var + "_next"
            R_i = eq(self.__bdd.var(var_next), self.__bdd.var(var))
            R.append(R_i)

        return R

    """
        Technical function which merges two elementary blocks by extending theirs transition matrix and merging 
        their attractors.

        all_attractors (list[Function]): list of parents attractors of lenght n.
        T_merged (Function): bdd representing the transition matrix of the parent block.
        T_block (Function):  bdd representing the transition matrix of the current elementary block.
        block_attractors (list[Function]): list of the attractors of the block of lenght m.
        block_variables (set): set holding names of the block variables.

        Returns:
            (T_new_merged, merged_attractors): (Function, list[Function] of the length n * m). 
    """
    def __merge_elementary_blocks(self, all_attractors: list[Function],
                                  T_merged: Function,
                                  all_variables: set,
                                  T_block: Function,
                                  block_attractors: list[Function],
                                  block_variables: set):
        # Firstly extend T_merged and T_block to whole set of variables block_variables + all_variables.
        R_all_variables = self.__create_non_movers_conditions_for_variables_set(all_variables)
        R_block_variables = self.__create_non_movers_conditions_for_variables_set(block_variables)

        T_block_extend = T_block
        for R_i in R_all_variables:
            T_block_extend &= R_i

        T_all_variables_extend = T_merged
        for R_i in R_block_variables:
            T_all_variables_extend &= R_i

        # Cross all attractors - because they are independent one can join them by using and operator.
        merged_attractors = []
        for at1 in all_attractors:
            for at2 in block_attractors:
                merged_attractors.append(at1 & at2)

        return T_block_extend | T_all_variables_extend, merged_attractors

    """
    Compute all attractors of a given Boolean Network (BN).

    This function implements one of the core and most technically complex features of the library — 
    the attractor detection algorithm based on the **block-split technique**.  
    Below we describe the method in detail.

    ---

    **Algorithm Overview**

    The algorithm identifies attractors by decomposing the Boolean Network into 
    **strongly connected components (SCCs)**.  
    Each SCC is treated as a separate **block**, and attractors are computed progressively, 
    starting from elementary (independent) blocks and extending their dynamics to dependent ones.

    1. **Decomposition into Blocks**
       - The Boolean Network is decomposed into SCCs (blocks).
       - Blocks that have no parent dependencies are called *elementary blocks*.
       - These blocks can be analyzed independently, so attractors are first computed there.

    2. **Extension to Dependent Blocks**
       - Once attractors for an elementary block are found, each dependent (child) block is processed.
       - For every attractor in the parent block, its dynamic behavior is extended to the child block,
         forming combined transition dynamics.

    ---

    **Detailed Example**

    Consider a Boolean Network consisting of two blocks:

         ___________              ____________
        | x1 <-> x2 |   B1       |  x3   x4  |   B2
        |   \  /    |            |    \   /  |
        |    x3 ----|------------|----> x5   |
        |___________|            |___________|

    **Step 1:**  
    Find attractors for block **B1**.  
    Since B1 has no parent blocks, it can be analyzed independently.  
    Let `T_B1` denote the transition matrix for B1, and let it have attractors `A1` and `A2`.

    **Step 2:**  
    Once B1 is processed, take its child block — **B2** (and enqueue any other children if they exist).

    **Step 3:**  
    Fix one attractor, for example `A1`, and derive its dynamics:  
    `dynamic_A1 = T_B1 & A1`  
    That is, restrict transitions in `T_B1` to those states belonging to attractor `A1`.

    **Step 4:**  
    Construct a new transition matrix for the merged block (B1 + B2).  
    Two kinds of node updates can occur:
    - Updates within B1 (`x1`, `x2`, `x3`) — follow the dynamics of `dynamic_A1`.
    - Updates within B2 (`x3`, `x4`, `x5`) — follow their respective Boolean functions.

    Note that node `x5` (from B2) depends on node `x3` (from B1).  
    Thus, when evaluating its Boolean function, `x3` can only take values 
    reachable within the attractor `A1`.  
    For example, if `A1 = {(1,1,1), (1,0,1)}`, then `x3 = 1` always, 
    so the input to `x5` must reflect that.

    Formally, we define:

        T_new = dynamic_A1 | (T_B2 & A1)

    where:
    - `dynamic_A1` corresponds to transitions within B1,
    - `T_B2 & A1` restricts transitions in B2 to valid inputs derived from `A1`.

    **Step 5:**  
    Using the combined transition matrix `T_new`, compute attractors for the merged block (B1 + B2).  
    The exploration space is limited to states consistent with `A1`.

    ---

    **Summary**
    This hierarchical approach — decomposing the BN into SCC blocks, 
    computing local attractors, and extending dynamics step-by-step — 
    significantly reduces computational complexity and enables efficient 
    computation of global attractors for large Boolean Networks. This algorithm 
    uses also BDD (Binary Decision Diagram) speeding up the computational process 
    and enables to work with large BN models.
    
    Args:
        all_space_constraints (Function): the BDD representing additional constraint on the set of states 
                                          where attractors are going to be found. If not None then 
                                          attractors found in BN will be cut to this space. Needed in 
                                          forward_edgetics method.
        path_to_file (string): string representing the path to the file where new found attractors are going 
                               to be saved. If this string is None then results are going to be saved in the 
                               current directory.
        filename (string): string representing the file name where new found attractors are going to be saved.
                           If None then results are going to be saved to two different files:
                           "attractors_dict_representation.txt" and "attractors_list_representation.txt".
        verbose (bool):     set to True to print details on the computation of attractors.
    Returns: 
        all_attractors (list[Function]): list of bdd's representing all attractors of the BN. 
                                         If all_space_constraint is not None then the result will be
                                         additionally cut to that space. 
    """
    def find_all_attractors(self, all_space_constraints: Function = None,
                            path_to_file: str = None,
                            filename: str = None,
                            verbose: bool = False) -> list[Function]:
        
        start_time = time.time()
        sys.setrecursionlimit(self._RECURSION_LIMIT)

        if verbose:
            print("Finding all attractors", flush=True)
            print(f"The network has {self.number_of_blocks}", flush=True)

        # Prepare containers for the further usage.
        q = deque(
            [
                block_nr
                for block_nr, (_, control_nodes) in self.blocks.items()
                if control_nodes == set()
            ]
        )
        processed_blocks_numbers = set()
        all_variables = set()
        all_attractors = []  # Will have the transit matrix T_loc, and attractor A.

        while q:
            block_nr = q.popleft()

            if block_nr in processed_blocks_numbers:
                continue

            # Find all children of the given block.
            all_control_nodes = self.blocks[block_nr][self._GET_CONTROL_NODES]
            all_parents = set([
                self.node_block_number[node_name] for node_name in all_control_nodes
            ])
            block_variables = self.blocks[block_nr][self._GET_SCC]

            # To correctly find update functions one has to
            # sort variables with respect to the general order.
            correct_order_of_block_variables = []
            for gen in self.node_names:
                if gen in block_variables:
                    correct_order_of_block_variables.append(gen)

            all_block_children = self.block_children[block_nr]
            block_nodes_next = [
                var + "_next" for var in correct_order_of_block_variables
            ]
            block_update_functions_bdd = [
                self.node_name_to_bdd_update_function[node_name] for node_name in correct_order_of_block_variables
            ]

            # All parent blocks are already processed.
            if all_parents.issubset(processed_blocks_numbers):

                # If the block is elementary block.
                if self.__is_elementary_block(all_parents):
                    if verbose:
                        print(f"The elementary block nr. {block_nr} is processed")

                    # Find transition matrix of the block.
                    T_loc = self.__create_bdd_from_boolean_network(nodes=correct_order_of_block_variables,
                                                                   nodes_next=block_nodes_next,
                                                                   functions=block_update_functions_bdd)

                    # With this transition matrix find all attractors of the elementary block.
                    block_attractors = self.__find_all_attractors_in_scc_block(T_loc,
                                                                               correct_order_of_block_variables,
                                                                               all_space=None)

                    # Mark the block as already processed.
                    processed_blocks_numbers.add(block_nr)

                    # If that is the first elementary block then just save the result:
                    # i.e. save transition matrix and the attractor states.
                    if not all_attractors:
                        all_attractors = [(T_loc, block_attractors)]
                    else:
                        # If the Boolean Network (BN) contains more than one elementary block,
                        # the results from these blocks must be merged.
                        #
                        # The merging procedure is straightforward:
                        # 1. Apply a logical "OR" operation to the transition matrices of the two elementary blocks.
                        #    Before doing so, extend each matrix to include all variables from both blocks —
                        #    this step preserves the asynchrony of the system.
                        # 2. For each attractor A from the previous elementary block,
                        #    combine it with each attractor B from the current elementary block
                        #    by computing their product (A & B).
                        if verbose:
                            print("Merging elementary blocks")
                        T_elementary_1, attractors_elementary_1 = all_attractors[0]
                        T_merged, attractors_merged = self.__merge_elementary_blocks(attractors_elementary_1,
                                                                                     T_elementary_1,
                                                                                     all_variables,
                                                                                     T_loc,
                                                                                     block_attractors,
                                                                                     block_variables)

                        # Save merged result to the general container.
                        all_attractors = [(T_merged, attractors_merged)]

                    # Mark block variables as already processed.
                    all_variables |= block_variables
                else:  # The block is not elementary.
                    if verbose:
                        print(f"The non-elementary block nr {block_nr} is processed")

                    # Create transition matrix for non-elementary block. At this stage it does not matter
                    # the dynamic of its control nodes.
                    T_child = self.__create_bdd_from_boolean_network(nodes=correct_order_of_block_variables,
                                                                     nodes_next=block_nodes_next,
                                                                     functions=block_update_functions_bdd)

                    # Extend transition matrix of the block to all variables. Similarly extend transition matrix
                    # of parent block to the block variables.
                    extend_conditions_for_new_block = self.__create_non_movers_conditions_for_variables_set(
                        all_variables)
                    extend_conditions_for_parent_block = self.__create_non_movers_conditions_for_variables_set(
                        block_variables)

                    # Each R_i represents the non-mover conditions for current block.
                    # More precisely by adding R_i we say that: variables from the parent block
                    # has to be fixed - it then preserves asynchronuous update mode.
                    for R_i in extend_conditions_for_new_block:
                        T_child &= R_i

                    # Find all new attractors with constructing realisations of the block with respect
                    # to the parent attractor and its dynamic.
                    new_all_attractors = []

                    # Iterate all attractors and theirs dynamics.
                    for (T_parent, attractor_list_parent) in all_attractors:
                        T_extend_parent = T_parent

                        # Similarly to the above code - we extend transition matrix for the parent
                        # on the child's variables.
                        for R_i in extend_conditions_for_parent_block:
                            T_extend_parent &= R_i

                        # For all attractors found under T_parent transition matrix create an realisation.
                        for parent_attractor in attractor_list_parent:
                            # Cut T_child to the previously found attractor.
                            T_child_extend = T_child & parent_attractor

                            # Find transition graph of the attractor.
                            attractors_dynamic = T_extend_parent & parent_attractor

                            # Create new merged transition matrix.
                            T_new = T_child_extend | attractors_dynamic

                            # Extend all_variables and processed_blocks_number containers.
                            all_variables |= block_variables
                            processed_blocks_numbers.add(block_nr)

                            # Find all attractors of the new merged blocks with the new
                            # transition matrix.
                            new_attractors = self.__find_all_attractors_in_scc_block(T_loc=T_new,
                                                                                     all_block_variables=list(
                                                                                         all_variables),
                                                                                     all_space=parent_attractor)
                            # if verbose:
                            #    print(f"Attractors of the block {block_nr} were found")

                            # Save the result.
                            new_all_attractors.append((T_new, new_attractors))

                    # Update all_attractors - this container keeps now all attractors together with theirs
                    # transition matrices of the new merged block.
                    all_attractors = new_all_attractors.copy()

                # Once the block is processed add all its block children to the queue q.
                for child in all_block_children:
                    q.append(child)

            # The block has parent which is not processed yet. Then put it on the end of
            # the queue.
            else:
                q.append(block_nr)

        # PRINTING STAFF.
        end_time = time.time()
        if verbose:
            print(f"All attractors are found in {int((end_time - start_time) / 60)} minutes")
        counter = 1

        if path_to_file or filename:        
            if not path_to_file:
                path_to_file = os.getcwd()
                print("The path to save results is not given. Results "
                    "are going to be saved in the current directory.\n",
                    flush=True)
            if not filename:
                print("The name of the file is not given. Results are going to be saved to the "
                    "attractors_dict_representation.txt for the dict representation and "
                    "to attractors_list_representation.txt "
                    "for a list representation.", flush=True)
                filename_1 = "attractors_dict_representation.txt"
                filename_2 = "attractors_list_representation.txt"
            else:
                filename_1 = filename + "_dict_representation.txt"
                filename_2 = filename + "_list_representation.txt"

            with open(os.path.join(path_to_file, filename_1), "w") as f, open(os.path.join(path_to_file, filename_2), "w") as g:
                g.write(f"The order is {self.node_names}")
                # all_attractors = [(T, A1), (T2, A2), .... , ], Ti - transition matrix of
                # merged blocks realisation, Ai - the final attractor in this realisation.
                for _, attractor_list in all_attractors:
                    for attractor in attractor_list:
                        f.write(f"Attractor {counter}:\n")
                        g.write(f"Attractor {counter}:\n")
                        f.write("=======================\n")
                        g.write("=======================\n")
                        print(f"Attractor nr {counter}", flush=True)
                        counter += 1
                        print(f"=======================")
                        states_counter = 0
                        # models: {name1 : True, name2 : False ,..., name_n : False}
                        for models in self.__bdd.pick_iter(attractor, care_vars=self.node_names):
                            state_string_dict = "{"
                            state_string_list = "("

                            for node_name in self.node_names:
                                name, value = node_name, int(models[node_name])
                                boolean_state = "1, " if models[node_name] else "0, "
                                state_string_dict += name + " : " + boolean_state
                                state_string_list += boolean_state

                            states_counter += 1
                            state_string_dict = state_string_dict[:-2]
                            state_string_list = state_string_list[:-2]
                            state_string_dict += "}"
                            state_string_list += ")"
                            f.write(state_string_dict + "\n")
                            g.write(state_string_list + "\n")
                            print(state_string_list, flush=True)

                        print(f"Number of states: {states_counter}\n"
                            f"-------------------", flush=True)
                        f.write(f"Number of states {states_counter} \n")
                        g.write(f"Number of states {states_counter} \n")

        # Returning attractors: one has to go through list of tuples and append
        # only bdd's representing attractors.
        remember_atractors = []
        for _, attractor_list in all_attractors:
            for attractor in attractor_list:
                if all_space_constraints:
                    intersection = attractor & all_space_constraints

                    if intersection != self.__bdd.false:
                        remember_atractors.append(intersection)
                else:
                    remember_atractors.append(attractor)

        return remember_atractors

    """
      From the source_node removes "target_node". 
      This is the function for modifying the Boolean Network for the forward-edgetics.

      Args: 
          source_node (string): string representing the name of the source node 
                                from the ispl file.
          target_node (string): string representing the name of the removed node  
          from the Boolean update function corresponding to the source_node.
          modify_model (bool): optional flag indicating whether the edge is removed
                               from the Boolean network or only the new function is
                               printed.

      Returns: string representing modified Boolean function.
    """
    def remove_edge(self, source_node: str, target_node: str, modify_model: bool = True) -> str:

        # Function which returns index such that list_of_names[index] = name_to_find.
        def find_index_in_names(list_of_names, name_to_find) -> int:
            index = -1

            for i, node_name in enumerate(list_of_names):
                if node_name == name_to_find:
                    index = i
                    break

            if index == -1:
                raise ValueError(f"The source node {name_to_find} does not exist.")

            return index

        # Find indeces of source and removed nodes in the general node_names list.
        source_node_index_general = find_index_in_names(self.node_names, source_node)
        removed_node_index_general = find_index_in_names(self.node_names, target_node)

        # Find function corresponding to the source_node.
        function_to_update = self.functions[source_node_index_general]

        # For technical future comparison - find symbol representing "removed_node".
        removed_symbol = self.list_of_nodes[removed_node_index_general]

        # Take all symbols from the function. To avoid duplicates use set.
        function_variables = list(set(function_to_update.get_symbols()))

        # Find iremove_edgendex of "removed_node" in functions variables.
        removed_symbol_index_in_fun_var = find_index_in_names(function_variables, removed_symbol)

        # Save old function and its general index to the class variables.
        self.original_function = self.functions_str[source_node_index_general]
        self.old_index = source_node_index_general

        # This variable will keep new function after variable removing.
        updated_function = self.__bool_algebra.FALSE

        # Prepare loop range.
        n = len(function_variables) - 1

        if n == 0:
            value_1 = function_to_update.subs({function_variables[0] : self.__bool_algebra.FALSE}, simplify=True)
            value_2 = function_to_update.subs({function_variables[0] : self.__bool_algebra.TRUE}, simplify=True)

            if value_1 == self.__bool_algebra.FALSE and value_2 == self.__bool_algebra.TRUE:
                value_of_removed_function = self.__bool_algebra.FALSE

            # Removed value is inhibitor.
            elif value_1 == self.__bool_algebra.TRUE and value_2 == self.__bool_algebra.FALSE:
                value_of_removed_function = self.__bool_algebra.TRUE

            # Removed value is not an activator and inhibitor
            else:
                value_of_removed_function = value_1

            updated_function = value_of_removed_function
        else:
            # This loop will generate all possible inputs to the function_to_update.
            for number in range(2 ** n):
                binary_representation = (bin(number))[2:].zfill(n)  # Extend boolean array to the length n.
                bin_list = list(binary_representation)

                # Substitute 1 by algebra.FALSE and 0 by algebra.TRUE for further substitutions.
                prepare_values = [
                    self.__bool_algebra.FALSE if bit == '0' else self.__bool_algebra.TRUE
                    for bit in bin_list
                ]

                # To understand this if, else logic we are going to demonstrate the example:
                # Assume F(x1,x2,x3,x4) is our function to update. This loop will generate
                # only boolean table of length 3. For example prepare_value = [false, true, false].
                # We have two technical considerations x_remove variable index is 4
                # (i.e. Tilde F(x1,x2,x3) = remove(F, x4)) or x_remove variable index is < 4.
                # Assume without loss of generality the second case where x_remove index < 4,
                # let for example x_remove_index = 2. We then want to create two vectors:
                # [false, TRUE, false, true] and [false, FALSE, false ,true]. More precisely
                # we extend array by the value stored in x_remove_index and then we put True and False
                # in the place of x_remove_index. In that way we are always going to generate
                #
                #                     |                     [x_1, x_2, ..., TRUE, ..., x_n, x_remove]
                #                    \ /                  /
                #  [ x_1, x_2, ..., x_remove, ..., x_n] ->
                #                                         \
                #                                           [x_1, x_2, ..., FALSE, ..., x_n, x_remove]
                if removed_symbol_index_in_fun_var < n:
                    prepare_values.append(prepare_values[removed_symbol_index_in_fun_var])
                else:
                    prepare_values.append("TECHNICHAL APPEND")  # To extend the length of prepare_values.

                # Firstly set FALSE in the removed variable and prepare dict to the substitution.
                prepare_values[removed_symbol_index_in_fun_var] = self.__bool_algebra.FALSE
                substitutions = dict(zip(function_variables, prepare_values))
                value_1 = function_to_update.subs(substitutions, simplify=True)

                # Now set TRUE into removed variable.
                substitutions[removed_symbol] = self.__bool_algebra.TRUE
                value_2 = function_to_update.subs(substitutions, simplify=True)

                # Apply forward_edgetics algortithm to find value of function
                # basing on the above value_1, value_2.
                # value_of_removed_function = self.__bool_algebra.FALSE

                # Removed variable is activator.
                if value_1 == self.__bool_algebra.FALSE and value_2 == self.__bool_algebra.TRUE:
                    value_of_removed_function = self.__bool_algebra.FALSE

                # Removed value is inhibitor.
                elif value_1 == self.__bool_algebra.TRUE and value_2 == self.__bool_algebra.FALSE:
                    value_of_removed_function = self.__bool_algebra.TRUE

                # Removed value is not an activator and inhibitor
                else:
                    value_of_removed_function = value_1

                # This fragment will create a boolean expression corresponding
                # to the variable values. For example if for
                # x_1 = true, x_2 = false, x_remove = false, x_4 = true we have
                # Tilde{F}(x_1, x_2, x_3) = True we do the following expression:
                # (x_1 & ~x_2 & x4)
                # which is true on exactly this vector.
                if value_of_removed_function == self.__bool_algebra.TRUE:
                    one_logical_block = self.__bool_algebra.TRUE

                    for i, val in enumerate(prepare_values):
                        if i != removed_symbol_index_in_fun_var:
                            if val == self.__bool_algebra.TRUE:
                                one_logical_block &= function_variables[i]
                            else:
                                one_logical_block &= ~function_variables[i]

                    # This logical blok is combining all expressions to the
                    # one function by putting "or" operator between "one_logical_block", i.e.
                    # the result will be: Tilde{F} = (expr_1) | (expr_2) | (expr_3) ... | (expr_n).
                    updated_function |= one_logical_block

        updated_function = updated_function.simplify()
        self.functions[source_node_index_general] = updated_function
        print("=============================================================================================")
        print(f"After removing the variable \"{target_node}\" from the function: \n"
              f"{function_to_update} \n"
              f"we get \n"
              f"{updated_function}")
        print("=============================================================================================")

        converted_to_string = str(updated_function)

        if modify_model:
            if converted_to_string == self._CONVERTED_FALSE_IN_BOOLEAN_PY_TO_STRING:
                self.functions_str[source_node_index_general] = str(self.node_names[self._FIRST_VARIABLE_NAME_IN_ISPL_FILE]) \
                                                                + "&~" \
                                                                + str(self.node_names[self._FIRST_VARIABLE_NAME_IN_ISPL_FILE])
                self.bdd_expressions[source_node_index_general] = self.__bdd.add_expr(self.functions_str[source_node_index_general])
            elif converted_to_string == self._CONVERTED_TRUE_IN_BOOLEAN_PY_TO_STRING:
                self.functions_str[source_node_index_general] = str(self.node_names[self._FIRST_VARIABLE_NAME_IN_ISPL_FILE]) \
                                                                + "|~" \
                                                                + str(
                    self.node_names[self._FIRST_VARIABLE_NAME_IN_ISPL_FILE])
                self.bdd_expressions[source_node_index_general] = self.__bdd.add_expr(self.functions_str[source_node_index_general])

            else:
                self.functions_str[source_node_index_general] = str(updated_function)
                self.bdd_expressions[source_node_index_general] = self.__bdd.add_expr(str(updated_function))

        return converted_to_string # Be aware of the string convertation of the True and False constants in boolean.py!


    """
        This function modifies a selected update function by removing its dependency on the removed_node.
        It then computes the n-step image of the initial state under the modified Boolean network and searches
        for all attractors reachable from that image.

        Args:
            source_node (string):               string representing the name of node (gene) from which the target_node
                                                dependency will be removed.
            target_node (string):               string representing the name of the removed node from the source node.
            number_of_steps (int):              integer representing the number of push_forward of the init_state via
                                                modified BN.
            init_states (tuple[int, ...]):      the tuple representing the initial state of BN from which the 
                                                analysis begins.
            verbose (bool):                     a flag indicating whether the function should run in verbose mode.

        Returns:
            list[Function]: the list of all attractors reachable from the push_forward(init_state, number_of_steps, modified BN).
    """
    def forward_edgetics(self,
                         source_node: str,
                         target_node: str,
                         number_of_steps: int,
                         init_state: tuple[int, ...],
                         verbose: bool = False) -> list[Function]:
        # Remove the edge between source_node and remove_node.
        self.remove_edge(source_node, target_node)

        # For testing
        # self.draw_state_transition_graph(os.path.join("models", "custom_models", "original_transition_matrix_modyfied.png"),
        #                                  selected_node=init_state,
        #                                  transient_state_color="white")

        # Create the copy of number of steps for further printing.
        n = number_of_steps

        models = dict(zip(self.node_names, init_state))

        # Create a BDD representation of init_state.
        bdds_of_init_states = self.__bdd_representation_of_boolean_state(models)
        #bdd_of_init_states = self.__bdd.false

        # for state_bdd in bdds_of_init_states:
        #     bdd_of_init_states = bdd_of_init_states | state_bdd
        if not CUDD_LOADED:
            self.__bdd.collect_garbage()

        # This BDD will hold whole push_forward image of initial state n-times.
        image = bdds_of_init_states
        
        # Technical container for checking if the image is not changing anymore.
        # Could be used if the BN is relatively small and n is big enough.
        last_image = bdds_of_init_states
        node_next = [name + "_next" for name in self.node_names]
        forward_vars = dict(zip(node_next, self.node_names))
        backward_vars = dict(zip(self.node_names, node_next))

        T_global = self.__create_bdd_from_boolean_network(self.node_names,
                                                          nodes_next=node_next,
                                                          functions=self.bdd_expressions)

        while n > 0:
            image = self._one_step_image(T_global,
                                         self.node_names,
                                         last_image,
                                         forward_vars=forward_vars,
                                         backward_vars=backward_vars,
                                         direction="forward")

            if last_image == image:
                break

            n -= 1
            last_image = image


        # This could be removed in a production version.
        # if verbose:
        #     print(f"After pushing forward for {number_of_steps} steps through the BDD we got: \n", flush=True)
        #     self._print_bdd(image)
        # ----------------

        # Replace a modified update function to the original one after the remove_edge method.
        self.bdd_expressions[self.old_index] = self.__bdd.add_expr(self.original_function)
        self.functions_str[self.old_index] = self.original_function
        self.functions[self.old_index] = self.__bool_algebra.parse(self.original_function, simplify=True)

        T_global = self.__create_bdd_from_boolean_network(self.node_names,
                                                          nodes_next=node_next,
                                                          functions=self.bdd_expressions)

        # Searching for the forward image.
        whole_forward_image = image
        remember_last_image = image

        while True:
            one_step_forward_image = self._one_step_image(T_global,
                                                          self.node_names,
                                                          remember_last_image,
                                                          forward_vars=forward_vars,
                                                          backward_vars=backward_vars,
                                                          direction="forward")
            forward_image_constructor = whole_forward_image | one_step_forward_image

            if whole_forward_image == forward_image_constructor:
                break

            whole_forward_image = forward_image_constructor
            remember_last_image = one_step_forward_image

        # Create all_space.
        all_space = whole_forward_image

        # These lines could be removed in the future.
        # if verbose:
        #     print("All image is then:", flush=True)
        #     # print(f"BEFORE REORDER {len(self.__bdd)}")
        #     self._print_bdd(all_space)
        #     # print(f"AFTER REORDER {len(self.__bdd)}")
        # ------------

        # bdd.reorder(self.__bdd, self.custom_order)
        # Find all atrractors in all_space.

        attractors = self.find_all_attractors(all_space_constraints=all_space)

        return attractors
    

    # For testing only - undocumented: Get the set of attractors reachable from the given set of states.
    def _get_reachable_attractors(self, states: list[tuple[int,...]], withTransientStates : bool = False):
        attractors = self.getAttractors()
        #attractors = self._enumerate_attractors(self.find_all_attractors())

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

        attractor_states = set()
        for attractor_key in reachableAttractors:
            attractor_states = attractor_states.union(attractors[attractor_key])

        if withTransientStates:
            transient_states_bin = set([state2bin(s) for s in processed_states]).difference(attractor_states)
            #return reachableAttractors, transient_states_bin
            return attractor_states, transient_states_bin

        #return reachableAttractors
        return attractor_states


    """

    """
    def get_reachable_attractors_with_bdd(self, states: list[tuple[int,...]]) -> dict[str, set[str]]:

        states_bdd = self.__bdd.false
        for state_bdd in self.__bdd_representation_of_boolean_states([list(s) for s in states]):
            states_bdd = states_bdd | state_bdd
        # print(f"Size of BDD manager before garbage collection: {len(self.__bdd)}")
        if not CUDD_LOADED:
            self.__bdd.collect_garbage()
        # print(f"Size of BDD manager after garbage collection: {len(self.__bdd)}")

        # Searching for the forward image.
        whole_forward_image = states_bdd
        remember_last_image = states_bdd

        while True:
            # ToDo: Consider decomposition-based implementation (?)
            one_step_forward_image = self.__one_step_image_monolithic(remember_last_image)
            forward_image_constructor = whole_forward_image | one_step_forward_image

            if whole_forward_image == forward_image_constructor:
                break

            whole_forward_image = forward_image_constructor
            remember_last_image = one_step_forward_image

        #bdd.reorder(self.__bdd, self.custom_order)
        # Find all attractors in all_space.
        attractors = self.find_all_attractors(all_space_constraints=whole_forward_image)

        return self._enumerate_attractors(attractors)
    


    def get_reachable_states_in_n_steps(self, states: list[tuple[int,...]], num_steps: int = 1) -> dict[str, set[str]]:

        states_bdd = self.__bdd.false
        for state_bdd in self.__bdd_representation_of_boolean_states([list(s) for s in states]):
            states_bdd = states_bdd | state_bdd
        # print(f"Size of BDD manager before garbage collection: {len(self.__bdd)}")
        if not CUDD_LOADED:
            self.__bdd.collect_garbage()
        # print(f"Size of BDD manager after garbage collection: {len(self.__bdd)}")

        # Searching for the forward image.
        one_step_forward_image = states_bdd
        remember_last_image = states_bdd

        for _ in range(num_steps):
            # ToDo: Consider decomposition-based implementation (?)
            one_step_forward_image = self.__one_step_image_monolithic(remember_last_image)
            
            if remember_last_image == one_step_forward_image:
                break

            remember_last_image = one_step_forward_image

        states = set()
        for models in list(self.__bdd.pick_iter(remember_last_image, care_vars=self.node_names)):
            state = []
            for node_name in self.node_names:
                state.append(1 if models[node_name] else 0)
            states.add(tuple(state))

        return states


    def get_reachable_attractors_with_edge_removed(self, source_node, target_node, n_steps, initial_state) -> dict[str, set[str]]:
        #reachable_attractors = []

        attr_from_s = self._enumerate_attractors(self.forward_edgetics(source_node, target_node, n_steps, initial_state))

        #for attr in attr_from_s.values():
        #    reachable_attractors.append(attr)

        return attr_from_s
    

    def simulate(self, nsteps: int, init_state: tuple[int, ...] = None):
        var_indices = {var: i for i, var in enumerate(self.node_names)}
        parent_variables = [sorted([var_indices[var.__str__()] for var in f.symbols]) for f in self.functions]
        truth_tables = [[y for _, y in truth_table(self.functions_str[i], [x.__str__() for x in self.functions[i].symbols])] for
                        i in range(self.num_nodes)]

        pbn = bang.PBN(self.num_nodes,
                       [1 for _ in range(self.num_nodes)],
                       [len(f.symbols) for f in self.functions],
                       truth_tables,
                       parent_variables,
                       [[1.] for _ in range(self.num_nodes)],
                       0.,
                       [],
                       n_parallel=min(max(77, self.num_nodes * 10), 2 ** self.num_nodes - 1))
        pbn._n_parallel = min(max(77, pbn.n_nodes * 10), 2 ** pbn.n_nodes - 1)
        pbn.device = "gpu"

        if init_state is not None:

            pbn.set_states([[True if bit==1 else False for bit in init_state]])

        else:

            init_state = [random.choices([True, False], k=self.num_nodes)]
            pbn.set_states(init_state)

        # Simulate the network for nsteps
        pbn.simple_steps(nsteps)

        # Access the simulation history
        print("Trajectory history:", pbn.history)
        # Access the final state
        print("Last state:", pbn.last_state)

        trajectory = [int2bin(s[0][0], self.num_nodes)[::-1] for s in pbn.history]

        return trajectory

