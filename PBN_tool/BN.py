import networkx as nx
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from pyvis.network import Network
import boolean as bool
from collections import deque

from sympy.codegen.ast import continue_

from utils import *
import copy
import os
import random
import sympy
import numpy as np
from functools import cache
import dd.cudd as bdd
from dd.autoref import Function
from tqdm import tqdm
import time
import sys

class BN():
    __bool_algebra = bool.BooleanAlgebra()
    __bdd = bdd.BDD()
    _V_INDEX = 0
    _V_LOWLINK_INDEX = 1
    _V_ON_STACK_INDEX = 2
    _FIRST_RULE = 0
    _RECURSION_LIMIT = 1000000
    __bdd.configure(reordering=True)
    _SAVING_STEP = 10

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

    # @staticmethod
    # def generateRandomBN(num_nodes, max_parent_nodes, min_parent_nodes=1, allowSelfLoops=True, allow_input_nodes=True, mode='asynchronous'):

    # ToDo: Works, but needs to be corrected - problem with handling the required minimum number of parent nodes,
    #       fixed nodes, and input nodes in a proper, efficient way.
    @classmethod
    def generateRandomBN(cls, num_nodes, max_parent_nodes, min_parent_nodes=1, allowSelfLoops=True,
                         allow_input_nodes=True, mode='asynchronous'):

        print(f"Generating random Boolean network with {num_nodes} nodes ...")

        list_of_nodes = ['x' + str(i) for i in range(num_nodes)]
        input_node_indices = set(range(num_nodes))

        BN_functions = []
        num_fixed = 0
        for i in range(num_nodes):
            num_parents = random.randint(min_parent_nodes, max_parent_nodes)

            if allowSelfLoops:
                nodes = list_of_nodes
            else:
                nodes = copy.copy(list_of_nodes)
                nodes.pop(i)

            parents = sorted(random.sample(nodes, num_parents))

            # If not an input node, remove it from the set of input nodes
            if len(parents) > 1 or list_of_nodes[i] not in parents:
                input_node_indices.remove(i)

            fun, is_fixed, num_ps = cls.__generateRandomBooleanFunction(parents)
            # INEFFICIENT !!!
            # Make sure that the node truly depends on at least the specified number of parent nodes
            while num_ps < min_parent_nodes:
                fun, is_fixed, num_ps = cls.__generateRandomBooleanFunction(parents)

            if is_fixed:
                num_fixed += 1
                input_node_indices.add(i)

            BN_functions.append(fun)

        # If required, make sure that there are no input nodes
        if not allow_input_nodes:
            for i in input_node_indices:
                num_parents = random.randint(1, max_parent_nodes)

                nodes = copy.copy(list_of_nodes)
                nodes.pop(i)

                parents = sorted(random.sample(nodes, num_parents))

                fun, is_fixed, num_ps = cls.__generateRandomBooleanFunction(parents)
                # INEFFICIENT !!!
                while num_ps < min_parent_nodes:
                    fun, is_fixed, num_ps = cls.__generateRandomBooleanFunction(parents)

                BN_functions[i] = fun

        print("=============================================================================================")
        print(f"A Boolean network with {num_nodes} nodes is generated. The Boolean functions are as follows:")
        for i, f in enumerate(BN_functions):
            print(f"{list_of_nodes[i]} = {f}")
        print(f"Number of fixed-value nodes: {num_fixed}")
        print("=============================================================================================")

        bn = BN(list_of_nodes, BN_functions, mode)

        return bn

    def heuristic_order(self):
        print("Finding custom order", flush=True)
        # parsed
        symbols = self.list_of_nodes
        boolean_functions = self.functions

        how_much_children = {node_name: 0 for node_name in symbols}
        parents = {node_name: [] for node_name in symbols}

        for i, fun in enumerate(boolean_functions):
            arguments = fun.symbols # take all symbols representing the function

            for arg in arguments:
                how_much_children[arg] += 1

            parents[symbols[i]] = arguments

        unordered_nodes = set(symbols)
        position = 0
        custom_order = dict()

        while unordered_nodes != set():
            min_child_number = how_much_children[
                min(unordered_nodes, key=how_much_children.get)
            ]

            nodes_with_minumum_children = {
                node_symbol
                for node_symbol in unordered_nodes
                if how_much_children[node_symbol] == min_child_number
            }
            process_nodes = nodes_with_minumum_children

            while process_nodes != set():
                unordered_nodes = unordered_nodes.difference(process_nodes)

                for node in process_nodes:
                    custom_order[node.obj] = position
                    position += 1

                all_parents = set()

                for node in process_nodes:
                    parents_of_the_node = parents[node]

                    for parent in parents_of_the_node:
                        if parent in unordered_nodes:
                            all_parents.add(parent)

                process_nodes = all_parents

        for x_next in self.next_variables:
            custom_order[x_next] = position
            position += 1

        print(custom_order)
        return custom_order

    """
        This function creates the BDD representation of the transition matrix T of BN.

        Description:
        Assume that the BN has n variables (x_1, ..., x_n). Assume moreover two different
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

        Args: None
        Returns: The BDD representation of the transition matrix T. 
        """
    def __create_bdd_from_boolean_network(self):
        # Define the lambda operator which describes the logic in
        # the description above.
        eq = lambda a, b: ~self.__bdd.apply('xor', a, b)

        # T represents the transition matrix of BN.
        T = self.__bdd.false
        saving_counter = 0

        print("Constructing BDD transition system from BN.")
        for i in tqdm(range(self.num_nodes)):
            # Initial conditions for R_i.
            R_i = self.__bdd.true

            for j in range(self.num_nodes):
                if i != j: # The case when i-th node is updated
                    # Here R_i represents the logic: xi_next = f_i.
                    R_i &= eq(self.__bdd.var(self.next_variables[j]), self.__bdd.var(self.node_names[j]))
                else: # The case: xj_next = xj for j != i.
                    R_i &= eq(self.__bdd.var(self.next_variables[j]), self.bdd_expressions[j])

            T |= R_i

        print("=============================================================================================")
        print("BDD transition system is done.", flush=True)
        print("=============================================================================================")
        #print(f"The size of T before reorder is {len(T)}")


        return T

    """
    From the source_node removes "removed_node". 
    This is the function for modifying the Boolean Network for the forward-edgetics.
    
    Args: 
        source_node (string): string representing the name of the source node 
                              from the ispl file.
        removed_node (string): string representing the name of the removed node  
        from the boolean update function corresponding to the source_node.
    
    Returns: string representing modified boolean function.
    """
    def remove_edge(self, source_node : str, removed_node : str) -> str:

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
        removed_node_index_general = find_index_in_names(self.node_names, removed_node)

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
        updated_function = self.__bool_algebra.TRUE

        # This is the flag of first insertion to the updated_function
        # (see algorithm in the remove_edge description below).
        first_true_value = False

        # Prepare loop range.
        n = len(function_variables) - 1

        # This loop will generate all possible inputs to the function_to_update.
        for number in range(2 ** n):
            binary_representation = (bin(number))[2 : ].zfill(n) # Extend boolean array to the length n.
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
                prepare_values.append("TECHNICHAL APPEND") # To extend the length of prepare_values.

            # Firstly set FALSE in the removed variable and prepare dict to the substitution.
            prepare_values[removed_symbol_index_in_fun_var] = self.__bool_algebra.FALSE
            substitutions = dict(zip(function_variables, prepare_values))
            value_1 = function_to_update.subs(substitutions, simplify=True)

            # Now set FALSE into removed variable.
            substitutions[removed_symbol] = self.__bool_algebra.TRUE
            value_2 = function_to_update.subs(substitutions, simplify=True)

            # Apply forward_edgetics algortithm to find value of function
            # basing on the above value_1, value_2.
            value_of_removed_function = self.__bool_algebra.FALSE

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
                if not first_true_value:
                    first_true_value = True
                    updated_function = one_logical_block
                else:
                    updated_function |= one_logical_block

        updated_function = updated_function.simplify()
        self.functions[source_node_index_general] = updated_function
        print("=============================================================================================")
        print(f"After removing the variable \"{removed_node}\" from the function: \n"
              f"{function_to_update} \n"
              f"we get \n"
              f"{updated_function}")
        print("=============================================================================================")

        self.functions_str[source_node_index_general] = str(updated_function)
        self.bdd_expressions[source_node_index_general] = self.__bdd.add_expr(str(updated_function))

        return str(updated_function)

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
        Initial_Set (list[list]): list of binary lists of the same length representing
                                 some boolean states of Boolean Network.
        direction (string): technical string which is "forward" if the output is served for 
                            searching image of some state and "backward" for the pre image of 
                            some boolean state.
    Returns:
        list: list of bdd's expressions corresponding to the states given in 
              Initial_Set.
    """
    def _bdd_representation_of_boolean_state(self,
                                             Initial_Set : list[list[int]],
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

    def _print_bdd(self, bdd: Function) -> None:
        for models in self.__bdd.pick_iter(bdd, care_vars=self.node_names):
            state = [
                int(models[node_name]) for node_name in self.node_names
            ]
            print(f"State: {state}", flush=True)

    def state_to_bin(self, state : list[int]):
        degree = 1
        result = 0

        for bit in reversed(state):
            result = result + bit * degree
            degree *= 2

        return result

    """
    This function modifies a selected update function by removing its dependency on the removed_node.
    It then computes the n-step image of the initial state under the modified Boolean network and searches
    for all attractors reachable from that image.
    
    Args: 
        source_node (string):   string representing the name of node (gene) from which the removed_node dependency 
                                will be removed.
        remove_node (string):   string representing the name of the removed node from the source node.
        number_of_steps (int):  integer representing the number of push_forward of the init_state via modified BN.
        init_state (list[int]): the state of BN from which the analysis begins.
        
    Returns:
        list[Function]: the list of all attractors reachable from the push_forward(init_state, number_of_steps, modified BN).
    """
    def forward_edgetics(self,
                         source_node: str,
                         removed_node: str,
                         number_of_steps: int,
                         init_state: list[int]) -> list[Function]:
        # Remove the edge between source_node and remove_node.
        self.remove_edge(source_node, removed_node)

        # Create the copy of number of steps for further printing.
        n = number_of_steps

        # Create a BDD representation of init_state.
        bdd_of_init_state = self._bdd_representation_of_boolean_state([init_state])[self._FIRST_RULE]

        # This BDD will hold whole push_forward image of initial state n-times.
        image = bdd_of_init_state

        # Technical container for checking if the image is not changing anymore.
        # Could be used if the BN is relatively small and n is big enough.
        last_image = bdd_of_init_state

        while n > 0:
            image = self._one_step_image(last_image, direction="forward")

            if last_image == image:
                break

            n -= 1
            last_image = image

        # This could be removed in a production version.
        print(f"After pushing forward for {number_of_steps} steps through the BDD we got: \n", flush=True)
        self._print_bdd(image)
        # ----------------

        # Replace a modified update function to the original one after the remove_edge method.
        self.bdd_expressions[self.old_index] = self.__bdd.add_expr(self.original_function)
        self.functions_str[self.old_index] = self.original_function
        self.functions[self.old_index] = self.__bool_algebra.parse(self.original_function, simplify=True)

        # Seaching for the forward image.
        whole_forward_image = image
        remember_last_image = image

        while True:
            one_step_forward_image = self._one_step_image(remember_last_image, direction="forward")
            forward_image_constructor = whole_forward_image | one_step_forward_image

            if whole_forward_image == forward_image_constructor:
                break

            whole_forward_image = forward_image_constructor
            remember_last_image = one_step_forward_image

        # Create all_space.
        all_space = whole_forward_image

        # This lines could be removed in the future.
        print("All image is then:", flush=True)
        #print(f"BEFORE REORDER {len(self.__bdd)}")
        self._print_bdd(all_space)
        #print(f"AFTER REORDER {len(self.__bdd)}")
        # ------------

        #bdd.reorder(self.__bdd, self.custom_order)
        # Find all atrractors in all_space.
        attractors = self.find_all_attractors(all_space=all_space)

        return attractors

    """
    This function will find some attractor state reachable from the node v. This is the 
    main method for the "find_attractor_from_state" method below. 
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
        index (list[int]): is the one-element list which holds the index in 
                           standard Tarjan algorithm. The list simulates C pointers.
        attractor_node (list[Function]): Once an attractor state is found this list will keep it.
                                         Until an attractor state is unknown - this list remains
                                         to be empty.
    """
    def __hybrid_tarjan_recursive(self,
                                  v: Function,
                                  index: list,
                                  attractor_node: list):
        if attractor_node:
            return

        # This is standard Tarjan beginning:
        # Take the unvisited node and update its index and lowlinks.
        # Then put it on the stack and process its neighbours (the processing
        # is described below).
        # All nodes information is stored in the self.visited_map
        # container.
        v_index = index[0]
        v_lowlink = index[0]
        v_onStack = True
        v_int = []
        for m in self.__bdd.pick_iter(v, care_vars=self.node_names):
            v_int = [
                int(m[node_name]) for node_name in self.node_names
            ]

        v_int = self.state_to_bin(v_int)
        self.visited_map[v_int] = [v_index, v_lowlink, v_onStack]
        index[0] += 1

        # Find all pairs (x,x_next) satisfying pre logic.
        # It means that x has to be v and x_next has to be
        # connected with x in BN.
        pre = v & self.bdd_transition_matrix

        # Once all pairs are found use the exist command for
        # taking x_next from the pair (x,x_next)
        post = self.__bdd.exist(self.node_names, pre)

        # x_next is the BDD represented in the x_next variables
        # and so one need to rename them for the future usage.
        rename_vars = dict(zip(self.next_variables, self.node_names))
        reach = self.__bdd.let(rename_vars, post)

        # Visit all neighbours of v.
        state_list = []
        for state in self.__bdd.pick_iter(reach, care_vars=self.node_names):
            state_list.append(state)

        for state in state_list:
            # Take a neighbour state and create a BDD representing it.
            bool_state = [
                int(state[node_name]) for node_name in self.node_names
            ]
            state_int = self.state_to_bin(bool_state)

            state_bdd = self._bdd_representation_of_boolean_state([bool_state])[0]

            if state_int not in self.visited_map: # state_bdd is not yet visited.
                self.__hybrid_tarjan_recursive(state_bdd, index, attractor_node)

                #  Checking if the DFS found attractor node already.
                if attractor_node:
                    self.visited_map.clear()

                    return

                self.visited_map[v_int][self._V_LOWLINK_INDEX]= min(self.visited_map[v_int][self._V_LOWLINK_INDEX],
                                                     self.visited_map[state_int][self._V_LOWLINK_INDEX])
            elif self.visited_map[state_int][self._V_ON_STACK_INDEX] is True:
                self.visited_map[v_int][self._V_LOWLINK_INDEX] = min(self.visited_map[v_int][self._V_LOWLINK_INDEX],
                                                      self.visited_map[state_int][self._V_INDEX])

        if self.visited_map[v_int][self._V_LOWLINK_INDEX] == self.visited_map[v_int][self._V_INDEX]:
            # Append node which lies in bscc.
            attractor_node.append(v)

            return

    """
    For a give boolean state this function will find some attractor state reachable from 
    it. It uses the Hybrid Tarjan algorithm (see hybrid_tyrjan method above).
    
    Args:
        state (list[int]): the boolean vector representing the state.
    
    Returns:
        A bdd representation of the attractor node reachable from the given state.
    """
    def find_some_attractor_state_from_fixed_state(self, state: list) -> Function:
        init_state = self._bdd_representation_of_boolean_state([state])[0]
        index = [0] # simulating C pointers.
        attractor_node = []
        self.__hybrid_tarjan_recursive(init_state, index, attractor_node)

        return attractor_node[0]

    """
    For a given BDD representation of the BN state (or states) this function
    will compute all states reachable from this state (or states) in one step. The output 
    will be also of the BDD representation. 

    Args:
        state (Function): the BDD representation of state or multiple states.   
    Returns:
        Function: the BDD representation of all states reachable from this state in one step 
                  through the BN. 
    """
    def _one_step_image(self, state: Function, direction: str) -> Function:
        if direction == "forward":
            nodes = self.node_names # x
            rename_vars = self.x_next_to_x # x_next -> x
        elif direction == "backward":
            nodes = self.next_variables # x_next
            rename_vars = self.x_to_x_next # x -> x_next
        else:
            raise ValueError(f"Wrong direction: {direction}")

        T = self.bdd_transition_matrix
        transition_image = state & T
        image = self.__bdd.exist(nodes, transition_image)
        image_renamed = self.__bdd.let(rename_vars, image)

        return image_renamed

    """
    This function will find all attractors of the BN reachable from the all_space BDD. The algorithm works as follows:
    Each time (until the space of all valid states is not empty), using the hybrid_tarjan DFS
    algorithm, we find the attractor state. After that the situation is straightforward:
    by finding the forward and backward image we find whole attractor and the set of states to delete. 
    The complexity of the problem highly depends on the structure of BDD of the attractor and images.
    
    Args:
        all_space (Function): the BDD representing the set of all states where attractors have to be found.
                              all_space is None by the default in the case when one is interested in finding
                              all attractors of BN without any additional constraints.
        is_whole_space (Bool): the flag representing if additional constraints are applied. For example
                               if one wants to find all attractors reachable from some state / states then
                               this flag has to be False and the all_space argument has to be set on the BDD
                               representing the whole forward and backward image of that state 
                               (see forward_edgetics method).     
    Returns: 
        list[Function]: list of BDD's representations of all attractors. 
    """
    def find_all_attractors(self, all_space=None) -> list[Function]:
        print("=============================================================================================")
        print("Finding all attractors", flush=True)

        is_whole_space = True if all_space is None else False
        # Set the recursion limit to alow deep DFS in hybrid_tarjan algorithm.
        sys.setrecursionlimit(self._RECURSION_LIMIT)

        if is_whole_space:
            all_space = self.__bdd.true

        # This variable will represent BDD of available states.
        all_attractors = [] # This list will contain BDD of all attractors.
        number_of_attractor = 1 # Attractor counter.

        # While all_space is not empty (this is equivalent to the
        # bdd.false):
        # (1) Find the attractor state using hybrid_tarjan algorithm.
        # (2) By finding the forward image of the attractor state find whole attractor.
        # (3) By backward image of the attractor state find all states connected to the attractor
        # state.
        # (4) Delete from all_space the forward_image and backward_image.
        while all_space != self.__bdd.false:
            start_time = time.time()

            # This list will hold some state from which we
            # are going to search the attractor node.
            take_init_state = []

            # This is the dict representing some valid state.
            model = next(self.__bdd.pick_iter(all_space, care_vars=self.node_names), None)

            # Read the state from the dict.
            if model is not None:
                take_init_state = [
                    int(model[node]) for node in self.node_names
                ]

            print("=============================================================================================")
            print(f"We are searching for an attractor number {number_of_attractor}.", flush=True)
            attractor_node = self.find_some_attractor_state_from_fixed_state(take_init_state)
            print("Attractor node is found.",flush=True)

            # This container will keep the BDD representation of the whole
            # attractor conaining the attractor_node.
            whole_attractor = attractor_node

            # This is the optimization we did: we update the general attractor
            # container by remembering only the last found one-step image.
            remember_last_image = attractor_node
            print("Computing whole attractor", flush=True)

            while True:
                # Find one-step image.
                image = self._one_step_image(remember_last_image, direction="forward")

                # Update all_space by deleting the new-found image.
                all_space = all_space & (~image)

                # Add image to the temper attractor holder.
                whole_attractor_constructor = whole_attractor | image

                # Check if the image updated. If not then we already
                # have the attractor.
                if whole_attractor_constructor == whole_attractor:
                    print("Whole attractor was found.", flush=True)

                    break

                # Update whole-attractor container and last found image.
                whole_attractor = whole_attractor_constructor
                remember_last_image = image
                #bdd.reorder(self.__bdd, self.custom_order)

            # Add new-found attractor to the list.
            all_attractors.append(whole_attractor)

            # Prepare variables for backward image. We need to rename variables
            # for the following logic: in backward image we actually have
            # x_next instead of x, i.e.
            # backwarad(x_next) = exist(x_next, T(x,x_next)) = the set of such x that T(x,x_next) = True.
            # And so to find backward of some set we rename it to set_next and do the same logic as
            # in push-forward image.
            rename_variable = self.x_to_x_next # Creating a renaming dict.
            attractor_node_rename = self.__bdd.let(rename_variable, attractor_node) # Rename vars in bdd.
            remember_last_image = attractor_node_rename # Create temper container for last found image.
            whole_backward_image = attractor_node_rename # Create container for the whole backward image.

            print("Computing backward image")
            while True:
                # Find backward image of the set "remember_last_image"
                back_image = self._one_step_image(remember_last_image, direction="backward")

                # The result of above command is again in x_next for the future while-loop
                # calling and so to update all_space we need to rename them again.
                rename_variables_back = self.x_next_to_x # Rename back_image back in terms of x.
                back_image_renamed = self.__bdd.let(rename_variables_back, back_image)
                all_space = all_space & (~back_image_renamed) # Update all_space by removing back_image.

                # Update the temper backward image container.
                whole_backward_image_constructor = whole_backward_image | back_image

                # Check if we have found everything.
                if whole_backward_image_constructor == whole_backward_image:
                    break

                # Update containers.
                whole_backward_image = whole_backward_image_constructor
                remember_last_image = back_image
                #bdd.reorder(self.__bdd, self.custom_order)

            # Printing stuff.
            end_time = time.time()
            print(f"The attractor {number_of_attractor} is done"
                  f" in {int((end_time - start_time) / 60)} minutes ", flush=True)
            number_of_attractor += 1

        # After the while loop write all attractors on the screen. This part is only for checking
        # and could be changed in arbitrary ways.
        print(f"The number of attractors is: {len(all_attractors)}")

        print("=============================================================================================")
        for i, at in enumerate(all_attractors):
            with open("file.txt", "a") as f:
                f.write(f"Attractor {i + 1}:\n")
                print(f"Attractor  {i + 1}: " , flush=True)
                states_counter = 0

                for models in self.__bdd.pick_iter(at, care_vars=self.node_names):
                    state_string = "("

                    for node_name in self.node_names:
                        state_string += "1, " if models[node_name] else "0, "

                    state_string = state_string[:-2]
                    state_string += ")"
                    states_counter += 1
                    print(state_string, flush=True)
                    f.write(state_string + "\n")
                print(f"Number of states: {states_counter}\n"
                     f"-------------------", flush=True)
        print("=============================================================================================")
        return all_attractors

    def create_blocks(self):
        G = nx.DiGraph()
        G.add_nodes_from(self.node_names)

        # Iterate through functions and create edges for networkx graph.
        for node, function in zip(self.node_names, self.functions_str):
            list_of_parents = []

            for potential_parent in self.node_names:
                if function.find(potential_parent) != -1:
                    list_of_parents.append(potential_parent)

            edges = [
                (parent, node) for parent in list_of_parents
            ]
            G.add_edges_from(edges)

        SCC_generator = nx.strongly_connected_components(G)
        SCC = list(SCC_generator)
        self.number_of_blocks = len(SCC)

        # For tests.
        for scc in SCC:
            print(scc)

        blocks  = dict()

        # Dict holding "node" -> "scc_numb" where it belongs.
        self.node_block_number = dict()
        for block_nr, scc in enumerate(SCC):
            # Technical.
            block_dict = dict([
                (node, str(block_nr)) for node in scc
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

        print(f"After SCC we have the following blocks of the structure graph: {blocks}")

        return blocks

    def _find_attractor_of_block(self,
                                 block_number : int,
                                 block_status : list[bool],
                                 computed_attractors : dict,
                                 blocks : dict):
        if block_status[block_number]:
            return

        # Take all numbers of parent blocks
        parent_blocks = set()
        scc, control_nodes = blocks[block_number]

        for control_node in control_nodes:
            parent_blocks.add(self.node_block_number[control_node])

        # If the block is elementary.


    def find_all_attractors_with_block_splitting(self):
        blocks = self.create_blocks()
        elementary_blocks = deque([
            block_number
            for block_number, (_, control_nodes) in blocks.items()
            if control_nodes == set()
        ])

        print(f"The network has {len(elementary_blocks)} elementary blocks.")
        print("================================\n"
              "Computing attractors for blocks:")

        blocks_status = [False] * self.number_of_blocks

        computed_attractors = dict()
        for block_number in range(self.number_of_blocks):
            self._find_attractor_of_block(block_number, blocks_status, computed_attractors, blocks)


    def __init__(self, list_of_nodes : list[str], list_of_functions : list[str], mode='asynchronous'):
        """
        Boolean Network (BN) constructor

        :param list_of_nodes: describe about parameter p1
        :param list_of_functions: describe about parameter p2
        :param mode: update scheme for the BN; one of the two values: 'asynchronous' or 'synchronous'.
        """

        if mode not in ['asynchronous', 'synchronous']:
            raise ValueError(f"Wrong update scheme: {self.mode}")

        self.node_block_number = None
        self.number_of_blocks = None

        # Mode of the model.
        self.mode = mode

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
        for fun in list_of_functions:
            self.bdd_expressions.append(self.__bdd.add_expr(fun))

        self.next_variables = [
            var + "_next" for var in self.node_names
        ]
        #self.custom_order = self.heuristic_order()
        self.bdd_transition_matrix = self.__create_bdd_from_boolean_network()

        # For the hybrid Tarjan algorithm.
        self.visited_map = dict()

        # Dicts for renaming in searching for the images.
        self.x_to_x_next = dict(zip(self.node_names, self.next_variables))
        self.x_next_to_x = dict(zip(self.next_variables, self.node_names))

    def export2Ispl(self, filename: str, initial_condition: str = None, verbose: bool = True):

        if verbose:
            print(f"Exporting the Boolean network to the {filename} file in the .ispl format ... ", end='')

        with open(filename, 'w') as file:

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
                file.write('\t\t' + node_name + '=true if ' + str(fun) + '=true;\n')
                file.write('\t\t' + node_name + '=false if ' + str(fun) + '=false;\n')
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

    def isFixPointAttractor(self, state):
        substitutions = dict()
        for i, node_value in enumerate(state):
            substitutions[
                self.list_of_nodes[i]] = self.__bool_algebra.TRUE if node_value == 1 else self.__bool_algebra.FALSE

        # Applying all functions one-by-one.
        for node_index, fun in enumerate(self.functions):
            new_node_value = 1 if fun.subs(substitutions, simplify=True) == self.__bool_algebra.TRUE else 0
            if new_node_value != state[node_index]:
                return False

        return True

    def getNeighborStates(self, state, excludedNodes=[]):
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

    def trajectory_statistics(self, init_state, num_steps):
        if self.mode == 'asynchronous':
            # Asynchronous update scheme

            def _generator(state):
                statistics = dict()
                statistics[state] = 1

                while True:
                    for _ in range(num_steps):
                        substitutions = dict()

                        for i, node_value in enumerate(state):
                            substitutions[self.list_of_nodes[
                                i]] = self.__bool_algebra.TRUE if node_value == 1 else self.__bool_algebra.FALSE

                        node_index = random.randrange(self.num_nodes)
                        fun = self.functions[node_index]

                        new_node_value = 1 if fun.subs(substitutions, simplify=True) == self.__bool_algebra.TRUE else 0

                        new_state = list(state)
                        new_state[node_index] = new_node_value

                        state = tuple(new_state)

                        if state in statistics:
                            statistics[state] += 1
                        else:
                            statistics[state] = 1

                    yield statistics

            return _generator(init_state)

        elif self.mode == 'synchronous':
            print("Not implemented yet!")
            return None

    def create_path_and_file(self, num_steps, file_path):
        if os.path.exists(file_path) == False:
            os.makedirs(file_path)

        file_name = str(self.mode) + "_" + str(num_steps) + "_trajectory_simulation.txt"
        whole_path = os.path.join(file_path, file_name)

        return whole_path

    def simulate(self, init_state, num_steps, pert_prob, file_path):
        file_name = self.create_path_and_file(num_steps, file_path)
        state = init_state

        with open(file_name, "w") as f:
            if self.mode == 'asynchronous':
                for j in range(num_steps):
                    substitutions = dict()

                    for i, node_value in enumerate(state):
                        substitutions[self.list_of_nodes[
                            i]] = self.__bool_algebra.TRUE if node_value == 1 else self.__bool_algebra.FALSE

                    # Losujemy node do zupdatowania.
                    node_index = random.randrange(self.num_nodes)
                    fun = self.functions[node_index]

                    new_node_value = 1 if fun.subs(substitutions, simplify=True) == self.__bool_algebra.TRUE else 0
                    new_state = list(state)
                    new_state[node_index] = new_node_value
                    state = tuple(new_state)
                    f.write(str(j) + "-> " + str(state) + "\n")
            else:  # Synchronous mode.
                for j in tqdm(range(num_steps)):
                    perturbation_occurred = False
                    substitutions = dict()

                    for i, node_value in enumerate(state):
                        if node_value:
                            substitutions[self.list_of_nodes[i]] = self.__bool_algebra.TRUE
                        else:
                            substitutions[self.list_of_nodes[i]] = self.__bool_algebra.FALSE

                    # Sprawdzamy czy było flipnięcie na jakimś nodzie i jeśli było to zapisujemy stan
                    # i kontynuujemy. Jeśli flipnięcia nie było to updatujemy cały układ.
                    for node_number in range(len(self.list_of_nodes)):
                        random_value = np.random.uniform(0, 1, 1)

                        # The case when the node is perturbated.
                        if random_value < pert_prob:
                            perturbation_occurred = True
                            fun = self.functions[node_number]
                            new_node_value = (1 if state[node_number] == False else 0)
                            state[node_number] = new_node_value

                    # Update all system.
                    if perturbation_occurred == False:
                        new_state = []

                        for node_number in range(len(self.list_of_nodes)):
                            fun = self.functions[node_number]
                            new_state.append(
                                1 if fun.subs(substitutions, simplify=True) == self.__bool_algebra.TRUE else 0)

                        state = new_state

                    f.write(str(j) + "-> " + str(state) + "\n")

    def generateStateTransitionGraph(self, excludedNodes=[]):
        G = nx.DiGraph()

        for initial_state_int in range(2 ** self.num_nodes):
            initial_state = bin2state(int2bin(initial_state_int, self.num_nodes))

            neighbor_states = self.getNeighborStates(initial_state, excludedNodes)

            edges_aux = []
            for ns in neighbor_states:
                edges_aux.append((state2bin(initial_state), state2bin(ns)))
            G.add_edges_from(edges_aux)

        return G

    def _generatePartialStateTransitionGraph(self, initial_states_bin, excludedNodes=[]):
        G = nx.DiGraph()

        states_to_process = set()
        processed_states = set()

        for initial_state_bin in initial_states_bin:
            states_to_process.add(bin2state(initial_state_bin))

        while len(states_to_process) > 0:
            source_state = states_to_process.pop()

            processed_states.add(source_state)

            neighbor_states = self.getNeighborStates(source_state, excludedNodes)

            # for s in neighbor_states:
            #     print(s)

            edges_aux = []
            for ns in neighbor_states:
                edges_aux.append((state2bin(source_state), state2bin(ns)))
            G.add_edges_from(edges_aux)

            states_to_process = states_to_process.union(neighbor_states.difference(processed_states))

        return G

    def getAttractors(self, excludedNodes=[]):
        GBn = self.generateStateTransitionGraph(excludedNodes)

        attractors = dict()
        for attractor_index, attractor in enumerate(nx.attracting_components(GBn)):
            attractors['A' + str(attractor_index)] = attractor

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

    @cache
    def getStructureGraph(self):

        G = nx.DiGraph()

        edges_aux = []
        for i, f in enumerate(self.functions):
            for s in f.symbols:
                # edges_aux.append((s, self.list_of_nodes[i]))
                edges_aux.append((str(s), self.node_names[i]))

        G.add_edges_from(edges_aux)

        return G

    def plotStructureMPL(self):

        Gbn = self.getStructureGraph()

        # nx.draw_networkx(Gbn)
        nx.draw_networkx(Gbn, pos=nx.spring_layout(Gbn))

        # ax = plt.axes()
        # nx.drawing.nx_pylab.draw(Gbn,ax=ax)

        plt.show()

    def plotStructure(self, filename):

        Gbn = self.getStructureGraph()

        struct_graph = nx.drawing.nx_agraph.to_agraph(Gbn)
        pos = struct_graph.layout('dot')
        # if not os.path.exists(folder):
        #     os.makedirs(folder)
        # struct_graph is an instant of the PyGraphviz.AGraph class
        struct_graph.draw(filename)

    def plotStructureInteractive(self, filename):

        bn_graph = self.getStructureGraph()

        g = Network(height=800, width=800, directed=True, notebook=True, cdn_resources='in_line')
        g.toggle_hide_edges_on_drag(False)
        g.toggle_physics(True)  # Triggering the toggle_physics() method allows for more fluid graph interactions
        g.barnes_hut()
        # g.show_buttons(filter_=["physics"])
        g.from_nx(bn_graph, default_node_size=20, )
        g.show(filename)

    def drawStateTransitionGraph(self, folder, filename, highlightAttractors=True):

        print("Drawing the state transition graph ...")

        Gbn = self.generateStateTransitionGraph()
        # Dpbn is a instant of the PyGraphviz.AGraph class
        Dbn = nx.drawing.nx_agraph.to_agraph(Gbn)

        if highlightAttractors:
            attractors = self.getAttractors()

            colors = mcolors.CSS4_COLORS
            by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name) for
                            name, color in colors.items())
            color_names = [name for hsv, name in by_hsv]
            color_start_ind = color_names.index('palegreen')

            # Modify node fillcolor and edge color.
            Dbn.node_attr.update(color='darkblue', style='filled', fillcolor='chartreuse')
            Dbn.edge_attr.update(arrowsize=1)
            color_ind = color_start_ind

            for BN_key in attractors.keys():
                color_ind = random.choice(range(color_start_ind, len(color_names)))
                for state in attractors[BN_key]:
                    n = Dbn.get_node(state)
                    n.attr['fillcolor'] = colors[color_names[color_ind]]
                # color_ind = (color_ind + 1) % len(color_names)

        pos = Dbn.layout('dot')
        if not os.path.exists(folder):
            os.makedirs(folder)
        # Dpbn is a instant of the PyGraphviz.AGraph class
        Dbn.draw(os.path.join(folder, filename))

        print(f"Done. The state transition graph is saved to {os.path.join(folder, filename)}")

    def getStrongArticulationPoints(self):
        # Inefficient implementation

        G = self.getStructureGraph()

        num_sccs = nx.number_strongly_connected_components(G)

        strong_articulation_points = []
        for i, node in enumerate(self.list_of_nodes):
            G = self.getStructureGraph()
            G.remove_node(node)
            if nx.number_strongly_connected_components(G) > num_sccs:
                strong_articulation_points.append(self.node_names[i])

        return strong_articulation_points

    def getComplementaryStates(self, states_bin: set, node: str) -> set:
        complementary_states = set()

        # Get the index of the node
        node_index = self.node_names.index(node)

        # Get the Boolean function for the node
        fun = self.functions[node_index]

        for state_bin in states_bin:

            state = bin2state(state_bin)

            substitutions = dict()
            for i, node_value in enumerate(state):
                substitutions[
                    self.list_of_nodes[i]] = self.__bool_algebra.TRUE if node_value == 1 else self.__bool_algebra.FALSE

            new_node_value = 1 if fun.subs(substitutions, simplify=True) == self.__bool_algebra.TRUE else 0

            new_state = list(state)
            new_state[node_index] = new_node_value

            complementary_states.add(state2bin(tuple(new_state)))

        return complementary_states


if __name__ == '__main__':
    # bn = BN.generateRandomBN(20,4,min_parent_nodes=2,allowSelfLoops=True,allow_input_nodes=True)
    # #print(bn.getAttractors()) # May take long ...
    # bn.plotStructureMPL()
    #   bn = BN.generateRandomBN(200,5,min_parent_nodes=3,allowSelfLoops=True,allow_input_nodes=False)
    # print(bn.getAttractors())
    #   bn.export2Ispl('C:/Users/AndrzejMizera/Downloads/bn.ispl')
    # bn.plotStructureInteractive('C:/Users/AndrzejMizera/Downloads/bn_structure.html')
    # bn.plotStructure('C:/Users/AndrzejMizera/Downloads/bn_structure.png')

    # UniStemDay 2025 presentation
    bn = BN.generateRandomBN(4, 3, allowSelfLoops=True, allow_input_nodes=True)
    bn.plotStructure('C:/Users/AndrzejMizera/Downloads/bn_structure.png')
    bn.drawStateTransitionGraph('C:/Users/AndrzejMizera/Downloads/', 'bn_dynamics.png')
