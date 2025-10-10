import numpy as np
import os
import boolean.boolean as bool
import sys
import subprocess
from typing import List, Tuple, Union

home_path = os.environ.get("HOME")
cabean = os.path.join("CABEAN-release-2.0.0",
                      "BinaryFile",
                      "Windows&Linux",
                      "cabean")
PATH_TO_CABEAN = os.path.join(home_path,
                              "Downloads",
                              cabean)
PBN_path = os.path.join(home_path, "Boolean-networks", "PBN_tool")
working_models = os.path.join(home_path, "Downloads", "tlgl_models_aux")
MODELS_DIR = os.path.join(PBN_path, "models")

#sys.path.append(PBN_path)
from BN import BN

SUCCESS = 0
FAILURE = 1

def run_CABEAN(model_name):
    model_path = os.path.join(MODELS_DIR, model_name)
    args = [PATH_TO_CABEAN, "-compositional", "2", model_path]
    save_file = MODELS_DIR + "/" + model_name.replace(".ispl", "") + ".txt"

    with open(save_file, "w") as f:
        proc = subprocess.Popen(args=args, stdout=f, stderr=subprocess.PIPE, text=True)

        if proc.returncode != SUCCESS:
            _, stderr = proc.communicate()
            error_file_name = save_file.split(".")[0] + "_error.txt"

            with open(error_file_name, "w") as g:
                g.write(stderr)

"""
 This function reads variables, update functions and initial data
 from the ispl file.
 
 Args:
    path_to_ispl_file (str): string representing path to the ispl BN model.
 Returns:
    Tuple: BN variables, BN update functions, initial data.
"""
def take_variables_and_functions_from_ispl(path_to_ispl_file : str) -> Tuple[List[str], List[str], List[str]]:
    if not os.path.isfile(path_to_ispl_file):
        raise FileNotFoundError(path_to_ispl_file)

    BN_variables = []
    BN_functions = []
    initial_state = []

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
                line = line.split("if")[1]
                line = line.split("=")[0]
                BN_functions.append(line.strip())
                line = ispl_file.readline()

            while line.strip() != "InitStates":
                line = ispl_file.readline()

            line = ispl_file.readline()
            initial_state = (line.strip()).split("and")

            break
    return BN_variables, BN_functions, initial_state

INDEX_OF_LEAVES_IN_ATTRACTOR = 4
EOF =""

""""
This function reads CABEAN output of some file and saves all of the attractors.
 
 Args:
    path_to_ispl_file (str): string representing path to the ispl BN model.
    path_to_CABEAN_output (str): string representing path to the CABEAN output file.
 Returns:
    List[List[List[str]]] representing all of the attractors of the model.
    
 Example: 
    Assume that model has 3 attractors:
    Attractor I:
    1, -, 0, 1 
    -, 0, -, 1
    Attractor II: 
    1, 1, 1, 1
    Attractor III:
    0, 0, 0, 0
    
    Then the output will be:
    [ [[1,-,0,1],[-,0,-,1]], [[1,1,1,1]], [[0,0,0,0]] ]
"""
def read_CABEAN_output(path_to_CABEAN_output : str, path_to_ispl_file : str) -> List[List[List[str]]]:
    if not os.path.isfile(path_to_CABEAN_output) or not os.path.isfile(path_to_ispl_file):
        raise FileNotFoundError(path_to_CABEAN_output, path_to_CABEAN_output)

    all_attractors = []
    number_of_variables = len((take_variables_and_functions_from_ispl(path_to_ispl_file))[0])

    # Here we are going to save text file to avoid working with the file
    # iterators.
    file = []

    with open(path_to_CABEAN_output, "r") as ispl_file:
        for line in ispl_file:
            file.append(line)

    # Add guardian for the further usage.
    file.append(EOF)

    # This function will find the line where the description of attractor begins.
    # It will return false if EOF and true, number_of_attractor_states, updated_line_number otherwise.
    def find_attractor_description(line_number : int) -> Union[bool, Tuple[bool, int, int]]:
        l = file[line_number]
        updated_line_number = line_number

        while l != EOF and not l.startswith(":"):
            updated_line_number += 1
            l = file[updated_line_number]

        if l == EOF:
            return False

        num_states = description_of_attractor(l)
        return True, (num_states, updated_line_number)

    # Helper function to read the attractor information line from CABEAN.
    def description_of_attractor(line : str) -> int:
        l = line.split()

        # Those are helper constants which are describing
        # the position of nodes and states information
        # in processing line of CABEAN.
        index_of_states = 5
        number_of_attractor_states = int(l[index_of_states])
        return number_of_attractor_states

    line_number = 0
    CORRECT_OUTPUT_LENGTH = 2

    while line_number < len(file):
        answer = find_attractor_description(line_number)

        # EOF case.
        if not isinstance(answer, tuple) or len(answer) != CORRECT_OUTPUT_LENGTH:
            break

        _, (number_of_attractor_states, line_number) = answer
        line_number += 1 # Go to the next line to receive attractor states.
        attractor = []

        while number_of_attractor_states > 0:
            line = file[line_number] # Read the attractor state line.
            partial_states = [] # Container for holding CABEAN output line of particular attractor.
            number_of_processed_variables = 0

            # This variable will be even always. It will be walking
            # through the CABEAN line to save the result.
            index = 0

            # Describes the number of variables where the value is arbitrary, i.e. 0 or 1.
            how_many_undefined = 0

            while number_of_processed_variables < number_of_variables:
                node_value = line[index]

                if node_value == "-":
                    how_many_undefined += 1

                partial_states.append(line[index])
                index += 2
                number_of_processed_variables += 1

            # Update while counters and containers.
            attractor.append(partial_states)
            line_number += 1
            number_of_attractor_states -= 2 ** how_many_undefined

        all_attractors.append(attractor)

    return all_attractors

class BN_extension(BN):
    def find_index_of_node(self, node_name : str):
        i = 0

        while i < len(self.node_names):
            if self.nodes_str[i] == node_name:
                return i
            i += 1

        if i == 0:
            raise ValueError(f"Invalid input: {node_name}")

    # From the node it will remove removed_node,
    # i.e. from the expression describing update of "node"
    # removes "removed_node".
    # I am not sure if the algorithm is correct because of the data.
    def remove_edge(self, node : str, removed_node : str) -> None:
        node_index = self.find_index_of_node(node)
        expression = self.functions_str[node_index]
        FIRST_INDEX = 0
        LAST_INDEX = len(expression) - 1
        index = expression.find(removed_node)

        if index == -1:
            raise ValueError(f"The removed node: {removed_node} does"
                             f" not exist in expression: {expression}")

        is_negated = True if index != FIRST_INDEX and expression[index - 1] == "~" else False
        and_appears = False

        # Check on the left.
        if index != FIRST_INDEX:
            if is_negated:
                if index - 2 >= FIRST_INDEX and expression[index - 2] == "&":
                    and_appears = True
            else:
                if index - 1 >= FIRST_INDEX and expression[index - 1] == "&":
                    and_appears = True

        # Check on the right.
        if index != LAST_INDEX and expression[index + 1] == "&":
            and_appears = True

        def decide_value_of_node(is_negated_, and_appears_) -> Tuple[bool, bool]:
            put_true = False
            put_false = False

            if is_negated_:

                # Case: ... ~node and ... , ... and ~node ... -> node = false
                if and_appears_:
                    put_false = True
                    return put_false, put_true

                # Case: ... ~node or ... , ... or ~node ... -> node = true
                else:
                    put_true = True
                    return put_false, put_true
            else:

                # Case: ... and node ... , ... node and ... -> node = true
                if and_appears_:
                    put_true = True
                    return put_false, put_true

                # Case: ... | node ... , ... node | ... -> node = false
                else:
                    put_false = True
                    return put_false, put_true

        put_false, put_true = decide_value_of_node(is_negated, and_appears)
        self.original_function = self.functions[node_index]

        if put_false:
            expression = expression.replace(removed_node, "FALSE")
        else:
            expression = expression.replace(removed_node, "TRUE")

        self.functions[node_index] = self._BN__bool_algebra.parse(expression, simplify=True)
        self.old_index = index
        print(expression)
        print(self.functions[node_index])


if __name__ == "__main__":
    var, fun, z = take_variables_and_functions_from_ispl(MODELS_DIR + "/tlgl_5.ispl")
    print(var)
    print(fun)
    bn = BN_extension(var, fun)
    #print("NOW")
    bn.remove_edge("v_BID", "v_BclxL")