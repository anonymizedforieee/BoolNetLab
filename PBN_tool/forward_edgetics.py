import os
import boolean.boolean as bool
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
from BN_SCC import BN_Realisation
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
                line = line.split(" if ")[1]
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
EOF = ""

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

def if_equal(l1 : list, l2 : list):
    if len(l1) != len(l2):
        #print("NIE SA ROWNE DLUGOSCI")
        return False

    for at_1 in l1:
        wystepuje = False

        for at_2 in l2:
            if at_1 == at_2:
                wystepuje = True
                break
        if not wystepuje:
            print("NIE ROWNE")
            return False

    print("JEST GIT")
    return True

if __name__ == "__main__":
    # var = ["x1", "x2", "x3"]
    # fun = ["!(x1 & x2)", "x1 & !x2", "!x2"]
    #
    # print("With SCC", flush=True)
    # bn = BN_Realisation(var, fun)
    # bn.find_all_attractors()
    # print("Without SCC", flush=True)
    # bn = BN(var, fun)
    # bn.find_all_attractors()
    #
    # # var_manually = ["x0", "x1", "x2", "x3", "x4"]
    # var_auto, fun_auto, _ = take_variables_and_functions_from_ispl("/home/bogdan/Boolean-networks/PBN_tool/models/tlgl_3.ispl")
    #
    # #     print(var_auto, fun_auto)
    # bn = BN_Realisation(var_auto, fun_auto)
    # bn.find_all_attractors()
    #bn = BN.generateRandomBN(5, 4, 2)

    # bn = BN_Realisation.generateRandomBN(6,2,0)
    # bn.find_all_attractors()
    # filename = os.path.join(MODELS_DIR, "custom_models", "structure_plot.png")
    var_auto, fun_auto, _ = take_variables_and_functions_from_ispl("/home/bogdan/Downloads/bladder_models/bladder_models_aux/bladder_0.ispl")
    #
    bn = BN_Realisation(var_auto, fun_auto)
    # #
    bn.find_all_attractors(path_to_file="/home/bogdan/Boolean-networks/PBN_tool/models/custom_models")
    # bn.plotStructure(filename)
    # bn.drawStateTransitionGraph(os.path.join(MODELS_DIR, "custom_models"), "transition_graph_2.png")
    #var = ["x1", "x2", "x5", "x6"]
    #fun = ["x1 & x2", "x1 | !x2", "x2 | x6", "x5 & x6"]
    #var = ["x1", "x2"]
    #fun = ["x1 & x2", "x1 | !x2"]

    #var = ["x1", "x2"]
    #fun = ["x1 & x2", "x1 | !x2"]
    #bn = BN(var, fun)
    #bn.find_all_attractors()
    #bn.drawStateTransitionGraph(os.path.join(MODELS_DIR, "custom_models"), "transition_graph_2.png")
    #var=["x1", "x2"]
    #fun=["x1 & x2", "x1 | !x2"]
    #bn = BN_Realisation(var,fun)
    #bn.find_all_attractors()
    # var = []
    # for i in range (1, 15):
    #     var.append("x" + str(i))
    #
    # fun = [
    #     "!x1 | x2",
    #     "!x1",
    #     "x2 | x5",
    #     "!x3",
    #     "x4",
    #     "x4 & x5 | x9",
    #     "!x6 | !x8",
    #     "!x6",
    #     "x7",
    #     "!x8",
    #     "!x10 | x14",
    #     "!x11",
    #     "x12 | !x11",
    #     "x13"
    # ]
    #
    # bn = BN(var, fun)
    # l1 = bn.find_all_attractors()
    # z = bn.getAttractors()
    #
    # for x,y in z.items():
    #     print(len(y))
    #
    # bn = BN_Realisation(var, fun)
    # l2 = bn.find_all_attractors()

    # vars = []
    # for i in range(10):
    #     var = "x"+str(i)
    #     vars.append(var)
    #
    # fun = ["x2",
    #        "x2 | ~x3 | (x0 & x4)",
    #        "(x2 & x8 & ~x5) | (x4 & x5 & ~x2 & ~x8) | (x5 & x8 & ~x2 & ~x4) | (~x2 & ~x4 & ~x5 & ~x8)",
    #        "(x4 & ~x1) | (~x4 & ~x9)",
    #        "(~x8 & ~x9) | (x8 & x9 & ~x3) | (x3 & ~x2 & ~x8) | (x3 & ~x2 & ~x9)",
    #         "~x5",
    #         "(x1 & ~x5) | (~x2 & ~x9) | (~x5 & ~x9) | (x5 & x9 & ~x1)",
    #         "x0",
    #         "x7",
    #             "x1"]
    # var = ["x0", "x1", "x2"]
    # fun = ["~x0",
    #        "~x1",
    #        "(x0 & ~x2) | (x2 & ~x0)"]
    # bn = BN(var, fun)
    # bn.plotStructure(os.path.join(MODELS_DIR, "custom_models", "structure_plot.png"))
    # l1 = bn.find_all_attractors()
    # bn = BN_Realisation(var, fun)
    # l2 = bn.find_all_attractors()
    #bn.drawStateTransitionGraph(os.path.join(MODELS_DIR, "custom_models"), "transition_graph_2.png")
    #print(if_equal(l1,l2))

    # answers = []
    # k = 0
    # for i in range(100):
    #     bn = BN.generateRandomBN(10, 5, 1)
    #     nodes = bn.node_names
    #     #print(nodes)
    #     fun = bn.functions_str
    #     #print(fun)
    #     qn = BN_Realisation(nodes, fun)
    #     #l1 = bn.find_all_attractors()
    #     l2 = qn.find_all_attractors(path_to_file="/home/bogdan/Boolean-networks/PBN_tool/models/custom_models")
    #     #z = if_equal(l1, l2)

        # if not z:
        #     k = 1321
        #     break
        #answers.append(if_equal(l1,l2))


    #
    # bn = BN_Realisation.generateRandomBN(20, 4, 1)
    # nodes = bn.node_names
    # fun = bn.functions_str





    #var = ["x1", "x2","x3","x4","x5","x6", "x7", "x8"]
    # fun = ["x1 & x2", "x1 | !x2", "!x4 & x3", "x1 | x3", "x2 | x6", "x5 & x6", "(x1 | x6)& x8", "x7 | x8"]
    # bn = BN(var, fun)
    # bn.find_all_attractors()
    # bn.drawStateTransitionGraph(os.path.join(MODELS_DIR, "custom_models"), "transition_graph_2.png")
    # print(bn.getAttractors())

    #bn = BN_Realisation(var, fun)
    #bn.find_all_attractors()


    #bn = BN(var, fun)
    #bn.find_all_attractors()
    #print("==================================", flush=True)
    #bn = BN_Realisation(var, fun)
    #bn.find_all_attractors()

    #bn = BN(var, fun)
    #bn.find_all_attractors()
    #bn.drawStateTransitionGraph(os.path.join(MODELS_DIR, "custom_models"), "transition_graph_1.png")
    # bn = BN(var, fun)
    # bdd = bn.get_BDD()
    # all_bdd_functions = bn.bdd_expressions
    # bdd.declare("x3", "x3_next")
    # all_bdd_functions.append(bdd.add_expr("!x2"))
    # attractors1 = bn.find_all_attractors()
    # attractor_dynamics = attractors1[0] & bn.bdd_transition_matrix
    # for models in bdd.pick_iter(attractor_dynamics):
    #     state = [
    #         int(models[node_name]) for node_name in bn.node_names
    #     ]
    #     state_next = [
    #         int(models[next_name]) for next_name in bn.next_variables
    #     ]
    #     print(f"State: {state, state_next}", flush=True)
    # print("============================================")
    #
    # #post = self.__bdd.exist(self.node_names, pre)
    # #project_dynamics = bdd.exist(["x1", "x1_next"], attractor_dynamics)
    # project_dynamics = attractor_dynamics
    # eq = lambda a, b: ~bdd.apply('xor', a, b)
    # vars = ["x1", "x2", "x3"]
    # vars_next = ["x1_next", "x2_next", "x3_next"]
    # R_2 = bdd.true
    # i = 2
    # for j in range(3):
    #     if i != j:  # The case when i-th node is updated
    #         # Here R_i represents the logic: xj_next = x_j.
    #         R_2 &= eq(bdd.var(vars_next[j]), bdd.var(vars[j]))
    #     else:  # The case: xj_next = xj for j != i.
    #         R_2 &= eq(bdd.var(vars_next[j]),bdd.add_expr("!x2"))
    # #R_2 = eq(bdd.var(vars_next[2]),bdd.add_expr("!x2"))
    # project_dynamics &= eq(bdd.var(vars[2]),bdd.var(vars_next[2]))
    # project_dynamics |= R_2
    # i = 1
    # for models in bdd.pick_iter(project_dynamics, care_vars=vars+vars_next):
    #     print(i)
    #     state = [
    #         int(models[node_name]) for node_name in vars
    #     ]
    #     state_next = [
    #         int(models[next_name]) for next_name in vars_next
    #     ]
    #     print(f"State: {state, state_next}", flush=True)
    #     i+=1
    #
    # fun += ["!x2"]
    # var += ["x3"]
    #
    # bn = BN(var, fun)
    # bn.find_all_attractors()
    # bdd.dump("new_bdd.png", roots=[attractor_dynamics])

    #var = var + ["x3"]
    #fun = fun + ["!x2"]

    #var2 = ["x4", "x5", "x6"]
    # fun2=["!x1 & !x2", "x1"]
    #
    # bn = BN(var + var2, fun+fun2)
    # bn.find_all_attractors()
    #bn = BN(var1 + var2, fun1 + fun2)
    #bn.find_all_attractors()
    #var1 = ["x4", "x5"]
    #
    # fun = [
    #     "x1 & x2 & x3",
    #     "x2 | x3",
    #     "(x1 | x2) & x3"
    # ]
    # fun1 = [
    #     "~x4 | x5",
    #     "x5 & x4"
    # ]
    #
    # var2 = ["x7", "x8"]
    # fun2 = [
    #     "~x8 | ((x1 & x2 | ~x3) & (x4 & ~x5))",
    #     "x7 | x5"
    # ]
    #
    # bn = BN(var, fun)
    # bn.find_all_attractors()
    # dn = BN(var1, fun1)
    # dn.find_all_attractors()
    # zn = BN(var + var1, fun + fun1)
    # zn.find_all_attractors()
    # hn = BN(var + var1 + var2, fun + fun1 + fun2)
    # print("FINAL")
    # hn.find_all_attractors()

    #var = var + var1

#     var_auto, fun_auto, _ = take_variables_and_functions_from_ispl("/home/bogdan/Boolean-networks/PBN_tool/tests/bn_examples/bn_5.ispl")
#     print(var_auto, fun_auto)
#     bn = BN(var_auto, fun_auto)
#     bn.find_all_attractors()
    # var = ["x0","x1","x2","x3", "x4"]
    # fun = [
    #     "~x2 | ~x4",
    #     "(x2 & x3 & x4) | (x2 & x4 & ~x1) | (x1 & x4 & ~x2 & ~x3) | (x3 & ~x1 & ~x2 & ~x4)",
    #     "x1 & x2",
    #     "x1 | ~x0",
    #     "x2 & ~x0"
    # ]
    # bn = BN(var, fun)
    # bn.find_all_attractors()
    # var = ["x1","x2", "x3", "x4", "x5"]
    # fun = [
    #     "x1 & x2",
    #     "!x1",
    #     "(x2 | !x3) & x5",
    #     "!x3 & x4",
    #     "x3 & x4"
    # ]
    # var = ["v" + str(i) for i in range(1, 9)]
    # fun = [
    #     "v1 & v2",
    #     "v1 | !v2",
    #     "!v4",
    #     "v1 & !v3",
    #     "v2 & v6",
    #     "v5",
    #     "(v1 & v6) & v8",
    #     "v7 | v8"
    # ]
    # bn = BN(var, fun)
    # bn.find_all_attractors()
    # bn.find_all_attractors_with_block_splitting()




    # bn = BN(var, fun)
    # bn.find_all_attractors()
    # bn.drawStateTransitionGraph(os.path.join(MODELS_DIR, "custom_models"), "transition_graph_2.png")
    # # var = []
    # for i in range(80):
    #     var.append("x"+str(i))

    # fun = [
    # "~x74",
    # "(x13 & x59) | (x13 & ~x32) | (x59 & ~x32) | (x32 & ~x13 & ~x59)",
    # "~x58",
    # "(~x5 & ~x50) | (x5 & x50 & ~x27)",
    # "x69 & ~x12",
    # "~x1",
    # "~x19",
    # "x66 & ~x67",
    # "~x36 | ~x61",
    # "x9",
    # "(x0 & ~x35) | (x0 & ~x66) | (~x35 & ~x66) | (x35 & x66 & ~x0)",
    # "x41 | (x21 & ~x57) | (x57 & ~x21)",
    # "x11 | x69",
    # "(x61 & x7) | (x7 & ~x2) | (x2 & ~x61 & ~x7)",
    # "x68",
    # "(x43 & ~x22) | (~x13 & ~x22)",
    # "x75",
    # "~x55",
    # " x3 & ~x36",
    # " ~x10 & ~x2",
    # " (x12 & ~x16) | (x16 & ~x12 & ~x13)",
    # " (x3 & x7) | (x7 & ~x27)",
    # " x42",
    # " ~x74",
    # " ~x67",
    # " x72 | ~x68",
    # " x54",
    # " x50 & ~x76",
    # " (x34 & x50 & ~x36) | (x36 & x50 & ~x34) | (~x34 & ~x36 & ~x50)",
    # " (x18 & x53) | (~x18 & ~x53)",
    # " (x16 & x45 & ~x67) | (x67 & ~x16 & ~x45)",
    # " x39 | ~x8",
    # " ~x71",
    # " ~x23",
    # " (~x43 & ~x74) | (x43 & x74 & ~x67)",
    # " (x20 & x79) | (x53 & ~x79) | (~x20 & ~x79)",
    # " (x11 & x18 & ~x3) | (x18 & x3 & ~x11)",
    # " (x36 & x43 & ~x5) | (x36 & x5 & ~x43) | (x43 & x5 & ~x36)",
    # " ~x33",
    # " x9 & ~x17",
    # " x69",
    # " x77",
    # " (x9 & ~x8) | (~x6 & ~x8) | (x6 & x8 & ~x9)",
    # " ~x38",
    # " ~x20",
    # " ~x49",
    # " ~x7",
    # " ~x73",
    # " x17",
    # " (x43 & x47 & x67) | (~x43 & ~x47 & ~x67)",
    # " (x14 & ~x24) | (x24 & ~x14) | (x28 & ~x14)",
    # " x21",
    # " (x44 & x56) | (~x44 & ~x56)",
    # " x21",
    # " (x54 & x57) | (x54 & ~x0) | (x57 & ~x0) | (x0 & ~x54 & ~x57)",
    # " (x10 & x61 & ~x70) | (x70 & ~x10 & ~x61)",
    # " (x6 & x75) | (x52 & ~x75)",
    # " x32 | x38",
    # " ~x44",
    # " x33 | x74",
    # " ~x50",
    # " x16 & ~x61",
    # " (x38 & x75) | (~x38 & ~x58)",
    # " (x23 & x65) | (x38 & ~x65) | (~x23 & ~x38)",
    # " ~x69",
    # " ~x49",
    # " x34",
    # " (~x39 & ~x48) | (x39 & x48 & ~x37)",
    # " (x24 & ~x9) | (x9 & ~x24)",
    # " (x52 & x75 & x79) | (~x52 & ~x75)",
    # " x31",
    # " ~x19",
    # " x25 & x55 & ~x63",
    # " x33 & x69",
    # " ~x29",
    # " (x16 & ~x25 & ~x7) | (x7 & ~x16 & ~x25)",
    # " x75 | (x4 & ~x23)",
    # " x28",
    # " x1",
    # " ~x62"
    # ]
    #bn = BN(var, fun)
    #bn.find_all_attractors()
    # This is some example of typicall usage of BN class.
    # Create variable names. It could be done manually or from the ispl file.
    # var_manually = ["x0", "x1", "x2", "x3", "x4"]
    # fun_manually = [
    #     "(x0 & ~x1 & ~x2) | (x2 & ~x0 & ~x1)",
    #     "~x3 | (x0 & x2)",
    #     "x4 & ~x1 & ~x2",
    #     "(x0 & x1 & x3) | (x3 & ~x0 & ~x1)",
    #     "(x1 & x2) | (x2 & ~x3) | (x3 & ~x1 & ~x2)"
    # ]
    #var_auto, fun_auto, _ = take_variables_and_functions_from_ispl(MODELS_DIR + "/tlgl_28.ispl")

    # Create the BN reprsented by above var and fun.
    #bn_manually = BN(var_manually, fun_manually, mode="asynchronous")
    #bn_auto = BN(var_auto, fun_auto, mode="asynchronous")

    # If the network is small - you can draw its transition graph and
    # manually open the picture.
    # bn_manually.drawStateTransitionGraph(folder=home_path,
    #                                      filename="picture_2a.png",
    #                                      highlightAttractors=True)

    # To find all attractors you simply write:
    #all_attractors = bn_manually.find_all_attractors()

    # This version of BN also allows one to play with
    # forward-edgetics as follows:
    # (1) choose the name of the node for the modification,
    # (2) choose the name of node which will be deleted from the (1).
    # (3) give some fixed state of BN.
    # (4) give the number of steps to be done with modified BN.
    # It will output all reachable attractors from all reached states after n-steps forward.
    # init_state = []
    # for c in range(bn_manually.num_nodes):
    #     init_state.append(0)
    #
    # bn_manually.forward_edgetics("x0",
    #                              "x1",
    #                              1000,
    #                              init_state)

    # bn = BN.generateRandomBN(5, 4, 2)
    # filename = os.path.join(MODELS_DIR, "custom_models", "structure_plot.png")
    # bn.plotStructure(filename)
    #
    # # Plot the transition graph.
    # bn.drawStateTransitionGraph(os.path.join(MODELS_DIR, "custom_models"), "transition_graph.png")

