import numpy as np
from pandas.core.algorithms import map_array
from scipy.optimize import minimize
import time
from Graph_generator import *
import warnings
warnings.filterwarnings('ignore')

def QAOA(graph, p_param, max_iter=10000, callback=False, logs=False, method='L-BFGS-B'):

    def mix_operator_preset():
        preset = []
        for qubit in range(1, n_nodes + 1):
            permutations = []
            for x in state_list:
                if x[qubit - 1] == "0":
                    perm = int(x, 2) + 2 ** (n_nodes - qubit)
                    if perm >= 2 ** (n_nodes - 1):
                        permutations.append(2 ** n_nodes - perm - 1)
                    else:
                        permutations.append(perm)
                else:
                    perm = int(x, 2) - 2 ** (n_nodes - qubit)
                    if perm >= 2 ** (n_nodes - 1):
                        permutations.append(2 ** n_nodes - perm - 1)
                    else:
                        permutations.append(perm)
            preset.append(permutations)

        return np.array(preset)

    def cost_function(depth, final=False):

        def function(ang_arr):
            gamma = ang_arr[0:depth]
            beta = ang_arr[depth:2 * depth]
            state = np.copy(superposition)
            for k in range(depth):
                vector = np.exp(-1j * gamma[k] * hamiltonian) * state
                for j in range(n_nodes):
                    vector = (vector * np.cos(beta[k]) - 1j * vector[mix_operator_preset[j]] * np.sin(beta[k]))
                state = vector
            energy = 2 * np.sum(np.conj(state) * hamiltonian * state)
            if final:
                return state, np.abs(energy)
            else:
                return energy

        return function

    def optimization():

        init_params = np.ones(2)
        for p in range(1, p_param + 1):
            result_min = minimize(cost_function(p), init_params, method=method, options={'maxiter': max_iter})
            optimal_angles = result_min.x
            init_params = np.insert(optimal_angles, round(len(optimal_angles) / 2), 1)
            init_params = np.append(init_params, 1)
            fin_function = cost_function(p, final=True)
            s, e = fin_function(optimal_angles)
            if logs:
                print(f"Глубина : {p}  Энергия : {e}")

        return s, e

    def maxcut_classical_solver_weighted():

        edges_weights = nx.get_edge_attributes(graph, "weight")
        optimal_state_list = []
        cut_arr = np.array([])
        optimal_cut = 0

        for s in state_list:
            cut = 0
            for k, j in graph.edges():
                if s[k] != s[j]:
                    cut -= edges_weights[(k, j)]
            cut_arr = np.append(cut_arr, cut)
            if cut == optimal_cut:
                optimal_state_list.append(s)
            elif cut < optimal_cut:
                optimal_cut = cut
                optimal_state_list = [s]

        return optimal_state_list, optimal_cut, cut_arr

    n_nodes = nx.number_of_nodes(graph)

    state_list = [("0" * n_nodes)[:n_nodes - len(bin(s)[2:])] + bin(s)[2:] for s in range(2 ** (n_nodes - 1))]
    superposition = (np.ones(2 ** (n_nodes - 1)) / (np.sqrt(2) ** n_nodes))
    result = 0

    start_time = time.time()
    c_solution, c_energy, hamiltonian = maxcut_classical_solver_weighted()
    processing_time_cl = time.time() - start_time

    mix_operator_preset = mix_operator_preset()

    start_time = time.time()
    state, energy = optimization()
    processing_time_qaoa = time.time() - start_time

    solution = state_list[np.argmax(np.abs(state) ** 2)]
    probability = np.max(np.abs(state) ** 2) * 2

    if callback:
        print("\033[37m {}".format(f"-------------------------------------"))
        if solution in c_solution:
            print("\033[32m {}".format("***SUCCESS***"))
            result = 1
        else:
            print("\033[31m {}".format("***FAIL***"))
            result = 0
        print(f"Number of nodes : {n_nodes}  Number of edges : {nx.number_of_edges(graph)} ")
        print(f"QAOA solution : {solution}   probability : {probability}   energy : {energy}")
        print(f"P_param : {p_param}   Optimization method : {method}")
        print(f"Classical solution : {c_solution}   energy : {c_energy}")
        print(f"QAOA time : {processing_time_qaoa} s   Classic time : {processing_time_cl} s")

    return (result, solution, probability, energy, c_solution, c_energy,
            processing_time_qaoa, processing_time_cl)
