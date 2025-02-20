from scipy.optimize import minimize
import time
from Graph_generator import *
import warnings
warnings.filterwarnings('ignore')


def QAOA(graph, p_param, max_iter=10000, callback=False, weighted=False, logs=False, method='L-BFGS-B', strategy="consistent"):

    def mix_operator_preset():
        X = np.array([[0, 1], [1, 0]])
        X_layer = 1
        preset = []

        for i in range(n_nodes):
            for k in range(n_nodes):
                if i == k:
                    X_layer = np.kron(X, X_layer)
                else:
                    X_layer = np.kron(np.eye(2), X_layer)
            preset.append([np.argmax(row) for row in X_layer])
            X_layer = 1
        for j in range(n_nodes):
            for h in range(2 ** n_nodes):
                if preset[j][h] >= 2 ** (n_nodes - 1):
                    preset[j][h] = (2 ** n_nodes - preset[j][h]) - 1
            preset[j] = preset[j][0:2 ** (n_nodes - 1)]
        return np.array(preset)

    def hamiltonian_unweighted():
        I_arr = np.ones(2 ** n_nodes)
        Z_pauli = np.array([1, -1])
        ZZ_operator = np.zeros(2 ** n_nodes)
        ZZ_layer = 1
        for j in range(n_nodes):
            for k in range(j, n_nodes):
                if j != k:
                    arr = []
                    if j in list(nx.all_neighbors(graph, k)):
                        for h in range(n_nodes):
                            if (h == j) or (h == k):
                                arr.append("Z")
                                ZZ_layer = np.kron(Z_pauli, ZZ_layer)
                            else:
                                ZZ_layer = np.kron(np.ones(2), ZZ_layer)
                                arr.append("I")
                        ZZ_operator += ZZ_layer
                    ZZ_layer = 1
        hamiltonian = (n_edges * I_arr - ZZ_operator) / 2
        hamiltonian = hamiltonian[0:2 ** (n_nodes - 1)]
        return hamiltonian * (-1)

    def hamiltonian_weighted():
        edges_weights = nx.get_edge_attributes(graph, "weight")
        I_arr = np.ones(2 ** n_nodes)
        Z_pauli = np.array([1, -1])
        ZZ_operator = np.zeros(2 ** n_nodes)
        ZZ_layer = 1
        for j in range(n_nodes):
            for k in range(j, n_nodes):
                if j != k:
                    arr = []
                    if j in list(nx.all_neighbors(graph, k)):
                        for h in range(n_nodes):
                            if (h == j) or (h == k):
                                arr.append("Z")
                                ZZ_layer = np.kron(Z_pauli, ZZ_layer)
                            else:
                                ZZ_layer = np.kron(np.ones(2), ZZ_layer)
                                arr.append("I")
                        ZZ_operator += edges_weights[(j, k)] * (I_arr - ZZ_layer) / 2
                    ZZ_layer = 1
        hamiltonian = ZZ_operator[0:2 ** (n_nodes - 1)]
        return hamiltonian * (-1)

    def cost_function(depth, stage=0, pre_params=[], iterative=False):

        if iterative:

            def function(ang_arr):

                gamma = np.ones(p_param)
                beta = np.ones(p_param)

                if stage == 1:
                    for i in range(int(len(pre_params) / 2)):
                        gamma[i] = pre_params[i]
                        beta[i] = pre_params[i + int(len(pre_params) / 2)]
                    gamma[int(len(pre_params) / 2)] = ang_arr[0]
                    beta[int(len(pre_params) / 2)] = ang_arr[1]

                if stage == 2:
                    for i in range(int(len(ang_arr) / 2)):
                        gamma[i] = ang_arr[i]
                        beta[i] = ang_arr[i + int(len(ang_arr) / 2)]
                state = np.copy(superposition)
                for k in range(depth):
                    vector = np.exp(-1j * gamma[k] * hamiltonian) * state
                    for j in range(n_nodes):
                        vector = (vector * np.cos(beta[k]) - 1j * vector[mix_operator_preset[j]] * np.sin(beta[k]))
                    state = vector
                energy = 2 * np.sum(np.conj(state) * hamiltonian * state)

                return energy

        else:

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

                return energy

        return function

    def optimization():
        state = 0
        energy = 0
        optimal_angles = []

        if strategy == "parallel":

            init_params = np.ones(2 * p_param)
            function = cost_function(p_param)
            result_min = minimize(function, init_params, method=method, options={'maxiter': max_iter})
            optimal_angles = result_min.x
            state, energy = final_state(p_param, optimal_angles)

        if strategy == "consistent":

            init_params = np.ones(2)
            for p in range(1, p_param + 1):

                function = cost_function(p)
                result_min = minimize(function, init_params, method=method, options={'maxiter': max_iter})
                optimal_angles = result_min.x
                init_params = np.insert(optimal_angles, round(len(optimal_angles) / 2), 1)
                init_params = np.append(init_params, 1)
                state, energy = final_state(p, optimal_angles)
                if logs:
                    print(f"Глубина : {p}  Энергия : {energy}")
        
        if strategy == "iterative":

            for p in range(1, p_param + 1):

                function = cost_function(p_param, iterative=True, stage=1, pre_params=optimal_angles)
                result_min = minimize(function, np.ones(2), method=method, options={'maxiter': max_iter})
                angles = result_min.x


                if logs:
                    state, energy = final_state(p_param, angles, stage=1, pre_params=optimal_angles, iterative=True)
                    print(f"Глубина : {p}  Этап : {1}  Энергия : {energy}")

                optimal_angles = np.insert(optimal_angles, round(len(optimal_angles) / 2), angles[0])
                optimal_angles = np.append(optimal_angles, angles[1])

                function = cost_function(p_param, iterative=True, stage=2, pre_params=optimal_angles)
                result_min = minimize(function, optimal_angles, method=method, options={'maxiter': max_iter})
                optimal_angles = result_min.x


                if logs:
                    state, energy = final_state(p_param, optimal_angles, stage=2, pre_params=optimal_angles, iterative=True)
                    print(f"Глубина : {p}  Этап : {2}  Энергия : {energy}")

        return state, energy

    def final_state(depth, ang_arr, stage=0, pre_params=[], iterative=False):

        state = 0
        energy = 0

        if iterative:
            gamma = np.ones(p_param)
            beta = np.ones(p_param)

            if stage == 1:
                for i in range(int(len(pre_params) / 2)):
                    gamma[i] = pre_params[i]
                    beta[i] = pre_params[i + int(len(pre_params) / 2)]
                gamma[int(len(pre_params) / 2)] = ang_arr[0]
                beta[int(len(pre_params) / 2)] = ang_arr[1]

            if stage == 2:
                for i in range(int(len(ang_arr) / 2)):
                    gamma[i] = ang_arr[i]
                    beta[i] = ang_arr[i + int(len(ang_arr) / 2)]
        else:
            gamma = ang_arr[0:depth]
            beta = ang_arr[depth:2 * depth]

        state = np.copy(superposition)
        for k in range(depth):
            vector = np.exp(-1j * gamma[k] * hamiltonian) * state
            for j in range(n_nodes):
                vector = (vector * np.cos(beta[k]) - 1j * vector[mix_operator_preset[j]] * np.sin(beta[k]))
            state = vector
        energy = np.abs(2 * np.sum(np.conj(state) * hamiltonian * state)) * (-1)

        return state, energy

    def solution(amplitudes):
        num_of_states = 2 ** n_nodes
        prob_arr = np.abs(amplitudes.reshape(1, -1))[0] ** 2
        state_arr = ['0' * n_nodes]
        for k in range(num_of_states):
            for j in range(n_nodes):
                if (k < 2 ** (j + 1)) and (k >= 2 ** j):
                    state_arr.append(f"{'0' * (n_nodes - j - 1)}{format(k, 'b')}"[::-1])

        max_ind = np.argmax(prob_arr)
        solution = [state_arr[max_ind], state_arr[(num_of_states - max_ind) - 1]]
        probability = prob_arr[max_ind] * 2

        return solution, probability

    def maxcut_classical_solver_unweighted():
        num_of_states = 2 ** n_nodes
        state_arr = np.array(['0' * n_nodes])
        for k in range(num_of_states):
            for j in range(n_nodes):
                if (k < 2 ** (j + 1)) and (k >= 2 ** j):
                    state_arr = np.append(state_arr, f"{'0' * (n_nodes - j - 1)}{format(k, 'b')}")

        cut_arr = np.array([])
        for state in state_arr:
            cut = 0
            for k, j in graph.edges():
                if state[k] != state[j]:
                    cut -= 1
            cut_arr = np.append(cut_arr, cut)

        solution = state_arr[np.where(cut_arr == np.min(cut_arr))[0]]
        energy = cut_arr[np.argmin(cut_arr)]

        return solution, energy

    def maxcut_classical_solver_weighted():

        edges_weights = nx.get_edge_attributes(graph, "weight")
        num_of_states = 2 ** n_nodes
        state_arr = np.array(['0' * n_nodes])
        for k in range(num_of_states):
            for j in range(n_nodes):
                if (k < 2 ** (j + 1)) and (k >= 2 ** j):
                    state_arr = np.append(state_arr, f"{'0' * (n_nodes - j - 1)}{format(k, 'b')}")

        cut_arr = np.array([])
        for state in state_arr:
            cut = 0
            for k, j in graph.edges():
                if state[k] != state[j]:
                    cut -= edges_weights[(k, j)]
            cut_arr = np.append(cut_arr, cut)

        solution = state_arr[np.where(cut_arr == np.min(cut_arr))[0]]
        energy = cut_arr[np.argmin(cut_arr)]

        return solution, energy

    n_nodes = nx.number_of_nodes(graph)
    n_edges = nx.number_of_edges(graph)


    if weighted:
        hamiltonian = hamiltonian_weighted()
    else:
        hamiltonian = hamiltonian_unweighted()
    mix_operator_preset = mix_operator_preset()
    superposition = (np.ones(2 ** (n_nodes - 1)) / (np.sqrt(2) ** n_nodes))
    result = 0

    start_time = time.time()
    state, energy = optimization()
    end_time = time.time()
    processing_time_qaoa = end_time - start_time

    solution, probability = solution(state)

    start_time = time.time()
    if weighted:
        c_solution, c_energy = maxcut_classical_solver_weighted()
    else:
        c_solution, c_energy = maxcut_classical_solver_unweighted()
    end_time = time.time()
    processing_time_cl = end_time - start_time

    if callback:
        print("\033[37m {}".format(f"-------------------------------------"))
        if solution[0] in c_solution:
            print("\033[32m {}".format("***SUCCESS***"))
            result = 1
        else:
            print("\033[31m {}".format("***FAIL***"))
            result = 0
        print(f"Number of nodes : {n_nodes}  P_param : {p_param}")
        print(f"QAOA solution : {solution[0]}   probability : {probability}   energy : {energy}")
        print(f"Optimization strategy : {strategy}   Method : {method}")
        print(f"Classical solution : {c_solution}   energy : {c_energy}")
        print(f"QAOA time : {processing_time_qaoa} s   Classic time : {processing_time_cl} s")

    return (result, solution, probability, energy, c_solution, c_energy,
            processing_time_qaoa, processing_time_cl)


# G = generate_graph(8, 0.7, weighted=False, visualise=False)
# QAOA(graph=G, p_param=10, callback=True, weighted=False, max_iter=5000, logs=True, strategy="consistent")

