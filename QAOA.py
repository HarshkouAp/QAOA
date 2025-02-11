from scipy.optimize import minimize
import time
from Graph_generator import *
import warnings
warnings.filterwarnings('ignore')


def QAOA(Graph, P_param, max_iter=10000, callback=False, weighted=False):

    def Mix_operator_preset():
        X = np.array([[0, 1], [1, 0]])
        X_layer = 1
        preset = []

        for i in range(N):
            for k in range(N):
                if i == k:
                    X_layer = np.kron(X, X_layer)
                else:
                    X_layer = np.kron(np.eye(2), X_layer)
            preset.append([np.argmax(row) for row in X_layer])
            X_layer = 1
        for j in range(N):
            for h in range(2 ** N):
                if preset[j][h] >= 2 ** (N - 1):
                    preset[j][h] = (2 ** N - preset[j][h]) - 1
            preset[j] = preset[j][0:2 ** (N - 1)]
        return np.array(preset)

    def Hamiltonian_unweighted():
        I_arr = np.ones(2 ** N)
        Z_pauli = np.array([1, -1])
        ZZ_operator = np.zeros(2 ** N)
        ZZ_layer = 1
        for j in range(N):
            for k in range(j, N):
                if j != k:
                    arr = []
                    if j in list(nx.all_neighbors(Graph, k)):
                        for h in range(N):
                            if (h == j) or (h == k):
                                arr.append("Z")
                                ZZ_layer = np.kron(Z_pauli, ZZ_layer)
                            else:
                                ZZ_layer = np.kron(np.ones(2), ZZ_layer)
                                arr.append("I")
                        ZZ_operator += ZZ_layer
                    ZZ_layer = 1
        hamiltonian = (N_e * I_arr - ZZ_operator) / 2
        hamiltonian = hamiltonian[0:2 ** (N - 1)]
        return hamiltonian * (-1)

    def Hamiltonian_weighted():
        edges_weights = nx.get_edge_attributes(Graph, "weight")
        I_arr = np.ones(2 ** N)
        Z_pauli = np.array([1, -1])
        ZZ_operator = np.zeros(2 ** N)
        ZZ_layer = 1
        for j in range(N):
            for k in range(j, N):
                if j != k:
                    arr = []
                    if j in list(nx.all_neighbors(Graph, k)):
                        for h in range(N):
                            if (h == j) or (h == k):
                                arr.append("Z")
                                ZZ_layer = np.kron(Z_pauli, ZZ_layer)
                            else:
                                ZZ_layer = np.kron(np.ones(2), ZZ_layer)
                                arr.append("I")
                        ZZ_operator += edges_weights[(j, k)] * (I_arr - ZZ_layer) / 2
                    ZZ_layer = 1
        hamiltonian = ZZ_operator[0:2 ** (N - 1)]
        return hamiltonian * (-1)

    def Cost_function(depth):

        def function(ang_arr):

            gamma = ang_arr[0:depth]
            beta = ang_arr[depth:2 * depth]
            state = np.copy(Superposition)
            for k in range(depth):
                vector = np.exp(-1j * gamma[k] * Hamiltonian) * state
                for j in range(N):
                    vector = (vector * np.cos(beta[k]) - 1j * vector[Mix_operator_preset[j]] * np.sin(beta[k]))
                state = vector
            energy = 2 * np.sum(np.conj(state) * Hamiltonian * state)

            return energy

        return function

    def Optimization():

        state = 0
        energy = 0
        optimal_angles = 0
        init_params = np.ones(2)
        for p in range(1, P_param + 1):
            function = Cost_function(p)
            result_min = minimize(function, init_params, method='L-BFGS-B', options={'maxiter': max_iter})
            optimal_angles = result_min.x
            init_params = np.insert(optimal_angles, round(len(optimal_angles) / 2), 1)
            init_params = np.append(init_params, 1)
            state, energy = Final_state(p, optimal_angles)
            if callback:
                print(f"Глубина : {p}, Энергия : {energy}")

        return state, energy

    def Final_state(depth, ang_arr):
        gamma = ang_arr[0:depth]
        beta = ang_arr[depth:2 * depth]
        state = np.copy(Superposition)
        for k in range(depth):
            vector = np.exp(-1j * gamma[k] * Hamiltonian) * state
            for j in range(N):
                vector = (vector * np.cos(beta[k]) - 1j * vector[Mix_operator_preset[j]] * np.sin(beta[k]))
            state = vector
        energy = np.abs(2 * np.sum(np.conj(state) * Hamiltonian * state))

        return state, energy

    def Solution(amplitudes):
        num_of_states = 2 ** N
        prob_arr = np.abs(amplitudes.reshape(1, -1))[0] ** 2
        state_arr = ['0' * N]
        for k in range(num_of_states):
            for j in range(N):
                if (k < 2 ** (j + 1)) and (k >= 2 ** j):
                    state_arr.append(f"{'0' * (N - j - 1)}{format(k, 'b')}"[::-1])

        max_ind = np.argmax(prob_arr)
        solution = [state_arr[max_ind], state_arr[(num_of_states - max_ind) - 1]]
        probability = prob_arr[max_ind] * 2

        return solution, probability

    def MaxCut_classical_solver_unweighted():
        num_of_states = 2 ** N
        state_arr = np.array(['0' * N])
        for k in range(num_of_states):
            for j in range(N):
                if (k < 2 ** (j + 1)) and (k >= 2 ** j):
                    state_arr = np.append(state_arr, f"{'0' * (N - j - 1)}{format(k, 'b')}")

        cut_arr = np.array([])
        for state in state_arr:
            cut = 0
            for k, j in Graph.edges():
                if state[k] != state[j]:
                    cut -= 1
            cut_arr = np.append(cut_arr, cut)

        solution = state_arr[np.where(cut_arr == np.min(cut_arr))[0]]
        energy = cut_arr[np.argmin(cut_arr)]

        return solution, energy

    def MaxCut_classical_solver_weighted():

        edges_weights = nx.get_edge_attributes(Graph, "weight")
        num_of_states = 2 ** N
        state_arr = np.array(['0' * N])
        for k in range(num_of_states):
            for j in range(N):
                if (k < 2 ** (j + 1)) and (k >= 2 ** j):
                    state_arr = np.append(state_arr, f"{'0' * (N - j - 1)}{format(k, 'b')}")

        cut_arr = np.array([])
        for state in state_arr:
            cut = 0
            for k, j in Graph.edges():
                if state[k] != state[j]:
                    cut -= edges_weights[(k, j)]
            cut_arr = np.append(cut_arr, cut)

        solution = state_arr[np.where(cut_arr == np.min(cut_arr))[0]]
        energy = cut_arr[np.argmin(cut_arr)]

        return solution, energy

    N = nx.number_of_nodes(Graph)
    N_e = nx.number_of_edges(Graph)

    if weighted:
        Hamiltonian = Hamiltonian_weighted()
    else:
        Hamiltonian = Hamiltonian_unweighted()
    Mix_operator_preset = Mix_operator_preset()
    Superposition = (np.ones(2 ** (N - 1)) / (np.sqrt(2) ** N))
    Result = 0

    Start_time = time.time()
    State, Energy = Optimization()
    End_time = time.time()
    Processing_time_QAOA = End_time - Start_time

    Solution, Probability = Solution(State)

    Start_time = time.time()
    if weighted:
        C_solution, C_energy = MaxCut_classical_solver_weighted()
    else:
        C_solution, C_energy = MaxCut_classical_solver_unweighted()
    End_time = time.time()
    Processing_time_Cl = End_time - Start_time

    if callback:
        print("\033[37m {}".format(f"--------------------------"))
        if Solution[0] in C_solution:
            print("\033[32m {}".format("***SUCCESS***"))
            Result = 1
        else:
            print("\033[31m {}".format("***FAIL***"))
            Result = 0
        print(f"Number of nodes : {N}  P_param : {P_param}")
        print(f"QAOA solution : {Solution[0]}   Probability : {Probability}   Energy : {Energy}")
        print(f"Classical solution : {C_solution}   Energy : {C_energy}")
        print(f"QAOA time : {Processing_time_QAOA} s   Classic time : {Processing_time_Cl} s")

    return (Result, Solution, Probability, Energy, C_solution, C_energy,
            Processing_time_QAOA, Processing_time_Cl)


# G = generate_graph(8, 0.7, weighted=False, visualise=True)
# QAOA(Graph=G, P_param=25, callback=True, weighted=False, max_iter=5000)

