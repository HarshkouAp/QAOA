import numpy as np
from scipy.optimize import minimize
import time
from Graph_generator import *
import warnings
warnings.filterwarnings('ignore')


def QAOA(Graph, P_param, max_iter=10000, callback=False):

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
            for h in range(2**N):
                if preset[j][h] >= 2**(N-1):
                    preset[j][h] = (2**N - preset[j][h]) - 1
            preset[j] = preset[j][0:2**(N-1)]
        return np.array(preset)

    def Hamiltonian():
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
        hamiltonian = hamiltonian[0:2**(N-1)]
        return hamiltonian * (-1)

    def Black_box_function(ang_arr):

        global Iterator
        Iterator += 1
        gamma = ang_arr[0:P_param]
        beta = ang_arr[P_param:2 * P_param]
        state = np.copy(Superposition)
        for k in range(P_param):
            vector = np.exp(-1j * gamma[k] * Hamiltonian) * state
            for j in range(N):
                vector = (vector * np.cos(beta[k]) - 1j * vector[Mix_operator_preset[j]] * np.sin(beta[k]))
            state = vector
        energy = 2 * np.sum(np.conj(state) * Hamiltonian * state)

        return energy

    def Final_state(p_param, ang_arr):
        gamma = ang_arr[0:p_param]
        beta = ang_arr[p_param:2 * p_param]
        state = np.copy(Superposition)
        for k in range(P_param):
            vector = np.exp(-1j * gamma[k] * Hamiltonian) * state
            for j in range(N):
                vector = (vector * np.cos(beta[k]) - 1j * vector[Mix_operator_preset[j]] * np.sin(beta[k]))
            state = vector
        energy = 2 * np.sum(np.conj(state) * Hamiltonian * state)

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

    def MaxCut_clasical_solver():
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

    N = nx.number_of_nodes(Graph)
    N_e = nx.number_of_edges(Graph)

    Hamiltonian = Hamiltonian()
    Mix_operator_preset = Mix_operator_preset()
    Superposition = (np.ones(2**(N-1)) / (np.sqrt(2)**N))
    global Iterator
    Iterator = 0

    Start_time = time.time()
    Init_params = np.ones(2 * P_param)
    Result_min = minimize(Black_box_function, Init_params, method='COBYLA', options={'maxiter': max_iter})
    End_time = time.time()
    Processing_time_QAOA = End_time - Start_time

    Optimal_angles = Result_min.x
    F_state, Energy = Final_state(P_param, Optimal_angles)
    Solution, Probability = Solution(F_state)

    Start_time = time.time()
    C_solution, C_energy = MaxCut_clasical_solver()
    End_time = time.time()
    Processing_time_Cl = End_time - Start_time

    if callback:
        print("\033[37m {}".format(f"--------------------------"))
        if Solution[0] in C_solution:
            print("\033[32m {}".format("***SUCCESS***"))
            result = 1
        else:
            print("\033[31m {}".format("***FAIL***"))
            result = 0
        print(f"Number of nodes : {N}  P_param : {P_param}")
        print(f"Iterations : {Iterator}")
        print(f"QAOA solution : {Solution[0]}   Probability : {Probability}   Energy : {np.real(Energy)}")
        print(f"Classical solution : {C_solution}   Energy : {C_energy}")
        print(f"QAOA time : {Processing_time_QAOA} s   Classic time : {Processing_time_Cl} s")

    return (Solution, Probability, np.real(Energy), C_solution, C_energy, Iterator,
            Processing_time_QAOA, Processing_time_Cl)

