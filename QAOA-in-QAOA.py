import networkx as nx
import cvxpy as cp
import cvxgraphalgs as cvxgr
import numpy as np
import cupy as cp
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
from os import environ
import warnings

warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)
environ['OMP_NUM_THREADS'] = '8'


def QAOA(Graph, P_param, weighted=False):

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

        return np.array(preset)

    def Cost_operator():
        def function(gamma):
            exp_operator = np.exp(-1j * gamma * Hamiltonian)
            return exp_operator

        return function

    def Mix_operator(vector, beta):
        for i in range(N):
            vector = (vector * np.cos(beta) - 1j * vector[Mix_operator_preset[i]] * np.sin(beta))
        return vector

    def Hamiltonian_standart():
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
        hamiltonian = ZZ_operator
        return hamiltonian

    def Black_box_function(p_param):
        def function(ang_arr):
            gamma = ang_arr[0:p_param]
            beta = ang_arr[p_param:2 * p_param]
            state = np.copy(Superposition)
            for k in range(p_param):
                state = Mix_operator(Cost_operator(gamma[k]) * state, beta[k])
            energy = cp.sum(np.conj(state) * Hamiltonian * state)
            return energy

        return function

    def Final_state(p_param, ang_arr):
        gamma = ang_arr[0:p_param]
        beta = ang_arr[p_param:2 * p_param]
        state = np.copy(Superposition)
        for k in range(p_param):
            state = Mix_operator(Cost_operator(gamma[k]) * state, beta[k])
        energy = cp.sum(np.conj(state) * Hamiltonian * state)
        return state, energy

    def Solution(amplitudes):
        num_of_states = 2 ** N
        prob_arr = np.abs(amplitudes.reshape(1, -1))[0] ** 2
        state_arr = ['0' * N]
        for k in range(num_of_states):
            for j in range(N):
                if (k < 2 ** (j + 1)) and (k >= 2 ** j):
                    state_arr.append(f"{'0' * (N - j - 1)}{format(k, 'b')}"[::-1])

        max_ind = np.argpartition(prob_arr, -2)[-2:]
        solution = [state_arr[max_ind[0]], state_arr[max_ind[1]]]
        probability = prob_arr[max_ind[0]] + prob_arr[max_ind[1]]

        return solution, probability

    def MaxCut_classical_solver_standard():
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
                    cut += edges_weights[(k, j)]
            cut_arr = np.append(cut_arr, cut)

        solution = state_arr[np.where(cut_arr == np.min(cut_arr))[0]]
        energy = cut_arr[np.argmin(cut_arr)]

        return solution, energy

    N = nx.number_of_nodes(Graph)
    N_e = nx.number_of_edges(Graph)

    if weighted:
        Hamiltonian = Hamiltonian_weighted()
    else:
        Hamiltonian = Hamiltonian_standart()
    Cost_operator = Cost_operator()
    Mix_operator_preset = Mix_operator_preset()
    Superposition = (np.ones(2 ** N) / (np.sqrt(2) ** N))
    Init_params = np.array(np.random.random_sample(2 * P_param))
    Black_box = Black_box_function(P_param)

    Start_time = time.time()
    Result_min = minimize(Black_box, Init_params, method='COBYLA', options={'maxiter': 1000})
    End_time = time.time()
    Processing_time_QAOA = End_time - Start_time
    Optimal_angles = Result_min.x
    F_state, Energy = Final_state(P_param, Optimal_angles)

    Solution, Probability = Solution(F_state)

    Start_time = time.time()
    if weighted:
        C_solution, C_energy = MaxCut_classical_solver_weighted()
    else:
        C_solution, C_energy = MaxCut_classical_solver_standard()
    End_time = time.time()
    Processing_time_Cl = End_time - Start_time

    return Solution, Probability, Energy, C_solution, C_energy, Processing_time_QAOA, Processing_time_Cl


def Get_connected_subgraphs(max_size):

    global Layer, Graph_division, Graph_nodes

    connected_components = list(nx.connected_components(G))
    subgraphs = []

    for component in connected_components:
        g = G.subgraph(component)
        nodes = list(g.nodes)

        visited = set()
        for i in range(len(nodes)):
            if nodes[i] in visited:
                continue
            subgraph_nodes = []
            stack = [nodes[i]]

            while stack and len(subgraph_nodes) < max_size:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    subgraph_nodes.append(node)
                    for neighbor in g.neighbors(node):
                        if neighbor not in visited:
                            stack.append(neighbor)

            if len(subgraph_nodes) > 1:
                subgraph = g.subgraph(subgraph_nodes)
                subgraphs.append(list(subgraph.edges))

    iterator = 1
    node_name = max(list(G.nodes()))
    layer_edges = {}
    layer_nodes = {}

    for sg in subgraphs:

        H = nx.Graph()
        H.add_edges_from(sg)
        G.add_node(node_name + iterator)
        layer_edges[node_name + iterator] = list(H.edges())
        layer_nodes[node_name + iterator] = list(H.nodes())

        for i in list(H.nodes()):
            neighbors = list(G.neighbors(i))

            for k in neighbors:
                if k not in list(H.nodes()):
                    G.remove_edge(i, k)
                    G.add_edge(k, node_name + iterator)
            G.remove_node(i)
        iterator += 1

    Graph_division[Layer] = layer_edges
    Graph_nodes[Layer] = layer_nodes
    Layer += 1


def Subgraph_solution(layer):

    solution_dict = {}
    sub = Graph_division[layer]

    if layer == 0:
        for node in sub.keys():
            H = nx.Graph()
            H.add_edges_from(sub[node])
            old_nodes = list(H.nodes())
            new_nodes = list(range(len(old_nodes)))
            mapping = dict(zip(old_nodes, new_nodes))
            H = nx.relabel_nodes(H, mapping)
            S, P, E, C_s, C_e, Pt_Q, Pt_C = QAOA(H, 15)
            solution_dict[node] = S[0]
    else:
        for node in sub.keys():
            H = nx.Graph()
            H.add_weighted_edges_from(sub[node])
            old_nodes = list(H.nodes())
            new_nodes = list(range(len(old_nodes)))
            mapping = dict(zip(old_nodes, new_nodes))
            H = nx.relabel_nodes(H, mapping)
            S, P, E, C_s, C_e, Pt_Q, Pt_C = QAOA(H, 15, weighted=True)
            solution_dict[node] = S[0]

    Sub_solution[layer] = solution_dict


def Weights(layer):

    sub_edges = Graph_division[layer]
    sub_nodes = Graph_nodes[layer - 1]

    for node in sub_edges.keys():
        for k, j in sub_edges[node]:
            weight = 0
            k_solution = Sub_solution[layer - 1][k]
            j_solution = Sub_solution[layer - 1][j]

            for k_node, j_node in zip(sub_nodes[k], sub_nodes[j]):
                if (k_node, j_node) in A_conections:
                    if k_solution[sub_nodes[k].index(k_node)] == j_solution[sub_nodes[j].index(j_node)]:
                        weight += 1
                    else:
                        weight -= 1

            Graph_division[layer][node][sub_edges[node].index((k, j))] += (weight,)


def Recursion(max_size):

    while 1:
        Get_connected_subgraphs(max_size)
        if len(G.nodes()) <= max_size:
            Graph_division[Layer] = {"F": list(G.edges())}
            Graph_nodes[Layer] = {"F": list(G.nodes())}
            break

    for layer in Graph_division.keys():
        if layer != max(Graph_division.keys()):
            Subgraph_solution(layer)
            Weights(layer + 1)
        else:
            Subgraph_solution(layer)


def Reformulate_solution():

    for layer in list(Sub_solution.keys())[-1:0:-1]:
        for node in Sub_solution[layer].keys():
            sub = Sub_solution[layer][node]
            for ind in range(len(sub)):
                if sub[ind] == "1":
                    subnodes = Graph_nodes[layer][node]
                    Sub_solution[layer - 1][subnodes[ind]] = Invert(Sub_solution[layer - 1][subnodes[ind]])


def Invert(solution):

    inv_solution = ""
    for i in solution:
        if i == "1":
            inv_solution += "0"
        else:
            inv_solution += "1"

    return inv_solution


def Answer():

    ans = []

    for node in Sub_solution[0]:
        for ind in range(len(Sub_solution[0][node])):
            ans.append([Graph_nodes[0][node][ind], Sub_solution[0][node][ind]])

    ans = np.array(ans)
    Df = pd.DataFrame(ans, columns=["node", "solution"])
    Df["node"] = Df["node"].apply(lambda x: int(x))
    Df.sort_values("node", inplace=True)

    ans = ""
    for i in Df["solution"]:
        ans += i

    return ans


def Goemans_williamson_weighted():
    sdp_cut = cvxgr.algorithms.goemans_williamson_weighted(A)
    return sdp_cut.evaluate_cut_size(A)


def Classical_solution(state):
    cut = 0
    for k, j in A.edges():
        if state[k] != state[j]:
            cut += 1
    return cut


Data = pd.DataFrame(columns=["N_nodes", "Seed", "Max_size", "Q_energy", "C_energy"])


for seed in range(1000, 1100):
    try:
        number_of_nodes = 20
        density = 0.5
        number_of_edges = number_of_nodes * (number_of_nodes - 1) * density / 2
        real_density = round(2 * int(number_of_edges) / (number_of_nodes * (number_of_nodes - 1)), 4)
        G = nx.dense_gnm_random_graph(number_of_nodes, int(number_of_edges), seed)
        A = G.copy()
        A_conections = list(A.edges())

        options = {'node_size': 1000, 'width': 1}
        nx.draw(G, nx.circular_layout(G), with_labels=False, **options)
        ax = plt.gca()
        node_lab = dict(zip(range(20), range(20)))
        nx.draw_networkx_labels(G, nx.circular_layout(G), node_lab, font_size=16, font_color="white")
        ax.collections[0].set_edgecolor("#000000")
        plt.show()

        Layer = 0
        Graph_division = {}
        Graph_nodes = {}
        Sub_solution = {}

        Recursion(5)
        Reformulate_solution()
        QAOA_ans = Answer()
        QAOA_energy = Classical_solution(QAOA_ans)
        C_energy = Goemans_williamson_weighted()

        print(Graph_nodes)

        M = nx.Graph()
        M.add_weighted_edges_from(Graph_division[1]["F"])
        options = {'node_size': 1000, 'width': 1}
        nx.draw(M, nx.circular_layout(M), with_labels=False, **options)
        nx.draw_networkx_edges(M, nx.circular_layout(M), edgelist=M.edges(), width=1)
        edge_labels = nx.get_edge_attributes(M, "weight")
        nx.draw_networkx_edge_labels(M, nx.circular_layout(M), edge_labels, font_size=16)
        print(dict(zip(range(len(Graph_nodes[1]["F"])), Graph_nodes[1]["F"])))
        node_lab = dict(zip(Graph_nodes[1]["F"], Graph_nodes[1]["F"]))
        nx.draw_networkx_labels(M, nx.circular_layout(M), node_lab, font_size=16, font_color="white")

        ax = plt.gca()
        ax.margins(0.08)
        plt.axis("off")
        plt.show()

        print("-------------------------------------")
        print(f"N_nodes : {number_of_nodes}   seed : {seed}   max_size : {6}")
        print(f"GW : {C_energy}   QAOA : {QAOA_energy}")

        New_row = {"N_nodes": number_of_nodes, "Seed": seed, "Max_size": 6,
                   "Q_energy": QAOA_energy, "C_energy": C_energy}

        Data = Data.append(New_row, ignore_index=True)

    except Exception:

        print("-------------------------------------")
        print("ERROR")
Data.to_csv(f'Data/test_8/Size_test.csv', encoding='utf-8')

# 76











