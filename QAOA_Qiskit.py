import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from scipy.optimize import minimize
from qiskit import *
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.visualization import plot_histogram
from qiskit.visualization import plot_state_city
from qiskit.circuit.library import *


def Graph_creator():
    graph = nx.Graph()
    graph.add_edges_from([[0, 3], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4]])
    return graph


def Graph_drawer(graph):
    plt.figure(figsize=(8, 8))
    options = {'node_color': "cadetblue",
               'node_size': 7500,
               'font_size': 20,
               'width': 2}
    pos = nx.bipartite_layout(graph, [0, 1, 2])
    nx.draw(graph, pos, with_labels=1, **options)
    plt.show()
    return graph


def Circuit_drawer(qc):
    backend = BasicSimulator()
    qc.draw(output="mpl", initial_state=True)
    result = backend.run(qc, shots=1000).result()
    counts = result.get_counts()
    plot_histogram(counts)
    plt.show()


def ZZ_operator(qc, q1, q2, gamma):
    qc.cx(q1, q2)
    qc.rz(2*gamma, q2)
    qc.cx(q1, q2)


def Cost_operator_circuit(qc, graph, gamma):
    for k, j in nx.edges(graph):
        ZZ_operator(qc, k, j, gamma)


def Mixer_operator_circuit(qc, graph, beta):
    for k in nx.nodes(graph):
        qc.rx(2 * beta, k)


def QAOA_circuit(graph, gamma, beta):
    p = len(beta)
    N = nx.number_of_nodes(graph)
    qc = QuantumCircuit(N, N)
    qc.h(range(N))
    for k in range(p):
        qc.barrier()
        Cost_operator_circuit(qc, graph, gamma[k])
        qc.barrier()
        Mixer_operator_circuit(qc, graph, beta[k])
    qc.barrier()
    qc.measure(range(N), range(N-1, -1, -1))
    return qc


def MaxCut_clasical_solver(state, graph):
    cut = 0
    for k, j in graph.edges():
        if state[k] != state[j]:
            cut -= 1
    return cut


def MaxCut_energy(qc, graph):
    energy = 0
    total_counts = 0
    backend = BasicSimulator()
    result = backend.run(qc, shots=1000).result()
    counts = result.get_counts()
    for state, state_counts in counts.items():
        state_val = MaxCut_clasical_solver(state, graph)
        energy += state_val * state_counts
        total_counts += state_counts
    return energy / total_counts


def Black_box_function(graph, p_param):
    def function(ang_arr):
        gamma = ang_arr[0:p_param]
        beta = ang_arr[p_param:2*p_param]
        QC = QAOA_circuit(G, gamma, beta)
        Energy = MaxCut_energy(QC, G)
        return Energy
    return function


G = Graph_creator()
Graph_drawer(G)
P_param = 4
Init_params = np.array(np.random.random_sample(2 * P_param))
Black_box = Black_box_function(G, P_param)
Result_min = minimize(Black_box, Init_params, method='COBYLA', options={'maxiter': 10000, 'disp': True})
Optimal_angles = Result_min['x']

print(Result_min)
print("________________________________________________")
print(f'Optimal gamma: {Optimal_angles[0:P_param]} \nOptimal beta:  {Optimal_angles[P_param:2*P_param]}')
print("________________________________________________")
QC = QAOA_circuit(G, Optimal_angles[0:P_param], Optimal_angles[P_param:2*P_param])
Circuit_drawer(QC)

