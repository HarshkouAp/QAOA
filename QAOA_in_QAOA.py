import time
from QAOA import *
from Graph_cutter import *
from Graph_from_dataset import *
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


def QAOA2(graph, max_size, depth=10, max_iter=10000, visualise=False, callback=False, logs=False):

    def recursion():

        for layer in reduction_edges.keys():

            # Находим решения для подграфов
            subgraph_solution(layer)

            if layer != 0:
                energy(layer - 1)

            if layer != max(reduction_edges.keys()):
                # Раcчитываем веса связей
                weights(layer)

        reformulate_solution()
        s = merge_solution()

        ind_1 = max(list(q_energy_dict.keys()))
        ind_2 = list(q_energy_dict[ind_1].keys())[0]
        q_e = q_energy_dict[ind_1][ind_2]
        c_e = c_energy_dict[ind_1][ind_2]

        return s, q_e, c_e

    def subgraph_solution(layer):
        solution = {}
        q_energy = {}
        c_energy = {}

        for node in list(reduction_nodes[layer].keys()):

            # Проверка на тривиальный подграф (одна вершина). Ф-ия QAOA не работает с таковыми.
            if len(reduction_nodes[layer][node]) != 1:

                if logs:
                    print("\033[37m {}".format(f"------------------------------"))
                    print(f"Layer : {layer}    Node : {node}")

                subgraph = nx.Graph()
                subgraph.add_nodes_from(reduction_nodes[layer][node])
                for k, j, s, d in reduction_edges[layer][node]:
                    subgraph.add_edge(k, j, weight=(s - d))

                # Для работы ф-ии QAOA необходимо чтобы вершины графа начинали нумероваться с 1
                old_nodes = list(subgraph.nodes())
                new_nodes = list(range(len(old_nodes)))
                mapping = dict(zip(old_nodes, new_nodes))
                subgraph = nx.relabel_nodes(subgraph, mapping)

                r, s, p, e, c_s, c_e, pt_q, pt_c = QAOA(subgraph, depth, weighted=True, max_iter=max_iter, logs=logs, callback=logs)
                solution[node] = s[0]
                q_energy[node] = e * (-1)
                c_energy[node] = c_e * (-1)

            else:
                solution[node] = "1"
                q_energy[node] = 0
                c_energy[node] = 0

        solution_dict[layer] = solution
        q_energy_dict[layer] = q_energy
        c_energy_dict[layer] = c_energy

    def weights(layer):

        edges = reduction_edges[layer + 1]
        nodes = reduction_nodes[layer]
        solutions = solution_dict[layer]
        connections = reduction_layers[layer]

        # Сначала добавим веса к reduction_edges[layer + 1]

        # перебираем все не взвешенные подграфы на данном шаге разбиения
        for subgraph in edges.keys():
            # перебираем все не взвешенные связи между вершинами k и j в конкретном подграфе
            for k, j in edges[subgraph]:
                # Создаём два списка с вершинами которые входят в состав вершин k и j
                k_nodes = nodes[k]
                j_nodes = nodes[j]
                # Создаём два списка с решениями для k, j
                k_sol = solutions[k]
                j_sol = solutions[j]

                similar = 0
                different = 0
                # Проверяем наличие связи между вершинами x из k_nodes и y из j_nodes
                for x, y, s, d in connections:

                    if x in k_nodes and y in j_nodes:
                        x_bit = k_sol[k_nodes.index(x)]
                        y_bit = j_sol[j_nodes.index(y)]
                        if  x_bit == y_bit:
                            # Если x и y в одном классе увеличиваем вес на значение веса связи
                            similar += s
                            different += d
                        else:
                            # Если x и y в разных классах, то уменьшаем вес на значение веса связи
                            similar += d
                            different += s

                    elif y in k_nodes and x in j_nodes:
                        x_bit = k_sol[k_nodes.index(y)]
                        y_bit = j_sol[j_nodes.index(x)]
                        if  x_bit == y_bit:
                            # Если x и y в одном классе увеличиваем вес на значение веса связи
                            similar += s
                            different += d
                        else:
                            # Если x и y в разных классах, то уменьшаем вес на значение веса связи
                            similar += d
                            different += s


                # Добавляем рассчитанный вес к ребру
                reduction_edges[layer + 1][subgraph][edges[subgraph].index((k, j))] = (k, j, similar, different)

        edges = reduction_layers[layer + 1]

        for k, j in edges:
            # Создаём два списка с вершинами которые входят в состав вершин k и j
            k_nodes = nodes[k]
            j_nodes = nodes[j]
            # Создаём два списка с решениями для k, j
            k_sol = solutions[k]
            j_sol = solutions[j]

            similar = 0
            different = 0
            # Проверяем наличие связи между вершинами x из k_nodes и y из j_nodes
            for x, y, s, d in connections:

                if x in k_nodes and y in j_nodes:
                    x_bit = k_sol[k_nodes.index(x)]
                    y_bit = j_sol[j_nodes.index(y)]
                    if x_bit == y_bit:
                        # Если x и y в одном классе увеличиваем вес на значение веса связи
                        similar += s
                        different += d
                    else:
                        # Если x и y в разных классах, то уменьшаем вес на значение веса связи
                        similar += d
                        different += s
                elif y in k_nodes and x in j_nodes:
                    x_bit = k_sol[k_nodes.index(y)]
                    y_bit = j_sol[j_nodes.index(x)]
                    if x_bit == y_bit:
                        # Если x и y в одном классе увеличиваем вес на значение веса связи
                        similar += s
                        different += d
                    else:
                        # Если x и y в разных классах, то уменьшаем вес на значение веса связи
                        similar += d
                        different += s


            # Добавляем рассчитанный вес к ребру
            reduction_layers[layer + 1][edges.index((k, j))] = (k, j, similar, different)

    def energy(layer):

        pre_sol = solution_dict[layer]
        sols = solution_dict[layer + 1]

        pre_q_e = q_energy_dict[layer]
        pre_c_e = c_energy_dict[layer]

        connections = reduction_layers[layer]
        edges = reduction_edges[layer + 1]
        pre_nodes = reduction_nodes[layer]
        nodes = reduction_nodes[layer + 1]

        q_e_dict = {}
        c_e_dict = {}

        # перебираем все подграфы на данном шаге разбиения
        for subgraph in edges.keys():

            q_e_dict[subgraph] = 0
            c_e_dict[subgraph] = 0
            sub_nodes = nodes[subgraph]

            # Рассчитываем сумму энергий подграфов с предыдущего слоя sum(C(x_i))
            for node in sub_nodes:
                q_e_dict[subgraph] += pre_q_e[node]
                c_e_dict[subgraph] += pre_c_e[node]

            # Рассчитаем величину w_ij = sum(x_iu * x_jv * W_iu_jv)
            sub_sol = sols[subgraph]

            # Перебираем все существующие связи между подграфами
            for k, j, s, d in edges[subgraph]:

                k_bit = sub_sol[sub_nodes.index(k)]
                j_bit = sub_sol[sub_nodes.index(j)]

                k_nodes = pre_nodes[k]
                j_nodes = pre_nodes[j]
                k_sol = pre_sol[k]
                j_sol = pre_sol[j]

                for x, y, pre_s, pre_d in connections:

                    if x in k_nodes and y in j_nodes:
                        x_bit = k_sol[k_nodes.index(x)]
                        y_bit = j_sol[j_nodes.index(y)]
                        if k_bit == j_bit:
                            # проверяем какой бит решения соответствует x и y
                            if x_bit == y_bit:
                                # Если x и y в одном классе увеличиваем вес на значение веса связи
                                q_e_dict[subgraph] += pre_d
                                c_e_dict[subgraph] += pre_d
                            else:
                                q_e_dict[subgraph] += pre_s
                                c_e_dict[subgraph] += pre_s

                        else:
                            if x_bit == y_bit:
                                # Если x и y в одном классе увеличиваем вес на значение веса связи
                                q_e_dict[subgraph] += pre_s
                                c_e_dict[subgraph] += pre_s
                            else:
                                q_e_dict[subgraph] += pre_d
                                c_e_dict[subgraph] += pre_d

                    elif y in k_nodes and x in j_nodes:
                        x_bit = k_sol[k_nodes.index(y)]
                        y_bit = j_sol[j_nodes.index(x)]
                        if k_bit == j_bit:
                            # проверяем какой бит решения соответствует x и y
                            if x_bit == y_bit:
                                # Если x и y в одном классе увеличиваем вес на значение веса связи
                                q_e_dict[subgraph] += pre_d
                                c_e_dict[subgraph] += pre_d
                            else:
                                q_e_dict[subgraph] += pre_s
                                c_e_dict[subgraph] += pre_s

                        else:
                            if x_bit == y_bit:
                                # Если x и y в одном классе увеличиваем вес на значение веса связи
                                q_e_dict[subgraph] += pre_s
                                c_e_dict[subgraph] += pre_s
                            else:
                                q_e_dict[subgraph] += pre_d
                                c_e_dict[subgraph] += pre_d


        q_energy_dict[layer + 1] = q_e_dict
        c_energy_dict[layer + 1] = c_e_dict

    def reformulate_solution():
        layers = list(reduction_layers.keys())[:0:-1]
        for l in layers:
            solutions = solution_dict[l]
            nodes = reduction_nodes[l]
            for node in nodes.keys():
                sol = solutions[node]
                sub_nodes = nodes[node]

                for i in range(len(sol)):
                    if sol[i] == "1":
                        solution_dict[l - 1][sub_nodes[i]] = invert(solution_dict[l - 1][sub_nodes[i]])

    def invert(solution):
        inv_solution = ""
        for i in solution:
            if i == "1":
                inv_solution += "0"
            else:
                inv_solution += "1"

        return inv_solution

    def merge_solution():
        sol_dict = {}
        solution = ""
        nodes = solution_dict[0].keys()
        for node in nodes:
            sub_nodes = reduction_nodes[0][node]
            sub_sol = solution_dict[0][node]
            for i in sub_nodes:
                sol_dict[i] = sub_sol[sub_nodes.index(i)]

        for i in range(min(list(sol_dict.keys())), max(list(sol_dict.keys())) + 1):
            solution += sol_dict[i]

        return solution

    def classical_solution(state):
        cut = 0
        for k, j, s, d in reduction_layers[0]:
            if state[k - 1] != state[j - 1]:
                cut += s
        return cut

    N_nodes = nx.number_of_nodes(graph)
    N_edges = nx.number_of_edges(graph)


    # Строим разбиение и свёртку графа
    s = time.time()
    global reduction_layers, reduction_edges, reduction_nodes
    reduction_layers, reduction_edges, reduction_nodes = grapf_reduction(graph, max_size, visualise=False)
    gp_t = time.time() - s

    # Создаём словари для сбора промежуточных результатов
    global solution_dict, q_energy_dict, c_energy_dict
    solution_dict = {}
    q_energy_dict = {}
    c_energy_dict = {}

    start = time.time()
    solution, q_energy, c_energy = recursion()
    r_t = time.time() - start

    c_s = classical_solution(solution)

    if callback:

        print("\033[33m {}".format(f"====================================="))
        print(f"Subgraph size : {max_size}   Depth : {depth}   Max iter : {max_iter}")
        print(f"Number of nodes : {N_nodes}   Number of edges : {N_edges}")
        print(f"QAOA2  ::  Q_energy : {round(q_energy, 6)}   C_energy : {c_energy}")
        print(f"Classical energy for QAOA2 state: {c_s}")
        print(f"Graph partition time : {round(gp_t, 9)}   Recursion time : {round(r_t, 9)}")
        print("\033[33m {}".format(f"====================================="))

    return solution, q_energy, c_energy, c_s


# G, optimal_cut = graph_from_dataset(60)
#
# Sol, Q_e, C_e, C_s = QAOA2(G, 10, 5, callback=True, visualise=False, logs=True)
# print(optimal_cut)