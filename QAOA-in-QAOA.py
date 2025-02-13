from Graph_generator import generate_graph
from QAOA import QAOA
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')



def QAOA2(graph, subgraphsize, depth=10, max_iter=10000, visualise=False, callback=False, logs=True):

    def Get_connected_subgraphs(max_size):

        global Layer, Graph_division, Graph_nodes

        connected_components = list(nx.connected_components(graph))
        subgraphs = []

        for component in connected_components:
            g = graph.subgraph(component)
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
        node_name = max(list(graph.nodes()))
        layer_edges = {}
        layer_nodes = {}

        for sg in subgraphs:

            H = nx.Graph()
            H.add_edges_from(sg)
            graph.add_node(node_name + iterator)
            layer_edges[node_name + iterator] = list(H.edges())
            layer_nodes[node_name + iterator] = list(H.nodes())

            for i in list(H.nodes()):
                neighbors = list(graph.neighbors(i))

                for k in neighbors:
                    if k not in list(H.nodes()):
                        graph.remove_edge(i, k)
                        graph.add_edge(k, node_name + iterator)
                graph.remove_node(i)
            iterator += 1

        Graph_division[Layer] = layer_edges
        Graph_nodes[Layer] = layer_nodes
        Layer += 1

    def Subgraph_solution(layer):

        global Q_energy, C_energy

        solution_dict = {}
        sub = Graph_division[layer]

        if layer == 0:
            for node in sub.keys():
                if logs:
                    print("\033[37m {}".format(f"------------------------------"))
                    print(f"Layer : {layer}    Node : {node}")
                H = nx.Graph()
                H.add_edges_from(sub[node])
                old_nodes = list(H.nodes())
                new_nodes = list(range(len(old_nodes)))
                mapping = dict(zip(old_nodes, new_nodes))
                H = nx.relabel_nodes(H, mapping)
                R, S, P, E, C_s, C_e, Pt_Q, Pt_C = QAOA(H, depth, max_iter=max_iter, logs=logs, callback=logs)
                solution_dict[node] = S[0]
                Q_energy += E
                C_energy -= C_e

        else:
            for node in sub.keys():
                if logs:
                    print("\033[37m {}".format(f"------------------------------"))
                    print(f"Layer : {layer}    Node : {node}")
                H = nx.Graph()
                H.add_weighted_edges_from(sub[node])
                old_nodes = list(H.nodes())
                new_nodes = list(range(len(old_nodes)))
                mapping = dict(zip(old_nodes, new_nodes))
                H = nx.relabel_nodes(H, mapping)
                R, S, P, E, C_s, C_e, Pt_Q, Pt_C = QAOA(H, depth, weighted=True, max_iter=max_iter, logs=logs, callback=logs)
                solution_dict[node] = S[0]
                Q_energy += E
                C_energy -= C_e

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
            if len(graph.nodes()) <= max_size:
                Graph_division[Layer] = {"F": list(graph.edges())}
                Graph_nodes[Layer] = {"F": list(graph.nodes())}
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

    def Classical_solution(state):
        cut = 0
        for k, j in A.edges():
            if state[k] != state[j]:
                cut += 1
        return cut


    try:

        A = graph.copy()
        A_conections = list(A.edges())

        global Graph_division, Graph_nodes, Sub_solution, Layer, Q_energy, C_energy
        Layer = 0
        Q_energy = 0
        C_energy = 0
        Graph_division = {}
        Graph_nodes = {}
        Sub_solution = {}

        Recursion(subgraphsize)
        Reformulate_solution()
        QAOA_ans = Answer()
        Classic_energy = Classical_solution(QAOA_ans)

        if visualise:

            M = nx.Graph()
            M.add_weighted_edges_from(Graph_division[1]["F"])
            options = {'node_size': 1000, 'width': 1}
            nx.draw(M, nx.circular_layout(M), with_labels=False, **options)
            nx.draw_networkx_edges(M, nx.circular_layout(M), edgelist=M.edges(), width=1)
            edge_labels = nx.get_edge_attributes(M, "weight")
            nx.draw_networkx_edge_labels(M, nx.circular_layout(M), edge_labels, font_size=16)
            # print(dict(zip(range(len(Graph_nodes[1]["F"])), Graph_nodes[1]["F"])))
            node_lab = dict(zip(Graph_nodes[1]["F"], Graph_nodes[1]["F"]))
            nx.draw_networkx_labels(M, nx.circular_layout(M), node_lab, font_size=16, font_color="white")

            ax = plt.gca()
            ax.margins(0.08)
            plt.axis("off")
            plt.show()

        if callback:

            print("\033[37m {}".format(f"------------------------------"))
            print(f"N_nodes : {20}   seed : random   max_size : {6}")
            print(f"Number of reductions : {len(list(Graph_division.keys()).copy()) - 1}")
            print(f"QAOA energy : {Q_energy}")
            print(f"Cumulative energy : {C_energy}")
            print(f"Classic energy ^ {Classic_energy}")

    except Exception:

        print("\033[31m {}".format("Something went wrong!!!"))
        print("The uncorrected graph")

G = generate_graph(20, 0.5, visualise=True)
QAOA2(G, 5, 10, callback=True, visualise=True, logs=True)











