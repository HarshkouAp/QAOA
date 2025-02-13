import networkx as nx
import matplotlib.pyplot as plt
from Graph_generator import generate_graph


def get_connected_subgraphs(max_size):
    global Layer
    connected_components = list(nx.connected_components(G))
    subgraphs = []

    for component in connected_components:
        g = G.subgraph(component)
        nodes = list(g.nodes)

        # Перебираем узлы, чтобы создать подграфы
        visited = set()
        for i in range(len(nodes)):
            if nodes[i] in visited:
                continue
            subgraph_nodes = []
            stack = [nodes[i]]

            # Используем DFS для получения связного подграфа
            while stack and len(subgraph_nodes) < max_size:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    subgraph_nodes.append(node)
                    for neighbor in g.neighbors(node):
                        if neighbor not in visited:
                            stack.append(neighbor)

            # Если подграф больше 1, добавляем его
            if len(subgraph_nodes) > 1:
                subgraph = g.subgraph(subgraph_nodes)
                subgraphs.append(list(subgraph.edges))

    Graph_division[Layer] = subgraphs
    Layer += 1

    iterator = 1
    node_name = max(list(G.nodes()))
    for sg in subgraphs:

        H = nx.Graph()
        H.add_edges_from(sg)
        G.add_node(node_name + iterator)
        new_noda_dict[str(node_name + iterator)] = list(H.nodes())

        for i in list(H.nodes()):
            neighbors = list(G.neighbors(i))

            for k in neighbors:
                if k not in list(H.nodes()):
                    G.remove_edge(i, k)
                    cuted_edges.append([i, k])
                    G.add_edge(k, node_name + iterator)
            G.remove_node(i)
        iterator += 1


G = generate_graph(125, 0.7, visualise=True)

options = {'node_size': 1000, 'width': 1, 'arrowsize': 10}

Layer = 0
cuted_edges = []
new_noda_dict = {}
Graph_division = {}

get_connected_subgraphs(5)
nx.draw(G, nx.circular_layout(G), with_labels=True, **options)
ax = plt.gca()
ax.collections[0].set_edgecolor("#000000")
plt.show()

get_connected_subgraphs(5)
Graph_division[Layer] = [list(G.edges)]
nx.draw(G, nx.circular_layout(G), with_labels=True, **options)
ax = plt.gca()
ax.collections[0].set_edgecolor("#000000")
plt.show()

print(len(Graph_division[0]))
print(len(Graph_division[1]))
print(len(Graph_division[2]))
