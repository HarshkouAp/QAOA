import networkx as nx

def graph_from_dataset(graph_ind):
    graph = nx.Graph()

    with open(f"Data/G_dataset/G{graph_ind}.txt", "r") as file:
        data = file.readlines()[1::]
        for line in data:
            edge = line.replace("\n", "").split()
            node_1 = edge[0]
            node_2 = edge[1]
            weight = edge[1]
            graph.add_edge(node_1, node_2, weight=weight)

    return graph

