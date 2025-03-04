import networkx as nx

def graph_from_dataset(graph_ind):
    graph = nx.Graph()

    with open(f"Data/G_dataset/G{graph_ind}.txt", "r") as file:
        data = file.readlines()[1::]

        for line in data:
            edge = line.replace("\n", "").split()
            node_1 = int(edge[0])
            node_2 = int(edge[1])
            weight = float(edge[2])
            graph.add_edge(node_1, node_2, weight=weight)

    with open(f"Data/G_dataset/G{graph_ind}_opt_value.txt", "r") as file:
        data = file.readlines()[0].replace("\n", "").split()
        optimal_cut = int(data[0])


    return graph, optimal_cut

