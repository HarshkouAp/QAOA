import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def generate_graph(number_of_nodes, density, weighted=False, seed='random', visualise=False):

    if seed == 'random':
        seed = round(1000 * np.random.random())

    if weighted == False:

        number_of_edges = number_of_nodes * (number_of_nodes - 1) * density / 2
        real_density = round(2 * int(number_of_edges) / (number_of_nodes * (number_of_nodes - 1)), 4)
        G = nx.dense_gnm_random_graph(number_of_nodes, int(number_of_edges), seed)

        for k, j in G.edges():
            nx.set_edge_attributes(G, {(k, j): {"weight": 1}})

        if visualise:

            plt.figure(figsize=(8, 8))
            pos = nx.circular_layout(G)

            # nx.draw_kamada_kawai(G, with_labels=True)
            # plt.show()
            # nx.draw_planar(G, with_labels=True)
            # plt.show()
            # nx.draw_random(G, with_labels=True)
            # plt.show()
            # nx.draw_spectral(G, with_labels=True)
            # plt.show()
            # nx.draw_spring(G, with_labels=True)
            # plt.show()
            # nx.draw_shell(G, with_labels=True)
            # plt.show()

            options = {'node_size': 1000,
                       'width': 1,
                       'arrowsize': 10}

            nx.draw(G, pos, with_labels=1, arrows=True, **options)
            ax = plt.gca()
            ax.collections[0].set_edgecolor("#000000")
            plt.show()

    if weighted == True:

        number_of_edges = number_of_nodes * (number_of_nodes - 1) * density / 2
        real_density = round(2 * int(number_of_edges) / (number_of_nodes * (number_of_nodes - 1)), 4)
        G = nx.dense_gnm_random_graph(number_of_nodes, int(number_of_edges))
        for k, j in G.edges():
            nx.set_edge_attributes(G, {(k, j): {"weight": round(2 * (np.random.random() - 0.5), 3)}})

        if visualise:

            options = {'node_size': 1000, 'width': 1}
            nx.draw(G, nx.circular_layout(G), with_labels=True, **options)
            nx.draw_networkx_edges(G, nx.circular_layout(G), edgelist=G.edges(), width=1)
            edge_labels = nx.get_edge_attributes(G, "weight")
            nx.draw_networkx_edge_labels(G, nx.circular_layout(G), edge_labels)

            ax = plt.gca()
            ax.margins(0.08)
            plt.axis("off")
            plt.show()

    return G


# Gr = generate_graph(10, 0.6, visualise=True)
# print(nx.get_edge_attributes(Gr, "weight"))

