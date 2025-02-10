import networkx as nx
import matplotlib.pyplot as plt

def generate_graph(number_of_nodes, density, seed='random', visualise=False):
    number_of_edges = number_of_nodes * (number_of_nodes - 1) * density / 2
    real_density = round(2 * int(number_of_edges) / (number_of_nodes * (number_of_nodes - 1)), 4)

    if seed == 'random':
        G = nx.dense_gnm_random_graph(number_of_nodes, int(number_of_edges))
    else:
        G = nx.dense_gnm_random_graph(number_of_nodes, int(number_of_edges), seed)

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

    return G