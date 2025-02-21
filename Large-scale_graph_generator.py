import networkx as nx
from Graph_generator import generate_graph
import matplotlib.pyplot as plt
import random

def LSGN():

    n = 6
    d = 0.7

    G = generate_graph(n, d)
    H = generate_graph(n, d)

    old_nodes = list(H.nodes())
    new_nodes = list(range(n, n + len(old_nodes)))
    mapping = dict(zip(old_nodes, new_nodes))
    H = nx.relabel_nodes(H, mapping)

    F = nx.compose(G,H)

    G_nodes = {}
    H_nodes = {}
    Cut = 0

    for node in G.nodes():
        G_nodes[node] = len(list(G.neighbors(node))) + 1

    for node in H.nodes():
        H_nodes[node] = len(list(H.neighbors(node))) + 1

    for node in G_nodes.keys():
        mirror_nodes = list(H.nodes())
        random.shuffle(mirror_nodes)
        for i in range(G_nodes[node]):
            G_nodes[node] -= 1
            H_nodes[mirror_nodes[i]] -= 1
            F.add_edge(node, mirror_nodes[i])
            Cut += 1

    for node in H_nodes.keys():
        mirror_nodes = list(G.nodes())
        random.shuffle(mirror_nodes)
        if H_nodes[node] > 0:
            for i in range(H_nodes[node]):
                H_nodes[node] -= 1
                G_nodes[mirror_nodes[i]] -= 1
                F.add_edge(node, mirror_nodes[i])
                Cut += 1

    return F, Cut




