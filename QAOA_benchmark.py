import numpy as np
import multiprocessing as mp
from networkx.algorithms.components import connected_components
from QAOA2 import *
import networkx as nx
import warnings
import pandas as pd
from GW_algoritm import *
warnings.filterwarnings('ignore')


def Experiment(seed, exist_data):

    if seed in exist_data:

        print("\033[37m {}".format(f"-------------{seed}-------------"))
        print("\033[32m {}".format("***Exist***"))

    else:
        graph = nx.erdos_renyi_graph(2000, 0.050025, seed)
        weights = dict(zip(graph.edges(), np.random.randint(0, 6, graph.number_of_edges())))
        nx.set_edge_attributes(graph, weights, "weight")
        q_cut, b_cut, c_cut = QAOA2(graph, 12, 10, max_iter=5000)
        print(f"Seed : {seed}   K-param : {1}")
        print(f"Number of nodes : 240   Number of edges : {1}")
        print(f"QAOA2 ::  Q energy : {q_cut}   BF-in-BF : {b_cut}")
        print(f"Classical energy for QAOA2 state: {c_cut}")
        f = open(f'Data/test_23/Comparison_{1}.csv', 'a')
        f.write(f'{seed},{q_cut}\n')
        f.close()


if __name__ == '__main__':

    exist_data = []
    try:
        with open(f'Data/test_23/Comparison_{1}.csv', "r") as file:
            for line in file.readlines()[1:]:
                try:
                    split_line = line.replace("\n", "").split(",")
                    seed = int(split_line[0])
                    exist_data.append(seed)
                except IndexError:
                    pass
    except FileNotFoundError:
        data = pd.DataFrame(columns=["seed", "QAOA_cut"])
        data.to_csv(f'Data/test_23/Comparison_{1}.csv', encoding='utf-8')

    tasks = [(seed, exist_data) for seed in range(1000, 1101, 1)]
    with mp.Pool(processes=7) as pool:
        pool.starmap(Experiment, tasks)