import numpy as np
import multiprocessing as mp
import warnings
import pandas as pd
from QAOA import QAOA
from Graph_generator import generate_graph
warnings.filterwarnings('ignore')


def Experiment(seed, number_of_nodes, density, depth, exist_data):
    global data
    if (depth in exist_data.keys()) and (seed in exist_data[depth]):

        print("\033[37m {}".format(f"-------------{seed}-------------"))
        print("\033[32m {}".format("***Exist***"))

    else:
        try:

            graph = generate_graph(number_of_nodes, density, seed=seed, weighted=False, visualise=False)
            r, s, p, e, c_s, c_e, pt_q, pt_c = QAOA(graph, depth, logs=False, callback=True, )

            f = open(f'Data/test_13/QAOA_test_{number_of_nodes}.csv', 'a')
            f.write(f'{number_of_nodes},{density},{seed},{depth},{r},{s[0]},{p},{e},{c_s},{c_e},{pt_q},{pt_c}\n')
            f.close()

        except Exception:

            print("\033[37m {}".format(f"-------------{seed}-------------"))
            print("ERROR")


if __name__ == '__main__':

    for nodes in range(3, 9, 1):
        exist_data = {}
        try:
            with open(f'Data/test_13/QAOA_test_{nodes}.csv', "r") as file:
                for line in file.readlines()[1:]:
                    try:
                        split_line = line.replace("\n", "").split(",")
                        p_param = int(split_line[3])

                        if p_param in exist_data.keys():
                            exist_data[p_param].append(int(split_line[2]))
                        else:
                            exist_data[p_param] = [int(split_line[2])]
                    except IndexError:
                        pass
        except FileNotFoundError:
            data = pd.DataFrame(columns=["Nodes", "Density", "Seed", "P_param", "Result", "QAOA_sol", "Probability", "QAOA_energy",
                                         "C_solution", "C_energy", "QAOA_time", "C_time"])
            data.to_csv(f'Data/test_13/QAOA_test_{nodes}.csv', encoding='utf-8')

        for depth in range(1, 20, 1):
            tasks = [(seed, nodes, 0.7, depth, exist_data) for seed in range(1000, 1105, 1)]
            with mp.Pool(processes=7) as pool:
                pool.starmap(Experiment, tasks)


