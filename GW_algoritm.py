import networkx as nx
import cvxpy as cp
import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple


def goemans_williamson(graph: nx.Graph, solver: str = "SCS", use_sparse: bool = True) -> Tuple[float, np.ndarray]:

    if not nx.is_weighted(graph):
        raise ValueError("Граф должен быть взвешенным.")

    n = graph.number_of_nodes()
    nodes = list(graph.nodes)

    if use_sparse:
        weight_matrix = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight="weight", format="csr")
    else:
        weight_matrix = nx.to_numpy_array(graph, nodelist=nodes, weight="weight")

    # Создаем переменную для полуопределенной матрицы X
    X = cp.Variable((n, n), symmetric=True)

    # Ограничения: X положительно полуопределена, диагональные элементы равны 1
    constraints = [X >> 0, cp.diag(X) == 1]

    # Целевая функция: максимизация суммы (w_ij * (1 - X[i, j])) / 4
    if use_sparse:
        objective = cp.sum(cp.multiply(weight_matrix, 1 - X)) / 4
    else:
        objective = cp.sum(cp.multiply(weight_matrix, 1 - X)) / 4

    # Формулируем и решаем задачу SDP
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver=solver, verbose=False)

    # Проверяем, что решение найдено
    if X.value is None:
        raise RuntimeError("Решение SDP не найдено.")

    # Извлекаем решение X
    X_value = X.value

    # Генерируем случайный вектор для округления
    w = np.random.randn(n)

    # Округляем решение: cut[i] = sign(<X_i, w>), где X_i — i-я строка матрицы X
    cut = np.sign(X_value @ w)

    # Вычисляем величину разреза
    if use_sparse:
        cut_value = weight_matrix.multiply(1 - np.outer(cut, cut)).sum() / 4
    else:
        cut_value = np.sum(weight_matrix * (1 - np.outer(cut, cut))) / 4

    return cut_value, cut


number_of_nodes = 100
density = 0.5
number_of_edges = number_of_nodes * (number_of_nodes - 1) * density / 2
graph = nx.dense_gnm_random_graph(number_of_nodes, int(number_of_edges))
weights = dict(zip(graph.edges(), np.ones(graph.number_of_edges())))
nx.set_edge_attributes(graph, weights, "weight")


test = []
for _ in range(500):
    cut, state = goemans_williamson(graph)
    print(cut)
    test.append(cut)


test = np.array(test)
print(np.std(test))
