import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
from Graph_generator import generate_graph


def split_graph_into_subgraphs(graph, max_size):
    """
    Разбивает граф на подграфы, размер которых не превышает max_size.
    Старается максимизировать связность внутри подграфов и избегать подграфов с одной вершиной.

    :param graph: Исходный граф (networkx.Graph)
    :param max_size: Максимальный размер подграфа
    :return: Два списка:
             1. Список подграфов в виде списков связей (рёбер).
             2. Список подграфов в виде списков вершин.
    """
    # Используем алгоритм обнаружения сообществ (например, greedy modularity)
    communities = list(community.greedy_modularity_communities(graph))

    subgraphs = []

    # Проходим по каждому сообществу и разбиваем его на подграфы, если оно слишком большое
    for comm in communities:
        comm_subgraph = graph.subgraph(comm)

        # Если размер сообщества меньше или равен max_size, добавляем его как есть
        if len(comm_subgraph) <= max_size:
            subgraphs.append(comm_subgraph)
        else:
            # Если сообщество слишком большое, разбиваем его на части
            # Используем алгоритм k-связных компонент или просто разбиваем на равные части
            # В данном примере разбиваем на равные части
            nodes = list(comm_subgraph.nodes())
            for i in range(0, len(nodes), max_size):
                subgraph_nodes = nodes[i:i + max_size]
                subgraph = graph.subgraph(subgraph_nodes)
                subgraphs.append(subgraph)

    # Убедимся, что подграфы не содержат изолированных вершин (если это возможно)
    # Если есть подграфы с одной вершиной, попробуем объединить их с соседними подграфами
    final_subgraphs = []
    single_node_subgraphs = []

    for subgraph in subgraphs:
        if len(subgraph) == 1:
            single_node_subgraphs.append(subgraph)
        else:
            final_subgraphs.append(subgraph)

    # Объединяем подграфы с одной вершиной с соседними подграфами
    for single_subgraph in single_node_subgraphs:
        node = list(single_subgraph.nodes())[0]
        neighbors = list(graph.neighbors(node))

        # Ищем соседний подграф, к которому можно добавить вершину
        merged = False
        for i, final_subgraph in enumerate(final_subgraphs):
            if any(neighbor in final_subgraph for neighbor in neighbors):
                # Добавляем вершину к этому подграфу, если это не нарушает max_size
                if len(final_subgraph) < max_size:
                    final_subgraphs[i] = graph.subgraph(list(final_subgraph.nodes()) + [node])
                    merged = True
                    break

        # Если не удалось объединить, добавляем как отдельный подграф
        if not merged:
            final_subgraphs.append(single_subgraph)

    # Преобразуем подграфы в списки связей и списки вершин
    subgraphs_edges = []
    subgraphs_nodes = []

    for subgraph in final_subgraphs:
        # Список связей (рёбер) подграфа
        edges = list(subgraph.edges())
        subgraphs_edges.append(edges)

        # Список вершин подграфа
        nodes = list(subgraph.nodes())
        subgraphs_nodes.append(nodes)

    return subgraphs_edges, subgraphs_nodes


def grapf_reduction(graph, max_size, visualise=False):
    """
    Итеративно сворачивает (редуцирует) граф.

    :param graph: Исходный граф (networkx.Graph)
    :param max_size: Максимальный размер подграфа
    :param visualise: Визуализация свёртки графа
    :return: Три словаря:
             1. reduction_layers - словарь содержащий все связи графа на каждой итерации.
                (keys = итерация, vals = связи графа)
             2. reduction_edges - словарь содержащий связи всех подграфов на каждой итерации.
                (keys1 = итерация, keys2 = вершина в которую сворачивается граф, vals = связи подграфа графа)
             3. reduction_edges - словарь содержащий вершины всех подграфов на каждой итерации.
                (keys1 = итерация, keys2 = вершина в которую сворачивается граф, vals = связи подграфа графа)
    """
    layer = 0
    reduction_edges = {}
    reduction_nodes = {}
    reduction_layers = {}
    copy_graph = graph.copy()

    while len(graph.nodes()) > 1:
        iterator = 1
        layer_edges = {}
        layer_nodes = {}
        reduction_layers[layer] = list(graph.edges())
        subgraphs_edges, subgraphs_nodes = split_graph_into_subgraphs(graph, max_size)

        node_name = max(list(graph.nodes()))

        for edges, nodes in zip(subgraphs_edges, subgraphs_nodes):

            graph.add_node(node_name + iterator)
            layer_edges[node_name + iterator] = edges
            layer_nodes[node_name + iterator] = nodes

            for node in nodes:
                neighbors = list(graph.neighbors(node))

                for k in neighbors:
                    if k not in nodes:
                        graph.remove_edge(node, k)
                        graph.add_edge(k, node_name + iterator)
                graph.remove_node(node)
            iterator += 1

        reduction_edges[layer] = layer_edges
        reduction_nodes[layer] = layer_nodes
        layer += 1


        if visualise:
            options = {'node_size': 1000, 'width': 1}
            nx.draw(graph, nx.circular_layout(graph), with_labels=True, **options)
            nx.draw_networkx_edges(graph, nx.circular_layout(graph), edgelist=graph.edges(), width=1)
            edge_labels = nx.get_edge_attributes(graph, "weight")
            nx.draw_networkx_edge_labels(graph, nx.circular_layout(graph), edge_labels)

            ax = plt.gca()
            ax.margins(0.08)
            plt.axis("off")
            plt.show()

    edge_weights = nx.get_edge_attributes(copy_graph, "weight")
    # Добавляем веса в исходный граф
    # (node_k, node_j, similar_sol, different_sol)

    reduction_layers[0] = list(map(lambda x: x + (edge_weights[x], 0,), reduction_layers[0]))

    # Добавляем веса в нулевой шаг свёртки
    # (node_k, node_j, similar_sol, different_sol)
    for node in list(reduction_edges[0].keys()):
        for i in range(len(reduction_edges[0][node])):
            k, j = reduction_edges[0][node][i]
            try:
                reduction_edges[0][node][i] += (edge_weights[(k, j)], 0)
            except KeyError:
                reduction_edges[0][node][i] += (edge_weights[(j, k)], 0)

    return reduction_layers, reduction_edges, reduction_nodes

# # Пример использования
# G = generate_graph(125, 0.7, visualise=False)  # Пример графа
# L, E, N = grapf_reduction(G, 5, visualise=False)