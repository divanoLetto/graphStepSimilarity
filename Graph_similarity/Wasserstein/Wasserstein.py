import math
import pickle
from pathlib import Path
import networkx as nx
from mathematic_utils.mathematic_utils import isnumeric, my_float
import numpy as np
import time
import methods.GromovWassersteinGraphToolkit as GwGt


def calculate_node_correctness(pairs, num_correspondence: int, g1, g2) -> float:
    node_correctness = 0
    for pair in pairs:
        id_i = pair[0]
        id_j = pair[1]
        node_i = g1.nodes()[id_i]
        node_j = g2.nodes()[id_j]
        num_neghbor_i = len([n for n in g1.neighbors(id_i)])
        num_neghbor_j = len([n for n in g2.neighbors(id_j)])
        if num_neghbor_i == num_neghbor_j:
            node_correctness += 1

    node_correctness /= num_correspondence
    return node_correctness


def calculate_node_errors(pairs, confidence,  g1, g2) -> float:
    v_confidence = np.array(confidence)
    normalized_confidence = v_confidence / np.sqrt(np.sum(v_confidence ** 2))
    nodes_error = 0
    for id_pair, pair in enumerate(pairs):
        current_error = 0
        id_i = pair[0]
        id_j = pair[1]
        node_i = g1.nodes()[id_i]
        node_j = g2.nodes()[id_j]
        num_neghbor_i = len([n for n in g1.neighbors(id_i)])
        num_neghbor_j = len([n for n in g2.neighbors(id_j)])
        if num_neghbor_i != num_neghbor_j:
            current_error += abs(num_neghbor_i - num_neghbor_j)
        for key_i, item_i in node_i.items():
            if key_i not in node_j.keys():
                current_error += 1
            else:
                item_j = node_j[key_i]
                if not isnumeric(item_i) or not isnumeric(item_j):
                    if item_i != item_j:  # todo better edit distance (?)
                        current_error += 1
                else:
                    mymax = max(my_float(item_i), my_float(item_j))
                    mymin = min(my_float(item_i), my_float(item_j))
                    den = abs(mymax) + abs(mymin)
                    if den != 0:
                        num = (-1) * (mymax - mymin) * (mymax - mymin)
                        diff = 1 - math.exp(num / den)
                        current_error += diff
        current_error *= normalized_confidence[id_pair]
        nodes_error += current_error
    num_nodes_1 = g1.number_of_nodes()
    num_nodes_2 = g2.number_of_nodes()
    nodes_error += abs(num_nodes_1 - num_nodes_2)  # todo o è meglio max(0, num_nodes_1 - num_nodes_2)??
    nodes_error /= num_nodes_1
    return nodes_error


def calc_wassertein_discrepancy(g_i, g_j, name_i, name_j, ot_dict, base_path, printt=False):
    adj_i = nx.adjacency_matrix(g_i)
    adj_j = nx.adjacency_matrix(g_j)
    nodes_degree_i = [d for n, d in g_i.degree()]
    nodes_degree_j = [d for n, d in g_j.degree()]
    p_i = np.ndarray(shape=(len(nodes_degree_i), 1))
    p_j = np.ndarray(shape=(len(nodes_degree_j), 1))
    num_nodes_1 = g_i.number_of_edges()
    num_nodes_2 = g_j.number_of_edges()
    for ik in range(len(nodes_degree_i)):
        p_i[ik, 0] = nodes_degree_i[ik] / num_nodes_1
    for ik in range(len(nodes_degree_j)):
        p_j[ik, 0] = nodes_degree_j[ik] / num_nodes_2
    idx2node_i = {o: node for o, node in enumerate(g_i.nodes())}
    idx2node_j = {o: node for o, node in enumerate(g_j.nodes())}
    save_name = name_i + "_" + name_j + ".pkl"
    save_path = base_path + save_name
    save_vect = Path(save_path)
    if not save_vect.exists():
        time_s = time.time()
        pairs_idx, pairs_name, pairs_confidence = GwGt.direct_graph_matching(
            0.5 * (adj_i + adj_i.T), 0.5 * (adj_j + adj_j.T), p_i, p_j, idx2node_i, idx2node_j, ot_dict)
        run_time = time.time() - time_s
        if printt:
            print("Gwl of " + name_i + " and " + name_j + " took duration {:.4f}s.".format(run_time))
        with open(save_path, 'wb') as f:
            pickle.dump([pairs_idx, pairs_name, pairs_confidence, run_time], f)
    else:
        with open(save_path, "rb") as input_file:
            pairs_idx, pairs_name, pairs_confidence, run_time = pickle.load(input_file)
    return pairs_idx, pairs_name, pairs_confidence, run_time


def calc_recursive_wassertein_discrepancy(g_i, g_j, name_i, name_j, ot_dict, base_path, printt=False):
    adj_i = nx.adjacency_matrix(g_i)
    adj_j = nx.adjacency_matrix(g_j)
    nodes_degree_i = [d for n, d in g_i.degree()]
    nodes_degree_j = [d for n, d in g_j.degree()]
    p_i = np.ndarray(shape=(len(nodes_degree_i), 1))
    p_j = np.ndarray(shape=(len(nodes_degree_j), 1))
    num_nodes_1 = g_i.number_of_edges()
    num_nodes_2 = g_j.number_of_edges()
    for ik in range(len(nodes_degree_i)):
        p_i[ik, 0] = nodes_degree_i[ik] / num_nodes_1
    for ik in range(len(nodes_degree_j)):
        p_j[ik, 0] = nodes_degree_j[ik] / num_nodes_2
    idx2node_i = {o: node for o, node in enumerate(g_i.nodes())}
    idx2node_j = {o: node for o, node in enumerate(g_j.nodes())}
    save_name = name_i + "_" + name_j + ".pkl"
    save_path = base_path + save_name
    save_vect = Path(save_path)
    if not save_vect.exists():
        time_s = time.time()
        pairs_idx, pairs_name, pairs_confidence = GwGt.direct_graph_matching(
            0.5 * (adj_i + adj_i.T), 0.5 * (adj_j + adj_j.T), p_i, p_j, idx2node_i, idx2node_j, ot_dict)

        pairs_idx, pairs_name, pairs_confidence = GwGt.recursive_direct_graph_matching(
            0.5 * (adj_i + adj_i.T), 0.5 * (adj_j + adj_j.T), p_i, p_j, idx2node_i, idx2node_j, ot_dict,
            weights=None, predefine_barycenter=False, cluster_num=2,
            partition_level=10, max_node_num=0)
        run_time = time.time() - time_s

        if printt:
            print("S-gwl of " + name_i + " and " + name_j + " took duration {:.4f}s.".format(run_time))
        with open(save_path, 'wb') as f:
            pickle.dump([pairs_idx, pairs_name, pairs_confidence, run_time], f)
    else:
        with open(save_path, "rb") as input_file:
            pairs_idx, pairs_name, pairs_confidence, run_time = pickle.load(input_file)
    return pairs_idx, pairs_name, pairs_confidence, run_time