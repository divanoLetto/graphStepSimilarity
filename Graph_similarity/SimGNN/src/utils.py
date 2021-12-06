"""Data processing utilities."""

import json
import math
from texttable import Texttable

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def process_pair(path):
    """
    Reading a json file with a pair of graphs.
    :param path: Path to a JSON file.
    :return data: Dictionary with data.
    """
    data = json.load(open(path))
    return data


def process_pair_graphh(graph_pair):
    data = {}
    g_1_nodes = []
    g_2_nodes = []
    id1_num_node = {}
    id2_num_node = {}
    for i, node in enumerate(graph_pair[0].full_graph):
        g_1_nodes.append(graph_pair[0].full_graph.nodes[node]["type"])
        id1_num_node[node] = i
    for i, node in enumerate(graph_pair[1].full_graph):
        g_2_nodes.append(graph_pair[1].full_graph.nodes[node]["type"])
        id2_num_node[node] = i
    data["labels_1"] = g_1_nodes
    data["labels_2"] = g_2_nodes
    g_1_edges = [[id1_num_node[e[0]], id1_num_node[e[1]]] for e in graph_pair[0].full_graph.edges]
    g_2_edges = [[id2_num_node[e[0]], id2_num_node[e[1]]] for e in graph_pair[1].full_graph.edges]
    data["graph_1"] = g_1_edges
    data["graph_2"] = g_2_edges
    data["ged"] = graph_pair[2]
    return data


def calculate_loss(prediction, target):
    """
    Calculating the squared loss on the normalized GED.
    :param prediction: Predicted log value of GED.
    :param target: Factual log transofmed GED.
    :return score: Squared error.
    """
    prediction = -math.log(prediction)
    target = -math.log(target)
    score = (prediction-target)**2
    return score


def calculate_normalized_ged(data):
    """
    Calculating the normalized GED for a pair of graphs.
    :param data: Data table.
    :return norm_ged: Normalized GED score.
    """
    norm_ged = data["ged"]  # /(0.5*(len(data["labels_1"])+len(data["labels_2"])))
    return norm_ged


def reverse_normalized_ged(value):
    value = value
    return value
