import time
from pathlib import Path
from Nodes.Node import FlatNode
from Nodes.Node_utils import get_nodes_from_datas, get_all_neighbor_nodes
from Parser.Parser import parse_file
from Parser.parser_utils import check_just_conteiner
from utils import raplace_nodes
import networkx as nx
import os


def all_nodes_to_graph(G, all_nodes, name, graph_saves_paths):
    start_time = time.time()
    for flat_node in all_nodes:
        # G.add_node(flat_node)
        dict_protery = FlatNode.get_dict_parameters(flat_node)
        G.add_nodes_from([(flat_node.id, dict_protery)])
    for flat_node in all_nodes:
        numeric_paramenters = []
        for par in flat_node.parameters:
            if isinstance(par, FlatNode):
                G.add_edge(flat_node.id, par.id)
            else:
                numeric_paramenters.append(par)
        flat_node.parameters = numeric_paramenters

    print("   Graphh of " + name + " model realizing time: %s seconds" % (time.time() - start_time))
    nx.write_graphml(G, graph_saves_paths)


def make_graph_simplex(file_name):
    name = os.path.splitext(file_name)[0]
    graph_saves_paths = 'C:/Users/Computer/PycharmProjects/graphStepSimilarity/graph_save/simplex/' + name + '_simplex.graphml'
    graph_path = Path(graph_saves_paths)

    if graph_path.exists():
        G_simplex = nx.read_graphml(graph_saves_paths)
        return G_simplex
    else:
        headers, datas = parse_file(file_name)
        print("file " + name + " parsed")
        all_flat_nodes = get_nodes_from_datas(datas)
        print("   All nodes obtained")
        # replace identifier with actual nodes
        raplace_nodes(all_flat_nodes)
        print("   All edges obtained")

        # TODO se ci sono due nodi uguali sono da considerarsi lo stesso? sopratutto nei composed
        # TODO i parametri strnga_vuota si eliminano?
        # TODO Il grafo è diretto o indiretto?

        G_simplex = nx.Graph()
        all_nodes_to_graph(G_simplex, all_flat_nodes, name, graph_saves_paths)

        return G_simplex


def make_graph_simplex_direct(file_name):
    name = os.path.splitext(file_name)[0]
    graph_saves_paths = 'C:/Users/Computer/PycharmProjects/graphStepSimilarity/graph_save/simplex_direct/' + name + '.graphml'
    graph_path = Path(graph_saves_paths)

    if graph_path.exists():
        G_simplex_d = nx.read_graphml(graph_saves_paths)
        return G_simplex_d
    else:
        headers, datas = parse_file(file_name)
        print("file " + file_name + " parsed")
        all_flat_nodes = get_nodes_from_datas(datas)
        print("   All nodes obtained")
        raplace_nodes(all_flat_nodes)
        print("   All edges obtained")
        G_simplex_d = nx.DiGraph()
        all_nodes_to_graph(G_simplex_d, all_flat_nodes, name, graph_saves_paths)

        return G_simplex_d


def make_graph_scomposed_direct(file_name):
    name = os.path.splitext(file_name)[0]
    graph_saves_paths = 'C:/Users/Computer/PycharmProjects/graphStepSimilarity/graph_save/scomposed_direct/' + name + '.graphml'
    graph_path = Path(graph_saves_paths)

    if graph_path.exists():
        G_simplex_d = nx.read_graphml(graph_saves_paths)
        return G_simplex_d
    else:
        headers, datas = parse_file(file_name)
        print("file " + file_name + " parsed")
        all_nodes, all_flat_nodes = get_nodes_from_datas(datas)
        print("   All nodes obtained")
        raplace_nodes(all_flat_nodes)
        print("   All edges obtained")
        G_simplex_d = nx.DiGraph()
        all_nodes_to_graph(G_simplex_d, all_flat_nodes, name, graph_saves_paths)

        return G_simplex_d


def spit_graph_in_parts(graph):
    partitions = {}
    shape_rep_found = False
    count = 0
    count_occ = 0
    for node in graph:
        if graph.nodes[node]["type"] == "SHAPE_REPRESENTATION":
            shape_rep_found = True
            name = graph.nodes[node]["SHAPE_REPRESENTATION_0"]
            list_of_nodes = [node]
            last_layer_list = [node]
            while len(last_layer_list) != 0:
                tmp_list = []
                for n in last_layer_list:
                    inner_neighbor = graph.predecessors(n)
                    tmp_list.extend(inner_neighbor)
                last_layer_list = tmp_list
                list_of_nodes.extend(last_layer_list)
                list_of_nodes = list(set(list_of_nodes))
            last_len = -1
            last_layer_list = list_of_nodes
            while len(list_of_nodes) != last_len:
                last_len = len(list_of_nodes)
                tmp_list = []
                for n in last_layer_list:
                    inner_neighbor = graph.successors(n)
                    tmp_list.extend(inner_neighbor)
                last_layer_list = tmp_list
                list_of_nodes.extend(last_layer_list)
                list_of_nodes = list(set(list_of_nodes))

            new_partition_graph = graph.subgraph(list_of_nodes)

            # check if is just the conteiner of all the parts:
            if not check_just_conteiner(new_partition_graph, name):

                number_of_occurance = 0
                for multiple_occ in new_partition_graph:
                    if new_partition_graph.nodes[multiple_occ]["type"] == "NEXT_ASSEMBLY_USAGE_OCCURRENCE":
                        number_of_occurance += 1
                if number_of_occurance <= 1:
                    if name in partitions.keys():
                        name = name + "_" + str(count)
                        count += 1
                    partitions[name] = new_partition_graph
                else:
                    for i in range(number_of_occurance):
                        count_occ += 1
                        partitions[name + "_occ_" + str(count_occ)] = new_partition_graph


    if shape_rep_found:
        return partitions
    else:
        for node in graph:
            if graph.nodes[node]["type"] == "PRODUCT":
                name = graph.nodes[node]["PRODUCT_0"]
                list_of_nodes = [node]
                last_layer_list = [node]
                while len(last_layer_list) != 0:
                    tmp_list = []
                    for n in last_layer_list:
                        inner_neighbor = graph.predecessors(n)
                        tmp_list.extend(inner_neighbor)
                    last_layer_list = tmp_list
                    list_of_nodes.extend(last_layer_list)
                    list_of_nodes = list(set(list_of_nodes))
                last_len = -1
                last_layer_list = list_of_nodes
                while len(list_of_nodes) != last_len:
                    last_len = len(list_of_nodes)
                    tmp_list = []
                    for n in last_layer_list:
                        inner_neighbor = graph.successors(n)
                        tmp_list.extend(inner_neighbor)
                    last_layer_list = tmp_list
                    list_of_nodes.extend(last_layer_list)
                    list_of_nodes = list(set(list_of_nodes))

                new_partition_graph = graph.subgraph(list_of_nodes)
                partitions[name] = new_partition_graph

        return partitions





