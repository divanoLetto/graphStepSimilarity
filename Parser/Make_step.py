import os
from Parser.Parser import parse_file, parse_header


def make_step_from_graph(step_file, full_graph, graph, graph_file_name, full_graph_conteiner):
    base_path = "/Dataset/recostructed/"

    full_graph_nodes = full_graph.nodes()
    graph_nodes = graph.nodes()
    difference = [item for item in full_graph_nodes if item not in graph_nodes]
    conteiner_nodes = set(full_graph_conteiner.nodes())
    difference = set(difference)
    difference = difference - conteiner_nodes
    _, datas = parse_file(step_file)
    headers = parse_header(step_file)

    with open(base_path + graph_file_name + ".step", 'a') as graph_file:
        graph_file.truncate(0)
        for line in headers:
            graph_file.write(line + "\n")
        graph_file.write("\n")
        graph_file.write("DATA;" + "\n")
        for line in datas[:-1]:
            id_type_arguments = line.split("=")
            id = id_type_arguments[0]
            if id not in difference:
                graph_file.write(line+";\n")

    graph_file.close()
