import os
import pathlib
from Graphh.Graphh import Graphh
from Parser.Make_graph import make_graph_simplex_direct


def calc_lines_files():

    base_path = str(pathlib.Path(__file__).parent)
    path_dataset = base_path + "/Datasets/"
    dataset_name = "DS_6/Models/"
    path_dataset = path_dataset + dataset_name
    print("Models at location: '" + dataset_name + "' properties:\n")
    graph_saves_path = base_path + "/Graphh/graph_save/simplex_direct/"

    file_names = []
    for file in os.listdir(path_dataset):
        if file.endswith(".stp") or file.endswith(".step"):
            file_names.append(file)
    names = [os.path.splitext(f)[0] for f in file_names]

    list_graphs = [make_graph_simplex_direct(f, graph_saves_path, dataset_path=path_dataset) for f in file_names]
    graphs_direct = []
    for id_name, name in enumerate(names):
        g = Graphh(list_graphs[id_name], name)
        graphs_direct.append(g)

    num_models = len(graphs_direct)
    print(str(num_models) + " models")
    num_components = 0
    for g in graphs_direct:
        num_components += g.get_num_parts()
    print(str(num_components) + " total components\n")

    for g in graphs_direct:
        print(g.get_name())
    print("")

    for g in graphs_direct:
        g.print_composition()

    print("\nNumber of lines x model:")
    for filename in os.listdir(path_dataset):
        with open(os.path.join(path_dataset, filename), 'r') as f: # open in readonly mode
            num_lines = sum(1 for line in f)
            print("  "+str(filename) + " : " + str(num_lines))


calc_lines_files()
