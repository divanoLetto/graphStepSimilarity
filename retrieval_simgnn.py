from Graph_similarity.SimGNN.src.simgnn_big import SimGNNTrainer_Big, SimGNN_Big
from Graph_similarity.SimGNN.src.utils import tab_printer
from Graph_similarity.SimGNN.src.param_parser import parameter_parser
from Graphh.Graphh import Graphh
from Parser.Make_graph import *
import os
import random
import numpy as np
import pathlib
from Printing_and_plotting.Printing import write_dataFrame
from utils import make_schema


def main():
    file_names = []
    base_path = str(pathlib.Path(__file__).parent)
    path_dataset = base_path + "/Datasets/DS_4/Models/"
    results_path = base_path + "/Datasets/DS_4/results/simgnn/256/"
    graph_saves_path = base_path + "/Graphh/graph_save/simplex_direct/"
    model_name = "model256_26_12"
    model_save_path = base_path + "/Datasets/DS_4/results/simgnn/256/" + model_name
    model_load_path = model_save_path  # None
    excel_save_name = "simgnn_score.xlsx"
    labels_name = "labels_saves.txt"
    labels_path = base_path + "/Graph_similarity/SimGNN/saves/" + labels_name
    epochs = 1
    seed = 0
    random.seed(seed)
    dataFrame_dict = {}

    for file in os.listdir(path_dataset):
        if file.endswith(".stp") or file.endswith(".step"):
            file_names.append(file)
    names = [os.path.splitext(f)[0] for f in file_names]

    print("Realizing graphs")
    list_graphs = [make_graph_simplex_direct(f, graph_saves_path, dataset_path=path_dataset) for f in file_names]
    graphs_direct = []
    for id_name, name in enumerate(names):
        g = Graphh(list_graphs[id_name], name)
        graphs_direct.append(g)

    for g in graphs_direct:
        g.print_composition()

    all_set = []
    for i, gh_i in enumerate(graphs_direct):
        for ph_i in gh_i.parts:
            for j, gh_j in enumerate(graphs_direct):
                for ph_j in gh_j.parts:
                    dist = 0  # dont care
                    all_set.append([ph_i, ph_j, dist])

    args = parameter_parser(model_save_path, model_load_path, epochs=epochs)
    tab_printer(args)
    trainer = SimGNNTrainer_Big(args, all_set, [], labels_path)
    trainer.load()

    tot_num_parts = 0
    for i, gh_i in enumerate(graphs_direct):
        tot_num_parts += len(gh_i.parts)

    names_parts = []
    i=0
    simgnn_values = np.zeros((tot_num_parts, tot_num_parts))
    for gh_i in graphs_direct:
        for ph_i in gh_i.parts:
            names_parts.append(gh_i.get_name()+"-"+ph_i.get_name())
            j = 0
            for gh_j in graphs_direct:
                for ph_j in gh_j.parts:
                    i_name = gh_i.get_name()+"-"+ph_i.get_name()
                    j_name = gh_j.get_name()+"-"+ph_j.get_name()
                    print("Evaluating " + i_name + " " + j_name)

                    prediction = trainer.evaluate(ph_i, ph_j)
                    simgnn_values[i, j] = prediction
                    j += 1
            i += 1

    gw_discrepancy_only_parts_schema = make_schema(simgnn_values, names_parts)
    dataFrame_dict["simgnn_schema"] = gw_discrepancy_only_parts_schema
    high_max = [False]
    write_dataFrame(df_dict=dataFrame_dict, file_name=excel_save_name,  base_path=results_path, high_max=high_max)


if __name__ == "__main__":
    main()
