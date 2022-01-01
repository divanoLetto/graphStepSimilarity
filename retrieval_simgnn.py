from scipy.optimize import linear_sum_assignment
from Graph_similarity.SimGNN.src.simgnn_big import SimGNNTrainer_Big, SimGNN_Big
from Graph_similarity.SimGNN.src.utils import tab_printer
from Graph_similarity.SimGNN.src.param_parser import parameter_parser_1024, parameter_parser_256, parameter_parser_512, parameter_parser_slim_256
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
    model_name = "model256_26_12.zip"
    model_save_path = base_path + "/Datasets/DS_4/results/simgnn/256/" + model_name
    model_load_path = model_save_path  # None
    excel_save_name_comp = "simgnn_components_score.xlsx"
    excel_save_name_models = "simgnn_models_score.xlsx"
    labels_name = "labels_saves.txt"
    labels_path = base_path + "/Graph_similarity/SimGNN/saves/" + labels_name
    epochs = 1
    args = parameter_parser_256(model_save_path, model_load_path, epochs=epochs)
    seed = 0
    random.seed(seed)

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

    tab_printer(args)
    trainer = SimGNNTrainer_Big(args, all_set, [], labels_path)
    trainer.load()

    tot_num_parts = 0
    for i, gh_i in enumerate(graphs_direct):
        tot_num_parts += len(gh_i.parts)

    # Retrieval for models
    num_models = len(graphs_direct)
    simgnn_values = np.zeros((num_models, num_models))
    names_models = []
    times = []
    for i, gh_i in enumerate(graphs_direct):
        names_models.append(gh_i.get_name())
        for j, gh_j in enumerate(graphs_direct):
            i_name = gh_i.get_name()
            j_name = gh_j.get_name()
            print("Evaluating " + i_name + " " + j_name)

            start_time = time.time()
            num_gh_i_parts = len(gh_i.parts)
            num_gh_j_parts = len(gh_j.parts)
            components_matrix = np.zeros((num_gh_i_parts, num_gh_j_parts))
            for z, ph_i in enumerate(gh_i.parts):
                for k, ph_j in enumerate(gh_j.parts):
                    prediction = trainer.evaluate(ph_i, ph_j)
                    components_matrix[z, k] = prediction

            comp_occ_matrix = Graphh.get_occ_match_matrix(components_matrix, gh_i, gh_j)
            # algoritmo ungherese
            row_ind, col_ind = linear_sum_assignment(comp_occ_matrix)
            best_assg = comp_occ_matrix[row_ind, col_ind].sum()
            simgnn_values[i, j] = best_assg + abs(gh_i.get_tot_num_parts_occurences() - gh_j.get_tot_num_parts_occurences())
            times.append(time.time() - start_time)

    print("Mean models evaluation time: " + str(np.mean(times)))
    df_score_models = make_schema(simgnn_values, names_models)
    write_dataFrame(dataframe=df_score_models, file_name=excel_save_name_models, base_path=results_path, high_max=False)

    # Retrieval for components
    names_parts = []
    i = 0
    times = []
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

                    start_time = time.time()
                    prediction = trainer.evaluate(ph_i, ph_j)
                    times.append(time.time() - start_time)
                    simgnn_values[i, j] = prediction
                    j += 1
            i += 1

    print("Mean components evaluation time: " + str(np.mean(times)))
    df_score_componets = make_schema(simgnn_values, names_parts)
    write_dataFrame(dataframe=df_score_componets, file_name=excel_save_name_comp, base_path=results_path, high_max=False)


if __name__ == "__main__":
    main()
