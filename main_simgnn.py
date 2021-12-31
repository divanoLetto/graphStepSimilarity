from Graph_similarity.SimGNN.src.simgnn_big import SimGNNTrainer_Big
from Graph_similarity.SimGNN.src.utils import tab_printer
from Graph_similarity.SimGNN.src.param_parser import parameter_parser
from Graphh.Graphh import Graphh
from Parser.Make_graph import *
import os
import pandas as pd
import pathlib
import random
from utils import split_training_testset


def main():
    file_names = []
    base_path = str(pathlib.Path(__file__).parent)
    path_dataset = base_path + "/Datasets/DS_4/Models/"
    excel_path = base_path + "/Datasets/DS_4/results/wasserstein/ww_parts_score.xlsx"
    graph_saves_path = base_path + "/Graphh/graph_save/simplex_direct/"
    model_name = "model256_26_12"
    model_save_path = base_path + "/Datasets/DS_4/results/simgnn/" + model_name
    model_load_path = model_save_path  # None
    labels_name = "labels_saves.txt"
    labels_path = base_path + "/Graph_similarity/SimGNN/saves/" + labels_name
    epochs = 70
    train = False
    load = True
    perc_train_test = 0.7
    save_epochs = 5
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
    df = pd.read_excel(excel_path)
    double_hash = {}
    for i in range(len(df.index.values)-1):
        for j in range(len(df.columns.values)-1):
            i_name = df.iloc[i+1, 0]
            j_name = df.iloc[0, j+1]
            dist = df.iloc[i + 1, j + 1]
            if i_name not in double_hash.keys():
                double_hash[i_name] = {}
            double_hash[i_name][j_name] = dist

    for i, gh_i in enumerate(graphs_direct):
        for ph_i in gh_i.parts:
            for j, gh_j in enumerate(graphs_direct):
                for ph_j in gh_j.parts:
                    i_name = gh_i.get_name()+"-"+ph_i.get_name()
                    j_name = gh_j.get_name()+"-"+ph_j.get_name()
                    dist = double_hash[i_name][j_name]
                    if dist > 1:  # todo fix this
                        dist = 1
                    all_set.append([ph_i, ph_j, dist])

    random.shuffle(all_set)
    one_set = []
    not_one_set = []
    # Fix number of examples similar and dissimilar to 50% 5
    for example in all_set:
        if example[2] != 1:
            not_one_set.append(example)
        else:
            one_set.append(example)
    one_set = one_set[:len(not_one_set)]
    all_set = one_set + not_one_set

    training_set, test_set = split_training_testset(all_set, perc_train_test)
    args = parameter_parser(model_save_path, model_load_path, epochs=epochs)
    tab_printer(args)
    trainer = SimGNNTrainer_Big(args, training_set, test_set, labels_path)
    if load:
        print("Load")
        trainer.load()
    if train:
        print("Fit")
        trainer.fit(save_epochs)
        trainer.save()
    trainer.score()


if __name__ == "__main__":
    main()
