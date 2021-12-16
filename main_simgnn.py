from Graph_similarity.SimGNN.src.simgnn_big import SimGNNTrainer_Big
from Graph_similarity.SimGNN.src.utils import tab_printer
from Graph_similarity.SimGNN.src.simgnn import SimGNNTrainer
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
    path_dataset = base_path + "/Dataset/"
    results_path = base_path + "/results/simgnn/"
    image_dir_path = base_path + "/images/models_images/"
    excel_path = base_path + "/results/wassertein/parts_score_match.xlsx"
    graph_saves_path = base_path + "/Graphh/graph_save/simplex_direct/"
    model_name = "model1"
    model_save_path = base_path + "/Graph_similarity/SimGNN/saves/" + model_name
    perc_train_test = 0.7
    save_epochs = 5

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

    # lista delle parti
    list_parts = []
    df = pd.read_excel(excel_path)
    for i, gh_i in enumerate(graphs_direct):
        for j, part in enumerate(gh_i.parts):
            list_parts.append(part)
    for i, part_i in enumerate(list_parts):
        for j, part_j in enumerate(list_parts):
            dist = df.iloc[i+1, j+1]
            if dist > 1:  # todo fix this
                dist = 1
            all_set.append([part_i, part_j, dist])

    training_set, test_set = split_training_testset(all_set, perc_train_test)
    args = parameter_parser(model_save_path)
    tab_printer(args)
    trainer = SimGNNTrainer_Big(args, training_set, test_set)
    if args.load_path:
        trainer.load()
    else:
        trainer.fit(save_epochs)
    if args.save_path:
        trainer.save()
    trainer.score()


if __name__ == "__main__":
    main()
