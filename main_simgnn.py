from Graph_similarity.SimGNN.src.utils import tab_printer
from Graph_similarity.SimGNN.src.simgnn import SimGNNTrainer
from Graph_similarity.SimGNN.src.param_parser import parameter_parser
from Graphh.Graphh import Graphh
from Parser.Make_graph import *
import os
import pandas as pd


def main():
    file_name1 = 'plate1.stp'
    file_name2 = 'plate2.stp'
    file_name3 = 'trolley.stp'
    file_name4 = 'Coffee Pot.stp'
    file_name6 = 'staabva.stp'
    file_name7 = 'stoabmm.stp'
    file_name9 = 'strainer.stp'
    file_name10 = 'TopFlange01.stp'
    file_name11 = 'TopFlange02.stp'
    file_name12 = 'Valve01.stp'
    file_name13 = 'wheel.stp'

    file_names = [file_name1, file_name2, file_name3, file_name4, file_name6, file_name7, file_name9, file_name10,
                  file_name11, file_name12, file_name13]

    names = [os.path.splitext(f)[0] for f in file_names]

    print("Realizing graphs")
    list_graphs = [make_graph_simplex_direct(f) for f in file_names]
    graphs_direct = []
    for id_name, name in enumerate(names):
        g = Graphh(list_graphs[id_name], name)
        graphs_direct.append(g)

    for g in graphs_direct:
        g.print_composition()

    training_set = []
    test_set = []

    # lista delle parti
    list_parts = []
    excel_path = "C:/Users/Computer/PycharmProjects/graphStepSimilarity/results/wassertein/parts_score_match.xlsx"
    df = pd.read_excel(excel_path)
    for i, gh_i in enumerate(graphs_direct):
        for j, part in enumerate(gh_i.parts):
            list_parts.append(part)
    for i, part_i in enumerate(list_parts):
        for j, part_j in enumerate(list_parts):
            dist = df.iloc[i+1, j+1]
            training_set.append([part_i, part_j, dist])

    args = parameter_parser()
    tab_printer(args)
    trainer = SimGNNTrainer(args, training_set, test_set)
    if args.load_path:
        trainer.load()
    else:
        trainer.fit()
    trainer.score()
    if args.save_path:
        trainer.save()


if __name__ == "__main__":
    main()
