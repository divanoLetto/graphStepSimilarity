from Graphh.Graphh import Graphh
from Graph_similarity.Wasserstein.Wasserstein import calculate_node_errors, calc_wassertein_discrepancy
from Parser.Make_graph import *
from Printing_and_plotting.Printing import write_dataFrame, write_dataFrame_by_images
import numpy as np
import os
from utils import make_schema
from scipy.optimize import linear_sum_assignment
import pathlib


def main():
    file_names = []
    base_path = str(pathlib.Path(__file__).parent).replace("\\", "/")
    path_dataset = base_path + "/Dataset/"
    results_path = base_path + "/results/wassertein/"
    graph_saves_path = base_path + "/Graphh/graph_save/simplex_direct/"
    # image_models_dir_path = base_path + "/images/models_images/"
    # image_parts_dir_path = base_path + "/images/models_images/parts/"
    parts_graph_saves_path = base_path + "/matrix_saves/GwGt_direct/parts_graph_match/"
    # Wasserstein options
    num_iter = 2000
    ot_dict = {'loss_type': 'L2', 'ot_method': 'proximal', 'beta': 0.025, 'outer_iteration': num_iter,
               'iter_bound': 1e-30, 'inner_iteration': 2, 'sk_bound': 1e-30, 'node_prior': 1e3, 'max_iter': 4,
               'cost_bound': 1e-26, 'update_p': False, 'lr': 0, 'alpha': 0}

    for file in os.listdir(path_dataset):
        if file.endswith(".stp") or file.endswith(".step"):
            file_names.append(file)
    names = [os.path.splitext(f)[0] for f in file_names]

    # dataframe for save results in excel

    dataFrame_dict_parts = {}

    print("Realizing graphs")
    Gs_simplexs_direct = [make_graph_simplex_direct(f, graph_saves_path, dataset_path=path_dataset) for f in file_names]
    graphs_direct = []
    for id_name, name in enumerate(names):
        g = Graphh(Gs_simplexs_direct[id_name], name)
        graphs_direct.append(g)

    for g in graphs_direct:
        g.print_composition()

    # Struct delle parti: (grafo parte, grafo modello)
    list_parts = []
    for i, model in enumerate(graphs_direct):
        for j, part in enumerate(model.parts):
            list_parts.append((part, model))
    gw_discrepancy_only_parts_values = np.zeros((len(list_parts), len(list_parts)))
    for i, (part_i, model_i) in enumerate(list_parts):
        for j, (part_j, model_j) in enumerate(list_parts):

            p_i = part_i
            p_j = part_j
            m_i = model_i
            m_j = model_j
            if gw_discrepancy_only_parts_values[j, i] != 0:
                gw_discrepancy_only_parts_values[i, j] = gw_discrepancy_only_parts_values[j, i]
            else:
                if p_j.number_of_nodes() < p_i.number_of_nodes():
                    # Swap operation: wasserstein distance is faster when p_i is smaller
                    tmp = p_i
                    p_i = p_j
                    p_j = tmp
                    tmp = m_i
                    m_i = m_j
                    m_j = tmp

                part_i_name = m_i.get_name() + "-" + p_i.get_name()
                part_j_name = m_j.get_name() + "-" + p_j.get_name()

                pairs_idx, pairs_name, pairs_confidence, run_time = calc_wassertein_discrepancy(g_i=p_i.get_full_graph(), g_j=p_j.get_full_graph(), name_i=part_i_name, name_j=part_j_name, ot_dict=ot_dict, base_path=parts_graph_saves_path)
                nc = calculate_node_errors(pairs=pairs_name, confidence=pairs_confidence, g1=p_i.get_full_graph(), g2=p_j.get_full_graph())
                gw_discrepancy_only_parts_values[i, j] = nc

    part_names = [p_m_graph[1].get_name() + "-" + p_m_graph[0].get_name() for p_m_graph in list_parts]
    gw_discrepancy_only_parts_schema = make_schema(gw_discrepancy_only_parts_values, part_names)
    dataFrame_dict_parts["gw_parts_schema"] = gw_discrepancy_only_parts_schema

    high_max = [False]
    write_dataFrame(df_dict=dataFrame_dict_parts, file_name='parts_score_match.xlsx',  base_path=results_path, high_max=high_max)


if __name__ == "__main__":
    main()

