from scipy.optimize import linear_sum_assignment

from Graphh.Graphh import Graphh
from Graph_similarity.Wasserstein.Wasserstein import calculate_node_errors, calc_wassertein_discrepancy
from Parser.Make_graph import *
from Printing_and_plotting.Printing import write_dataFrame, write_dataFrame_by_images
import numpy as np
import os
from utils import make_schema
import pathlib


def main():
    file_names = []
    base_path = str(pathlib.Path(__file__).parent).replace("\\", "/")
    path_dataset = base_path + "/Datasets/DS_4/Models/"
    results_path = base_path + "/Datasets/DS_4/results/wasserstein/"
    graph_saves_path = base_path + "/Graphh/graph_save/simplex_direct/"
    parts_graph_saves_path = base_path + "/Datasets/DS_4/results/wasserstein/parts_graph_match/"
    excel_name_1 = 'ww_components_score.xlsx'
    excel_name_2 = 'ww_models_score.xlsx'
    # Wasserstein options
    num_iter = 20
    ot_dict = {'loss_type': 'L2', 'ot_method': 'proximal', 'beta': 0.025, 'outer_iteration': num_iter,
               'iter_bound': 1e-30, 'inner_iteration': 2, 'sk_bound': 1e-30, 'node_prior': 1e3, 'max_iter': 4,
               'cost_bound': 1e-26, 'update_p': False, 'lr': 0, 'alpha': 0}

    for file in os.listdir(path_dataset):
        if file.endswith(".stp") or file.endswith(".step"):
            file_names.append(file)
    names = [os.path.splitext(f)[0] for f in file_names]

    print("Realizing graphs")
    Gs_simplexs_direct = [make_graph_simplex_direct(f, graph_saves_path, dataset_path=path_dataset) for f in file_names]
    graphs_direct = []
    for id_name, name in enumerate(names):
        g = Graphh(Gs_simplexs_direct[id_name], name)
        graphs_direct.append(g)

    for g in graphs_direct:
        g.print_composition()

    # Retrieval for models
    num_models = len(graphs_direct)
    simgnn_values = np.zeros((num_models, num_models))
    names_models = []
    for i, gh_i in enumerate(graphs_direct):
        names_models.append(gh_i.get_name())
        for j, gh_j in enumerate(graphs_direct):
            print("Evaluating " + gh_i.get_name() + " " + gh_j.get_name())

            num_gh_i_parts = len(gh_i.parts)
            num_gh_j_parts = len(gh_j.parts)
            models_matrix = np.zeros((num_gh_i_parts, num_gh_j_parts))
            for z, ph_i in enumerate(gh_i.parts):
                for k, ph_j in enumerate(gh_j.parts):

                    p_i = ph_i
                    p_j = ph_j
                    m_i = gh_i
                    m_j = gh_j
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

                    pairs_idx, pairs_name, pairs_confidence, run_time = calc_wassertein_discrepancy(
                        g_i=p_i.get_full_graph(),
                        g_j=p_j.get_full_graph(),
                        name_i=part_i_name,
                        name_j=part_j_name,
                        ot_dict=ot_dict,
                        base_path=parts_graph_saves_path)

                    nc = calculate_node_errors(pairs=pairs_name,
                                               confidence=pairs_confidence,
                                               g1=p_i.get_full_graph(),
                                               g2=p_j.get_full_graph())
                    models_matrix[z, k] = nc

            # algoritmo ungherese
            row_ind, col_ind = linear_sum_assignment(models_matrix)
            best_assg = models_matrix[row_ind, col_ind].sum()
            simgnn_values[i, j] = best_assg + abs(
                gh_i.get_tot_num_parts_occurences() - gh_j.get_tot_num_parts_occurences())

    df_score_models = make_schema(simgnn_values, names_models)
    write_dataFrame(dataframe=df_score_models, file_name=excel_name_2, base_path=results_path, high_max=False)

    # Struct delle parti: (grafo componente, grafo modello)
    list_parts = []
    for i, model in enumerate(graphs_direct):
        for j, part in enumerate(model.parts):
            list_parts.append((part, model))
    ww_components_matrix = np.zeros((len(list_parts), len(list_parts)))
    for i, (part_i, model_i) in enumerate(list_parts):
        for j, (part_j, model_j) in enumerate(list_parts):

            p_i = part_i
            p_j = part_j
            m_i = model_i
            m_j = model_j
            # If already computed d(B, A)
            if ww_components_matrix[j, i] != 0:
                ww_components_matrix[i, j] = ww_components_matrix[j, i]
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
                ww_components_matrix[i, j] = nc

    part_names = [p_m_graph[1].get_name() + "-" + p_m_graph[0].get_name() for p_m_graph in list_parts]
    df_score_components = make_schema(ww_components_matrix, part_names)
    write_dataFrame(dataframe=df_score_components, file_name=excel_name_1,  base_path=results_path, high_max=False)


if __name__ == "__main__":
    main()

