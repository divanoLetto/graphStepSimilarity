from Graphh.Graphh import Graphh
from Graph_similarity.Wasserstein.Wasserstein import calculate_node_errors, calc_wassertein_discrepancy
from Parser.Make_graph import *
from Printing_and_plotting.Printing import write_dataFrame, write_dataFrame_by_images
import numpy as np
import os
from utils import make_schema
from scipy.optimize import linear_sum_assignment


def main():
    file_name1 = 'plate1.stp'
    file_name2 = 'plate2.stp'
    file_name3 = 'trolley.stp'
    file_name4 = 'Coffee Pot.txt'
    file_name5 = 'Engine1.stp'  # to long
    file_name6 = 'staabva.stp'
    file_name7 = 'stoabmm.stp'
    file_name8 = 'stoadmm.stp'  # maybe to long
    file_name9 = 'strainer.stp'
    file_name10 = 'TopFlange01.stp'
    file_name11 = 'TopFlange02.stp'
    file_name12 = 'Valve01.stp'
    file_name13 = 'wheel.stp'

    dataFrame_dict = {}
    dataFrame_dict_parts = {}
    dataFrame_time_dict = {}
    # dump_file_names = ["dump_quad.step", "dump_quad_big.step",  "dump_rettangolo.step", "dump_quad_tri.step", "dump_piramid.step"]
    file_names = [file_name1, file_name2, file_name3, file_name4, file_name6, file_name7, file_name8, file_name9,file_name10, file_name11, file_name12, file_name13]

    names = [os.path.splitext(f)[0] for f in file_names]

    print("Realizing graphs")
    Gs_simplexs_direct = [make_graph_simplex_direct(f) for f in file_names]
    graphs_direct = []
    for id_name, name in enumerate(names):
        g = Graphh(Gs_simplexs_direct[id_name], name)
        graphs_direct.append(g)

    for g in graphs_direct:
        g.print_composition()

    num_iter = 3
    ot_dict = {'loss_type': 'L2', 'ot_method': 'proximal', 'beta': 0.025, 'outer_iteration': num_iter,
               'iter_bound': 1e-30, 'inner_iteration': 2, 'sk_bound': 1e-30, 'node_prior': 1e3, 'max_iter': 4,
               'cost_bound': 1e-26, 'update_p': False, 'lr': 0, 'alpha': 0}

    print("\nGromov-Wasserstein discrepancy on parts:")
    base_path = "C:/Users/Computer/PycharmProjects/graphStepSimilarity/matrix_saves/GwGt_direct/parts_graph_match/"
    gw_discrepancy_parts_values = np.zeros((len(names), len(names)))
    gw_discrepancy_parts_times = np.zeros((len(names), len(names)))
    # Parallelizzabile
    for i, gh_i in enumerate(graphs_direct):
        num_parts_i = gh_i.get_num_parts()
        for j, gh_j in enumerate(graphs_direct):
            num_parts_j = gh_j.get_num_parts()
            parts_matching_matrix = np.zeros((num_parts_i, num_parts_j))
            total_time = 0
            for k, gh_i_part in enumerate(gh_i.get_parts_graphs()):
                for t, gh_j_part in enumerate(gh_j.get_parts_graphs()):

                    # if gh_j_part.number_of_nodes() < gh_i_part.number_of_nodes():
                    #     tmp = gh_i_part
                    #     gh_i_part = gh_j_part
                    #     gh_j_part = tmp
                    gh_i_part_name = gh_i.get_name_of_part(k)
                    gh_j_part_name = gh_j.get_name_of_part(t)

                    pairs_idx, pairs_name, pairs_confidence, run_time = calc_wassertein_discrepancy(gh_i_part, gh_j_part, names[i]+"-"+gh_i_part_name, names[j]+"-"+gh_j_part_name, ot_dict, base_path)
                    nc = calculate_node_errors(pairs_name, pairs_confidence, gh_i_part, gh_j_part)
                    total_time += run_time
                    parts_matching_matrix[k, t] = nc

            tot_parts_match_matrix = Graphh.get_occ_match_matrix(parts_matching_matrix, gh_i, gh_j)
            row_ind, col_ind = linear_sum_assignment(tot_parts_match_matrix)
            best_assg = tot_parts_match_matrix[row_ind, col_ind].sum()
            gw_discrepancy_parts_values[i, j] = best_assg + abs(gh_i.get_tot_num_parts_occurences() - gh_j.get_tot_num_parts_occurences())  # TODO o abs() o max() o niente
            gw_discrepancy_parts_times[i, j] = total_time
            print("Gwl of " + gh_i.get_name() + " and " + gh_j.get_name() + " took duration {:.4f}s.".format(total_time))

    gw_discrepancy_parts_schema = make_schema(gw_discrepancy_parts_values, names)
    dataFrame_dict["gw_parts_schema"] = gw_discrepancy_parts_schema
    gw_discrepancy_parts_time_schema = make_schema(gw_discrepancy_parts_times, names)
    dataFrame_time_dict["gw_parts_schema"] = gw_discrepancy_parts_time_schema

    # tripletta delle parti: (nome, grafo, nome del padre)
    list_parts = []
    for i, gh_i in enumerate(graphs_direct):
        for j, part in enumerate(gh_i.parts_graphs):
            list_parts.append((gh_i.parts_names[j], part, gh_i.name))
    gw_discrepancy_only_parts_values = np.zeros((len(list_parts), len(list_parts)))
    for i, (gh_i_name, gh_i, name_i) in enumerate(list_parts):
        for j, (gh_j_name, gh_j, name_j) in enumerate(list_parts):

            # if gh_j.number_of_nodes() < gh_i.number_of_nodes():
            #     print("Swap")
            #     tmp = gh_i
            #     gh_i = gh_j
            #     gh_j = tmp
            #     tmp = gh_i_name
            #     gh_i_name = gh_j_name
            #     gh_j_name = tmp

            part_i_name = name_i + "-" + gh_i_name
            part_j_name = name_j + "-" + gh_j_name
            pairs_idx, pairs_name, pairs_confidence, run_time = calc_wassertein_discrepancy(gh_i, gh_j, name_i=part_i_name, name_j=part_j_name, ot_dict=ot_dict, base_path=base_path)
            try:
                nc = calculate_node_errors(pairs_name, pairs_confidence, gh_i, gh_j)
            except:
                print("Error at " + part_i_name + "_" + part_j_name)
            gw_discrepancy_only_parts_values[i, j] = nc

    part_names = [name[2]+"_"+name[0] for name in list_parts]
    gw_discrepancy_only_parts_schema = make_schema(gw_discrepancy_only_parts_values, part_names)
    dataFrame_dict_parts["gw_only_on_parts_schema"] = gw_discrepancy_only_parts_schema

    base_path = "C:/Users/Computer/PycharmProjects/graphStepSimilarity/results/wassertein/"
    image_dir_path = "C:/Users/Computer/PycharmProjects/graphStepSimilarity/images/models_images/"
    high_max = [False, False]
    write_dataFrame(df_dict=dataFrame_dict, file_name='score_match.xlsx', base_path=base_path, high_max=high_max)
    write_dataFrame_by_images(df_dict=dataFrame_dict, file_name='retrieval_img_match.xlsx', names=names, base_path=base_path, image_dir_path=image_dir_path, by_min=high_max)
    write_dataFrame(df_dict=dataFrame_time_dict, file_name='time_match.xlsx', base_path=base_path, high_max=high_max)

    high_max = [False]
    image_dir_path = "C:/Users/Computer/PycharmProjects/graphStepSimilarity/images/models_images/parts/"
    write_dataFrame(df_dict=dataFrame_dict_parts, file_name='parts_score_match.xlsx',  base_path=base_path, high_max=high_max)
    write_dataFrame_by_images(df_dict=dataFrame_dict_parts, file_name='parts_retrieval_match.xlsx', names=part_names, base_path=base_path, image_dir_path=image_dir_path, by_min=high_max, scale_x=0.03, scale_y=0.03)


if __name__ == "__main__":
    main()

