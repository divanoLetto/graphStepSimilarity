from scipy.optimize import linear_sum_assignment
from Graph_similarity.beivecchimetodi.Histogram_intersection import histo_from_graph
from Graph_similarity.beivecchimetodi.Jaccard_similarity import jaccard_similarity
from Graphh.Graphh import Graphh
from Nodes.Node_utils import get_nodes_type_hystogramm, get_num_neighbor_for_node_type, get_composed_node_types
from Parser.Make_graph import *
from Printing_and_plotting.Printing import write_dataFrame, write_dataFrame_ordered_by_name, write_dataFrame_by_images
from mathematic_utils.mathematic_utils import *
import numpy as np
import os
from utils import make_schema


def main():
    file_name1 = 'plate1.stp'
    file_name2 = 'plate2.stp'
    file_name3 = 'trolley.stp'
    file_name4 = 'Coffee Pot.stp'
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
    file_names = [file_name1, file_name2,  file_name3, file_name4, file_name6, file_name7, file_name9, file_name10, file_name11, file_name12, file_name13]
    dump_file_name = ["dump_quad.step", "dump_quad_big.step",  "dump_rettangolo.step", "dump_quad_tri.step", "dump_piramid.step", file_name10, file_name11,file_name7, file_name8]  #, "dump_quad.step", "dump_quad_big.step", "dump_rettangolo.step", "dump_quad_tri.step", ]#"dump_cone.step"  , "dump_double_cone.step"]

    names = [os.path.splitext(f)[0] for f in file_names]
    headers_datas = [(parse_file(f)) for f in file_names]

    histograms = [get_nodes_type_hystogramm(f[1]) for f in headers_datas]

    print("Jaccard distance")
    jaccard_values = np.zeros((len(names), len(names)))
    for i1, histo1 in enumerate(histograms):
        set1 = set()
        for key1, value1 in histo1.items():
            set1.add(key1)
        for i2, histo2 in enumerate(histograms):
            set2 = set()
            for key2, value2 in histo2.items():
                set2.add(key2)
            jaccard_values[i1, i2] = jaccard_similarity(set1, set2)
    jaccard_schema = make_schema(jaccard_values, names)
    dataFrame_dict["jaccard_schema"] = jaccard_schema

    print("Histograms intersection")
    histo_diff = calc_histogram_intersections(histograms)
    histointersection_schema = make_schema(histo_diff, names)
    dataFrame_dict["histo_intersection_schema"] = histointersection_schema
    # for i, histo in enumerate(histograms):
    #     plt_histo(histo, show=False, save=False, file_name=names[i])

    print("Realizing graphs")
    Gs_simplexs_direct = [make_graph_simplex_direct(f) for f in file_names]
    graphs_direct = []
    for id_name, name in enumerate(names):
        g = Graphh(Gs_simplexs_direct[id_name], name)
        graphs_direct.append(g)

    for g in graphs_direct:
        g.print_composition()

    histoparts_values = np.zeros((len(names), len(names)))
    histoparts_times = np.zeros((len(names), len(names)))
    for i, gh_i in enumerate(graphs_direct):
        num_parts_i = gh_i.get_num_parts()
        for j, gh_j in enumerate(graphs_direct):
            num_parts_j = gh_j.get_num_parts()
            parts_matching_matrix = np.zeros((num_parts_i, num_parts_j))
            total_time = 0
            for k, gh_i_part in enumerate(gh_i.get_parts_graphs()):
                for t, gh_j_part in enumerate(gh_j.get_parts_graphs()):

                    start_time = time.time()
                    set1 = histo_from_graph(gh_i_part)
                    set2 = histo_from_graph(gh_j_part)
                    nc = 1 - histogram_intersection(set1, set2)
                    run_time = time.time() - start_time
                    total_time += run_time
                    if nc < 0.00001:
                        nc = float(0.0)
                    parts_matching_matrix[k, t] = nc

            tot_parts_match_matrix = Graphh.get_occ_match_matrix(parts_matching_matrix, gh_i, gh_j)
            row_ind, col_ind = linear_sum_assignment(tot_parts_match_matrix)
            best_assg = tot_parts_match_matrix[row_ind, col_ind].sum()
            histoparts_values[i, j] = best_assg + abs(gh_i.get_tot_num_parts_occurences() - gh_j.get_tot_num_parts_occurences())
            histoparts_times[i, j] = total_time
            print("Histo diff on parts of " + gh_i.get_name() + " and " + gh_j.get_name() + " took duration {:.4f}s.".format(total_time))

    histo_parts_schema = make_schema(histoparts_values, names)
    dataFrame_dict["histo_parts_schema"] = histo_parts_schema
    histo_parts_time_schema = make_schema(histoparts_times, names)
    dataFrame_time_dict["histo_parts_schema"] = histo_parts_time_schema

    list_parts = []
    for i, gh_i in enumerate(graphs_direct):
        for j, part in enumerate(gh_i.parts_graphs):
            list_parts.append((gh_i.name+"_"+gh_i.parts_names[j], part))
    gw_discrepancy_only_parts_values = np.zeros((len(list_parts), len(list_parts)))
    for i, (gh_i_name, gh_i) in enumerate(list_parts):
        for j, (gh_j_name, gh_j) in enumerate(list_parts):
            set1 = histo_from_graph(gh_i)
            set2 = histo_from_graph(gh_j)
            nc = 1 - histogram_intersection(set1, set2)
            gw_discrepancy_only_parts_values[i, j] = nc

    part_names = []
    for (n, _) in list_parts:
        if n not in part_names:
            part_names.append(n)
    # part_names = [name[0] for name in list_parts]
    gw_discrepancy_only_parts_schema = make_schema(gw_discrepancy_only_parts_values, part_names)
    dataFrame_dict_parts["histo_intersection_only_on_parts_schema"] = gw_discrepancy_only_parts_schema

    base_path = "C:/Users/Computer/PycharmProjects/graphStepSimilarity/results/beivecchimetodi/"
    high_max = [True, True, False]
    write_dataFrame(df_dict=dataFrame_dict, file_name='score_matching.xlsx',  high_max=high_max, base_path=base_path)
    write_dataFrame(df_dict=dataFrame_time_dict, file_name='time_matching.xlsx',  high_max=high_max, base_path=base_path)
    image_dir_path = "C:/Users/Computer/PycharmProjects/graphStepSimilarity/images/models_images/"
    write_dataFrame_by_images(df_dict=dataFrame_dict, file_name='retrieval_img_matching.xlsx',  names=names, by_min=high_max, base_path=base_path, image_dir_path=image_dir_path)

    image_dir_path = "C:/Users/Computer/PycharmProjects/graphStepSimilarity/images/models_images/parts/"
    high_max = [False]
    write_dataFrame(df_dict=dataFrame_dict_parts, file_name='parts_score_matching.xlsx',  high_max=high_max, base_path=base_path)
    write_dataFrame_by_images(df_dict=dataFrame_dict_parts, file_name='parts_retrieval_img_matching.xlsx', names=part_names, by_min=high_max, base_path=base_path, image_dir_path=image_dir_path, scale_x=0.03, scale_y=0.03)


if __name__ == "__main__":
    main()

