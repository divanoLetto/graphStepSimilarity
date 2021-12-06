from Graph_similarity.Graph_similarity import spectrum_distance
from Graphh.Node_utils import get_num_neighbor_for_node_type, get_composed_node_types
from Parser.Make_graph import *
from Printing_and_plotting.Printing import write_dataFrame, write_dataFrame_ordered_by_name
from mathematic_utils.mathematic_utils import *
import numpy as np
import os
from utils import make_schema


def main():
    file_name1 = 'just_part_bearing_plate1.stp'
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
    dataFrame_time_dict = {}
    file_names = [file_name1, file_name2,  file_name3, file_name4, file_name6, file_name7, file_name9, file_name10, file_name11, file_name12, file_name13]
    dump_file_name = ["dump_quad.step", "dump_quad_big.step",  "dump_rettangolo.step", "dump_quad_tri.step", "dump_piramid.step", file_name10, file_name11,file_name7, file_name8]  #, "dump_quad.step", "dump_quad_big.step", "dump_rettangolo.step", "dump_quad_tri.step", ]#"dump_cone.step"  , "dump_double_cone.step"]
    # file_names = dump_file_name

    names = [os.path.splitext(f)[0] for f in file_names]
    headers_datas = [(parse_file(f)) for f in file_names]

    print("Realizing graphs")
    Gs_simplexs = [make_graph_simplex(f) for f in file_names]
    Gs_simplexs_direct = [make_graph_simplex_direct(f) for f in file_names]
    # Gs_simplexs_direct = [make_graph_scomposed_direct(f) for f in file_names]

    for i, graph in enumerate(Gs_simplexs):
        print("Graphh of file " + str(names[i]) + " has " + str(graph.number_of_nodes()) + " nodes and " + str(graph.number_of_edges()) + " edges")
    g_test = Gs_simplexs_direct[0]
    composed_node_type_dict = get_composed_node_types(g_test)
    test_dict = get_num_neighbor_for_node_type(g_test)

    models_spits_parts = []
    for i_name, name in enumerate(names):
        current_model_parts = {}
        current_graph = Gs_simplexs_direct[i_name]
        # current_model_parts["complete_graph"] = current_graph
        parts_dict = spit_graph_in_parts(current_graph)
        current_model_parts.update(parts_dict)
        models_spits_parts.append(current_model_parts)
        print(name + " is composed by " + str(len(current_model_parts)) + " components:")
        for key in current_model_parts.keys():
            print("   " + key+": " + str(current_model_parts[key].number_of_nodes()))

    adj_eigenvalues = calc_adj_eigenvalues(Gs_simplexs, names)
    for i in range(len(adj_eigenvalues)):
        adj_eigenvalues[i] = np.sort(adj_eigenvalues[i])

    num_models = len(names)
    print("Finished adj eigenvalues calcolation")

    print("\nAdjacency eigenvalues distances (firsts values):")
    adj_eigenvalues_distances_firsts_values = np.zeros((num_models, num_models))
    for i, adj_eigen_i in enumerate(adj_eigenvalues):
        for j, adj_eigen_j in enumerate(adj_eigenvalues):
            adj_eigenvalues_distances_firsts_values[i, j] = spectrum_distance(adj_eigen_i, adj_eigen_j, first_values_comparison=True)
    adj_firsts_schema = make_schema(adj_eigenvalues_distances_firsts_values, names)
    dataFrame_dict["adj_firsts_schema"] = adj_firsts_schema

    print("\nAdjacency eigenvalues distances (lasts values):")
    adj_eigenvalues_distances_lasts_values = np.zeros((num_models, num_models))
    for i, adj_eigen_i in enumerate(adj_eigenvalues):
        for j, adj_eigen_j in enumerate(adj_eigenvalues):
            adj_eigenvalues_distances_lasts_values[i, j] = spectrum_distance(adj_eigen_i, adj_eigen_j, first_values_comparison=False)
    adj_lasts_schema = make_schema(adj_eigenvalues_distances_lasts_values, names)
    dataFrame_dict["adj_lasts_schema"] = adj_lasts_schema

    laplacian_eigenvalues = calc_laplacian_eigenvalues(Gs_simplexs, names)
    print("\nFinished laplacian eigenvalues calcolation")

    print("\nLaplacian eigenvalues distances (firsts values):")
    lap_eigenvalues_distances_firsts_values = np.zeros((num_models, num_models))
    for i, lap_eigen_i in enumerate(laplacian_eigenvalues):
        for j, lap_eigen_j in enumerate(laplacian_eigenvalues):
            lap_eigenvalues_distances_firsts_values[i, j] = spectrum_distance(lap_eigen_i, lap_eigen_j, first_values_comparison=True)
    Lap_firsts_schema = make_schema(adj_eigenvalues_distances_firsts_values, names)
    dataFrame_dict["Lap_firsts_schema"] = Lap_firsts_schema

    print("\nLaplacian eigenvalues distances (lasts values):")
    lap_eigenvalues_distances_lasts_values = np.zeros((num_models, num_models))
    for i, lap_eigen_i in enumerate(laplacian_eigenvalues):
        for j, lap_eigen_j in enumerate(laplacian_eigenvalues):
            lap_eigenvalues_distances_lasts_values[i, j] = spectrum_distance(lap_eigen_i, lap_eigen_j, first_values_comparison=False)
    Lap_lasts_schema = make_schema(adj_eigenvalues_distances_firsts_values, names)
    dataFrame_dict["Lap_lasts_schema"] = Lap_lasts_schema

    write_dataFrame(dataFrame_dict, 'Validation', 'results/old_results/score_matching.xlsx', 1)
    write_dataFrame_ordered_by_name(dataFrame_dict, 'Validation', 'retrieval_matching.xlsx', 1, names)
    # write_dataFrame(dataFrame_time_dict, 'Validation', 'results/old_results/time_matching.xlsx', 1)

    # row_ind, col_ind, cost = graph_hungarian_algorithm(G2_simplex, G2_simplex)
    # print("Hungarian alg finished")
    # nx.graph_edit_distance(G1_simplex, G2_simplex, node_match=flat_nodes_match)
    # nx.draw_networkx(G1_simplex, node_size=1, with_labels=False, alpha=0.6, node_color='#0f0f0f')
