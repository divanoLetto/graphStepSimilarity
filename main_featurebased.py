from Graph_similarity.Feature_based.Distance import canberra_distance
from Graph_similarity.Feature_based.Graph_features import get_signature
from Graphh.Graphh import Graphh
from Nodes.Node_utils import get_nodes_type_hystogramm, get_num_neighbor_for_node_type, get_composed_node_types
from Parser.Make_graph import *
from Printing_and_plotting.Printing import write_dataFrame, write_dataFrame_ordered_by_name, write_dataFrame_by_images
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

    names = [os.path.splitext(f)[0] for f in file_names]
    headers_datas = [(parse_file(f)) for f in file_names]

    histograms = [get_nodes_type_hystogramm(f[1]) for f in headers_datas]

    print("Realizing graphs")
    Gs_simplexs_direct = [make_graph_simplex_direct(f) for f in file_names]
    graphs_direct = []
    for id_name, name in enumerate(names):
        g = Graphh(Gs_simplexs_direct[id_name], name)
        graphs_direct.append(g)

    for g in graphs_direct:
        g.print_composition()

    full_graph_signatures = {k.name: get_signature(k.full_graph) for k in graphs_direct}

    full_signatures_values = np.zeros((len(names), len(names)))
    for i, (key1, sign1) in enumerate(full_graph_signatures.items()):
        for j, (key2, sign2) in enumerate(full_graph_signatures.items()):
            dist = canberra_distance(sign1, sign2)
            full_signatures_values[i, j] = dist

    full_signatures_schema = make_schema(full_signatures_values, names)
    dataFrame_dict["full_signatures_schema"] = full_signatures_schema

    list_parts = []
    for i, gh_i in enumerate(graphs_direct):
        for j, part in enumerate(gh_i.parts_graphs):
            list_parts.append((gh_i.name+"_"+gh_i.parts_names[j], part))
    parts_signatures_values = np.zeros((len(list_parts), len(list_parts)))
    for i, (gh_i_name, gh_i) in enumerate(list_parts):
        for j, (gh_j_name, gh_j) in enumerate(list_parts):
            s1 = get_signature(gh_i)
            s2 = get_signature(gh_j)
            dist = canberra_distance(s1, s2)
            parts_signatures_values[i, j] = dist
    part_names = []
    for (n, _) in list_parts:
        if n not in part_names:
            part_names.append(n)
    parts_signatures_schema = make_schema(parts_signatures_values, part_names)
    dataFrame_dict_parts["parts_signatures_schema"] = parts_signatures_schema

    base_path = "C:/Users/Computer/PycharmProjects/graphStepSimilarity/results/features_based/"
    high_max = [False]
    write_dataFrame(df_dict=dataFrame_dict, file_name='score_matching.xlsx',  high_max=high_max, base_path=base_path)
    # write_dataFrame(df_dict=dataFrame_time_dict, file_name='time_matching.xlsx',  high_max=high_max, base_path=base_path)
    image_dir_path = "C:/Users/Computer/PycharmProjects/graphStepSimilarity/images/models_images/"
    write_dataFrame_by_images(df_dict=dataFrame_dict, file_name='retrieval_img_matching.xlsx',  names=names, by_min=high_max, base_path=base_path, image_dir_path=image_dir_path)

    image_dir_path = "C:/Users/Computer/PycharmProjects/graphStepSimilarity/images/models_images/parts/"
    high_max = [False]
    write_dataFrame(df_dict=dataFrame_dict_parts, file_name='parts_score_matching.xlsx',  high_max=high_max, base_path=base_path)
    write_dataFrame_by_images(df_dict=dataFrame_dict_parts, file_name='parts_retrieval_img_matching.xlsx', names=part_names, by_min=high_max, base_path=base_path, image_dir_path=image_dir_path, scale_x=0.03, scale_y=0.03)


if __name__ == "__main__":
    main()

