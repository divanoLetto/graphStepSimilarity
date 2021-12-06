from Graphh.Graphh import Graphh
from Graphh.Node_utils import get_nodes_type_hystogramm
from Parser.Make_graph import *
from Parser.Make_step import make_step_from_graph
import os


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
    dataFrame_time_dict = {}
    file_names = [file_name1, file_name2,  file_name3, file_name4, file_name6, file_name7, file_name9, file_name10, file_name11, file_name12, file_name13]
    dump_file_name = ["dump_quad.step", "dump_quad_big.step",  "dump_rettangolo.step", "dump_quad_tri.step", "dump_piramid.step", file_name10, file_name11,file_name7, file_name8]  #, "dump_quad.step", "dump_quad_big.step", "dump_rettangolo.step", "dump_quad_tri.step", ]#"dump_cone.step"  , "dump_double_cone.step"]
    # file_names = dump_file_name

    names = [os.path.splitext(f)[0] for f in file_names]
    headers_datas = [(parse_file(f)) for f in file_names]

    histograms = [get_nodes_type_hystogramm(f[1]) for f in headers_datas]

    print("Realizing graphs")
    Gs_simplexs_direct = [make_graph_simplex_direct(f) for f in file_names]
    graphs_direct = []
    for id_name, file_name in enumerate(file_names):
        name = os.path.splitext(file_name)[0]
        extension = os.path.splitext(file_name)[1]
        g = Graphh(Gs_simplexs_direct[id_name], name, ext=extension)
        graphs_direct.append(g)

    for g in graphs_direct:
        g.print_composition()

    for g in graphs_direct:
        base_name = g.name
        # Path(base_model_path_dir+dir_name).mkdir(parents=True, exist_ok=True)
        full_graph = g.full_graph
        for j, part in enumerate(g.parts_graphs):
            graph = part
            part_name = base_name + "_" + g.parts_names[j]
            make_step_from_graph(g.name + g.extention, full_graph, graph, part_name, g.parts_conteiner)


if __name__ == "__main__":
    main()
