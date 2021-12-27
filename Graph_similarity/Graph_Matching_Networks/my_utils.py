import numpy as np
import collections


def get_next_batch(batch_size, training_set, one_hot_dict):
    """Yields batches of pair data."""
    i = 0
    while True:
        batch_graphs = []
        batch_labels = []
        for _ in range(batch_size):
            if i >= len(training_set):
                i = 0
            g1, g2, sim = training_set[i]
            batch_graphs.append((g1, g2))
            batch_labels.append(sim)
            i += 1

        packed_graphs = my_pack_batch(batch_graphs, one_hot_dict)
        labels = np.array(batch_labels, dtype=np.int32)
        yield packed_graphs, labels


def my_pack_batch(graphs, one_hot_dict):
    Graphs = []
    for graph in graphs:
        for inergraph in graph:
            Graphs.append(inergraph)
    graphs = Graphs
    from_idx = []
    to_idx = []
    graph_idx = []
    node_features = []

    n_total_nodes = 0
    n_total_edges = 0
    for i, gh in enumerate(graphs):
        g = gh.get_full_graph()
        n_nodes = g.number_of_nodes()
        n_edges = g.number_of_edges()
        idx_2_int = {}
        int_2_idx = {}
        for j, idx in enumerate(g.nodes()):
            idx_2_int[idx] = j
            int_2_idx[j] = idx
        fixed_edges = [(idx_2_int[e[0]], idx_2_int[e[1]]) for e in g.edges()]
        edges = np.array(fixed_edges, dtype=np.int32)
        # shift the node indices for the edges
        from_idx.append(edges[:, 0] + n_total_nodes)
        to_idx.append(edges[:, 1] + n_total_nodes)
        graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)

        for n in g.nodes():
            type = g.nodes[n]["type"]
            node_features.append([1.0 if one_hot_dict[type] == i else 0.0 for i in one_hot_dict.values()])

        n_total_nodes += n_nodes
        n_total_edges += n_edges

    GraphData = collections.namedtuple('GraphData', [
        'from_idx',
        'to_idx',
        'node_features',
        'edge_features',
        'graph_idx',
        'n_graphs'])

    return GraphData(
        from_idx=np.concatenate(from_idx, axis=0),
        to_idx=np.concatenate(to_idx, axis=0),
        # this task only cares about the structures, the graphs have no features.
        # setting higher dimension of ones to confirm code functioning
        # with high dimensional features.
        node_features=np.ones((n_total_nodes, 8), dtype=np.float32),
        edge_features=np.ones((n_total_edges, 4), dtype=np.float32),
        graph_idx=np.concatenate(graph_idx, axis=0),
        n_graphs=len(graphs),
    )