def histo_from_graph(graph):
    histo = {}
    for node_id in graph:
        node = graph.nodes[node_id]
        key = node["type"]
        if key not in histo.keys():
            histo[key] = 0
        histo[key] += 1
    return histo