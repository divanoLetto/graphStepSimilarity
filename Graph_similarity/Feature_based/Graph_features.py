import numpy as np
from scipy import stats
ALL = 3
TRANSITIVITY_ZERO = 1
import networkx as nx


def get_egonet (g):
    egonet = {k: list(g.neighbors(k)) for k in g.nodes()}
    return egonet

""" To find the degree of nodes in the given graph, Return value will be a dictionary {vertex_index : degree}
    e.g. {1:2 , 2:2 }
"""
def get_di(g):
    neighbor_size = {k: g.degree(k) for k in g.nodes()}
    return neighbor_size

""" To find the clustering index of the nodes in the given graph. Return value will be a dictionary {vertex_index: clustering_index}
"""
def get_ci(g):

    clustering_index = {}
    for k in g.nodes():
        neighbors_k = list(g.neighbors(k))
        if len(neighbors_k) > 0:
            clustering_index[k] = nx.average_clustering(G=g, nodes=neighbors_k)
        else:
            clustering_index[k] = 0
    #clustering_index = {k : g.transitivity_local_undirected(vertices=k,mode=TRANSITIVITY_ZERO) for k in g.nodes()}
    return clustering_index

""" To find the average number of two hop neighbors of all nodes in the given graph. Return value will be a dictionary {vertex_index: average_two_hop_neighbors}
"""
def get_dni(g):
    two_hop_neighbors = {}
    for key,value in get_egonet(g).items():
        avg_hop = np.mean([g.degree(k) for k in value])
        two_hop_neighbors[key] = avg_hop
    return two_hop_neighbors

""" To find the average clustering coefficient of all nodes in the given graph. Return value will be a dictionary {vertex_index: average_clustering coefficient}
"""
def get_cni(g):
    avg_ci = {}
    ci = get_ci(g)
    for key,value in get_egonet(g).items():
        temp = np.mean([ci[k] for k in value])
        avg_ci[key] = temp
    return avg_ci


def get_eegoi(g):
    egonet = get_egonet(g)
    eegoi = {}
    for vertex in g.nodes():
        sg = g.subgraph(egonet[vertex] + [vertex])
        egonet_es = [(k[0],k[1]) for k in sg.edges()]
        eegoi[vertex] = len(egonet_es)
    return eegoi

""" To find the number of edges going out from the egonet of each node in the given graph. Return value will be a dictionary {vertex_index: outgoing_edges_from_egonet}
"""
def get_eoegoi(g):
    egonet = get_egonet(g)
    eoegoi = {}
    for vertex in g.nodes():
        total_vs = [vertex]
        for k in egonet[vertex]:
            total_vs = total_vs + egonet[k] + [k]
        total_vs = list(set(total_vs))
        sg = g.subgraph(total_vs)
        total_es = [(k[0],k[1]) for k in sg.edges()]
        sg_egonet = g.subgraph(egonet[vertex] + [vertex])
        egonet_es = [(k[0],k[1]) for k in sg_egonet.edges()]
        eoegoi[vertex] = len(list(set(total_es) - set(egonet_es)))
    return eoegoi

""" To find the number of neighbors of the egonet of each node in the given graph. Return value will be a dictionary {vertex_index: neighbors_of_egonet}
"""
def get_negoi(g):
    egonet = get_egonet(g)
    negoi = {}
    for vertex in g.nodes():
        egonet_vs = [vertex] + egonet[vertex]
        total_vs = []
        for k in egonet[vertex]:
            total_vs = total_vs +egonet[k]
        total_vs = list(set(total_vs))
        total_vs = [i for i in total_vs if i not in egonet_vs]
        negoi[vertex] = len(total_vs)
    return negoi



def get_features(g):
    di = get_di(g)
    ci = get_ci(g)  # useless in tree
    dni = get_dni(g)
    cni = get_cni(g)  # useless in tree
    eego = get_eegoi(g)
    eoego = get_eoegoi(g)
    negoi = get_negoi(g)
    all_features = [(di[v],ci[v],dni[v],cni[v],eego[v],eoego[v],negoi[v]) for v in g.nodes()];

    return all_features


def get_signature(g):
    all_features = get_features(g)
    num_nodes = len(all_features)
    signature = []
    for k in range(0, 7):
        feat_agg = [all_features[i][k] for i in range(0, num_nodes)]
        mn = np.mean(feat_agg)
        md = np.median(feat_agg)
        std_dev = np.std(feat_agg)
        skw = stats.skew(feat_agg)
        krt = stats.kurtosis(feat_agg)
        signature = signature + [mn, md, std_dev, skw, krt]
    # del all_features;
    return signature