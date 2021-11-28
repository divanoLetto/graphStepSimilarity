import networkx as nx
import numpy as np
from pathlib import Path
import time

from utils import histogram_intersection


def calc_adj_eigenvalues(graphs, names):
    print("Calcolating adj eigenvalues")
    adj_eigenvalues = []
    for i, graph in enumerate(graphs):
        save_path = "C:/Users/Computer/PycharmProjects/graphStepSimilarity/matrix_saves/Adj_eigenvalues/" + names[i] + ".npy"
        save_vect = Path(save_path)

        if not save_vect.exists():
            start_time = time.time()
            adj_eig = nx.linalg.spectrum.adjacency_spectrum(graph, weight="weight") # todo va bene usare 1 come peso di connessione?
            np.save(save_path, adj_eig)
            adj_eigenvalues.append(adj_eig)
            print("   Time for " + names[i] + " model eigenvalues: " + str(time.time() - start_time)+ " seconds")
        else:
            adj_eig = np.load(save_path)
            adj_eigenvalues.append(adj_eig)

    return adj_eigenvalues


def calc_laplacian_eigenvalues(graphs, names):
    print("Calcolating laplacian eigenvalues")
    laplacian_eigenvalues = []
    for i, graph in enumerate(graphs):
        save_path = "C:/Users/Computer/PycharmProjects/graphStepSimilarity/matrix_saves/Lap_eigenvalues/" + names[i] + ".npy"
        save_vect = Path(save_path)

        if not save_vect.exists():
            start_time = time.time()
            lap_eig = nx.linalg.spectrum.adjacency_spectrum(graph, weight="weight") # todo va bene usare 1 come peso di connessione?
            np.save(save_path, lap_eig)
            laplacian_eigenvalues.append(lap_eig)
            print("   Time for " + names[i] + " model eigenvalues: " + str(time.time() - start_time)+ " seconds")
        else:
            lap_eig = np.load(save_path)
            laplacian_eigenvalues.append(lap_eig)

    return laplacian_eigenvalues


def calc_histogram_intersections(histograms):
    save_path = "C:/Users/Computer/PycharmProjects/graphStepSimilarity/matrix_saves/Histograms_intersections/histo_intersection.npy"
    save_vect = Path(save_path)
    if not save_vect.exists() or True:  # todo fix this
        time_start = time.time()
        histo_diff = np.zeros((len(histograms), len(histograms)))
        for i, h1 in enumerate(histograms):
            for j, h2 in enumerate(histograms):
                histo_diff[i, j] = histogram_intersection(h1, h2)
        np.save(save_path, histo_diff)
        print("   Time for histogram intersection: " + str(time.time() - time_start) + " seconds")
    else:
        histo_diff = np.load(save_path)
    return histo_diff


def isnumeric(string):
    if string.isnumeric():
        return True
    elif string[:-1].isnumeric() and string[-1]==".":
        return True
    return False


def my_float(string):
    if string[:-1].isnumeric() and string[-1]==".":
        return float(string[:-1])
    else:
        return float(string)


