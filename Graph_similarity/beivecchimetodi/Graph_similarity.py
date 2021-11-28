import warnings
from scipy.optimize import minimize_scalar
import math


"""def matrix_cost_graphs(g1, g2):
    len1 = len(g1.nodes())
    len2 = len(g2.nodes())
    matrix_cost = np.zeros((len1, len2))  # TODO sparse matrix?
    for i, n1 in enumerate(g1.nodes):
        print(i)
        for j, n2 in enumerate(g2.nodes):
            cost = node_simarity(g1.nodes[n1], g2.nodes[n2])
            matrix_cost[i, j] = cost
            # print("   "+str(cost))
    return matrix_cost"""


"""def graph_hungarian_algorithm(g1, g2):
    print("Calcolating matrix cost")
    tmp_matrix = matrix_cost_graphs(g1, g2)
    save_path = "/matrix_saves/matrix.npy"
    # np.save(save_path, matrix)
    matrix = np.load(save_path)
    row_ind, col_ind = linear_sum_assignment(matrix)
    cost = 0
    different = []
    for i in range(len(row_ind)):
        price_i = matrix[row_ind[i], col_ind[i]]
        cost += price_i
        if price_i != 0:
            different.append((row_ind[i], col_ind[i], price_i))
    return row_ind, col_ind, cost"""

#  Distanza spettrale laplaciana: non richiede la corrispondenza dei nodi. Si basa solo sul Laplaciano che è dato dalle connessioni.
def spectrum_distance(eigens_1, eigens_2, first_values_comparison=True):
    quad_distance = 0
    min_len = min(len(eigens_1), len(eigens_2))
    # TODO come pesi della matrice di adieacenza va bene 1?

    if first_values_comparison:

        for i in range(min_len):
            delta_lambda = (eigens_1[i] - eigens_2[i])
            quad_distance += delta_lambda * delta_lambda
    else:
        for i in range(min_len, 0, -1):
            delta_lambda = (eigens_1[-i] - eigens_2[-i])
            quad_distance += delta_lambda * delta_lambda
    warnings.simplefilter("ignore")
    spectrum_distance = math.sqrt(quad_distance)
    return spectrum_distance


def gaussian_mixture(x, eigens):
    sum = 0
    n = len(eigens)
    twice_delta_quad = 0.015 * 0.015 * 2
    for i in range(n):
        x_eigen_diff = abs(x - eigens[i])
        numerator = math.exp(- x_eigen_diff*x_eigen_diff / twice_delta_quad)
        denominator = math.sqrt(math.pi*twice_delta_quad)
        sum += numerator / denominator
    return sum


def sqr_gaussian_func_j(j, i, eigens1, eigens2):
    theta1 = gaussian_mixture(i, eigens1)
    theta2 = gaussian_mixture(j, eigens2)
    quad_diff = (theta1-theta2)*(theta1-theta2)
    res = math.sqrt(quad_diff + (i-j)*(i-j))
    return res


def sqr_gaussian_func_i(i, j, eigens1, eigens2):
    theta1 = gaussian_mixture(i, eigens1)
    theta2 = gaussian_mixture(j, eigens2)
    quad_diff = (theta1-theta2)*(theta1-theta2)
    res = math.sqrt(quad_diff + (i-j)*(i-j))
    return res


#  Distanza spettrale laplaciana normalizzata: può essere applicata a grafi di dimensione diversa
def laplacian_spectrum_normalized_distance(eigen1, eigen2):
    n = 2000
    distance = 0
    for i in range(n):
        fact1 = minimize_scalar(sqr_gaussian_func_j, args=(i, eigen1, eigen2))
        fact2 = minimize_scalar(sqr_gaussian_func_i, args=(i, eigen1, eigen2))
        distance += (fact1.fun + fact2.fun)
    distance = distance/(n+1)
    return distance


