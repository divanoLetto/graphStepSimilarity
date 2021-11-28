import scipy


def canberra_distance(sign1, sign2):
    return abs(scipy.spatial.distance.canberra(sign1, sign2));