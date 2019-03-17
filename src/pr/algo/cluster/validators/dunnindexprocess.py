import numpy


def compute_min_between_cluster_distance(min_between_cluster_distance):
    return min_between_cluster_distance

def compute_min_between_cluster_distance2(i_iter, min_between_cluster_distance, clusters_flatten, cluster):
    for i in i_iter:
        min_between_cluster_distance = min(min_between_cluster_distance, numpy.min(numpy.linalg.norm(clusters_flatten - cluster[i], axis=1)))
    return min_between_cluster_distance