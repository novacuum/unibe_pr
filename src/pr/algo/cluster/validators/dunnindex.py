import math
from joblib import Parallel, delayed
import numpy
from pr.algo.cluster.validators.validator import Validator


class DunnIndex(Validator):
    def __init__(self):
        super().__init__()
        self.min_batch_size = 50
        self.num_processes = 8

    def get_name(self):
        return 'Dunn-Index'

    def compute(self, clusters):
        min_between_cluster_distance = math.inf
        max_cluster_diameter = -math.inf
        delayed_compute_min_between_cluster_distance = delayed(compute_min_between_cluster_distance)
        parallel = Parallel(n_jobs=self.num_processes, max_nbytes='1M', verbose=True, backend='threading')

        for k in self.clusterer.k_iter:
            cluster = clusters[k]

            size = len(cluster)
            print('dunn index @{} with size {}'.format(k, size))

            for i in range(size-1):
                max_cluster_diameter = max(max_cluster_diameter, numpy.max(numpy.linalg.norm(cluster[i+1:size] - cluster[i], axis=1)))

            # for the lust cluster the distances between clusters are already computed
            if k+1 == self.clusterer.k:
                break

            clusters_flatten = clusters[k+1:self.clusterer.k]
            clusters_flatten = [item for sublist in clusters_flatten for item in sublist]

            patch_size = max(math.ceil(size/self.num_processes), self.min_batch_size)
            delayed_process = []
            for i in range(math.ceil(size/patch_size)):
                #delayed_process.append(delayed_compute_min_between_cluster_distance(0))
                delayed_process.append(delayed_compute_min_between_cluster_distance(range(i * patch_size, min((i + 1) * patch_size, size)), min_between_cluster_distance, clusters_flatten, cluster))

            min_between_cluster_distance = numpy.min(parallel(delayed_process))
            print('run k {} round, min_between_cluster_distance: {}, max_cluster_diameter: {}'.format(k, min_between_cluster_distance, max_cluster_diameter))

        return min_between_cluster_distance/max_cluster_diameter


def compute_min_between_cluster_distance(i_iter, min_between_cluster_distance, clusters_flatten, cluster):
    for i in i_iter:
        min_between_cluster_distance = min(min_between_cluster_distance, numpy.min(numpy.linalg.norm(clusters_flatten - cluster[i], axis=1)))
    return min_between_cluster_distance