import numpy
import scipy.special
from pr.algo.cluster.validators.validator import Validator


class CIndex(Validator):
    def get_name(self):
        return 'C-Index'

    def compute(self, clusters):
        alpha = 0
        sigma = 0
        distances_over_all = []

        for k in self.clusterer.k_iter:
            cluster = clusters[k]

            size = len(cluster)
            pairs = int(scipy.special.binom(size, 2))
            alpha += pairs
            print('c-index @{} pairs {} for size {}'.format(k, pairs, size))
            distances = numpy.zeros(pairs)
            for i in range(size-1):
                inner_size = size - i
                distances[i*inner_size:((i+1)*inner_size)-1] = numpy.linalg.norm(cluster[i+1:size] - cluster[i], axis=1)

            sigma += numpy.sum(distances)
            distances_over_all = numpy.concatenate((distances_over_all, distances))

            clusters_2 = clusters[k+1:self.clusterer.k]
            clusters_2 = [item for sublist in clusters_2 for item in sublist]

            for i in range(size):
                distances_over_all = numpy.concatenate((distances_over_all, numpy.linalg.norm(clusters_2 - cluster[i], axis=1)))

            print('run k {} round, computed distances: {}'.format(k, len(distances_over_all)))

        if alpha >= len(distances_over_all):
            # because min = max and therefore the denominator is zero
           return 0
        else:
            min = numpy.sum(numpy.partition(distances_over_all, alpha)[0:alpha])
            max = numpy.sum(numpy.partition(distances_over_all, -alpha)[-alpha:])

        return (sigma - min) / (max - min)
