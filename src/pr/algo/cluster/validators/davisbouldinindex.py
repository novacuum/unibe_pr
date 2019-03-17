import numpy
from pr.algo.cluster.validators.validator import Validator


class DavisBouldinIndex(Validator):
    def get_name(self):
        return 'Davis Bouldin Index'

    def compute(self, clusters):
        centers = self.clusterer.centers
        diameters = numpy.empty(self.clusterer.k, dtype=float)

        for k in self.clusterer.k_iter:
            cluster = clusters[k]
            center = centers[k]
            diameters[k] = numpy.sum(numpy.linalg.norm(cluster - center, axis=1)) / len(cluster)

        r_list = numpy.zeros(self.clusterer.k, dtype=float)
        for k in self.clusterer.k_iter:
            center = centers[k]
            r = numpy.add(numpy.delete(diameters, k, 0), diameters[k]) / numpy.linalg.norm(numpy.delete(centers, k, 0) - center, axis=1)
            r_list[k] = numpy.max(r)

        return numpy.mean(r_list)
