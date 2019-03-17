from unittest import TestCase
import numpy
from pr.algo.cluster.kmeans import KMeans
from pr.algo.cluster.validators.cindex import CIndex

class TestCIndex(TestCase):
    def test_compute(self):
        kmean = KMeans(3, 1)
        kmean.centers = numpy.array([[10], [20], [30]])
        kmean.train_data = numpy.array([[9], [11], [18], [22], [27], [33]])
        clusters = [[numpy.array([9]), numpy.array([11])], [numpy.array([18]), numpy.array([22])], [numpy.array([27]), numpy.array([33])]]
        self.assertEqual(kmean.create_clusters(), clusters)
        index = CIndex()
        index.set_clusterer(kmean)
        self.assertEqual(0, index.compute(clusters))
