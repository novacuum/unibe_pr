from unittest import TestCase
import numpy
from pr.algo.cluster.kmeans import KMeans
from pr.algo.cluster.validators.davisbouldinindex import DavisBouldinIndex


class TestDavisBouldinIndex(TestCase):
    def test_compute(self):
        kmean = KMeans(3, 1)
        kmean.centers = numpy.array([[10], [20], [30]])
        kmean.train_data = numpy.array([[8], [12], [18], [22], [28], [32]])
        self.assertEqual(kmean.create_clusters(), [[numpy.array([8]), numpy.array([12])], [numpy.array([18]), numpy.array([22])], [numpy.array([28]), numpy.array([32])]])
        index = DavisBouldinIndex()
        index.set_clusterer(kmean)
        self.assertEqual(0.2135378629581951, index.compute())
