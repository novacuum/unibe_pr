from pr.algo.cluster.kmeans import KMeans


class Validator:
    def __init__(self):
        self.clusterer = KMeans(1)

    def get_name(self):
        return 'wtf'

    def set_clusterer(self, clusterer: KMeans):
        self.clusterer = clusterer

    def compute(self, clusters):
        pass
