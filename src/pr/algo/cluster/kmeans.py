import math, numpy


def append_train_data(cluster, train_item, train_index):
    cluster.append(train_item)


def append_train_index(cluster, train_item, train_index):
    cluster.append(train_index)


class KMeans:
    def __init__(self, k, max_iterations=1000):
        self.train_data = None
        self.centers = None
        self.k = k
        self.k_iter = range(k)
        self.max_iterations = max_iterations
        self.validators = []

    def set_train_data(self, train_data):
        self.train_data = train_data

    def add_validator(self, validator):
        validator.set_clusterer(self)
        self.validators.append(validator)

    def cluster(self):
        self.centers = self.train_data[numpy.random.choice(self.train_data.shape[0], (self.k,), replace=False)]
        iterations = 0
        clusters = None

        while self.max_iterations > iterations:
            clusters = self.create_clusters()

            new_centers = list()
            for cluster in clusters:
                new_centers.append(numpy.mean(cluster, axis=0))
            new_centers = numpy.array(new_centers)

            if numpy.linalg.norm(self.centers - new_centers) < 1:
                break

            iterations += 1
            self.centers = new_centers

        if self.max_iterations == iterations:
            print('clustering was limited by max_iterations {}'.format(iterations))
        else:
            print('clustering needed {} iterations'.format(iterations))

        if clusters is not None:
            for validator in self.validators:
                result = validator.compute(clusters)
                print('Result for {} is {}'.format(validator.get_name(), result))

    def create_clusters(self, append=append_train_data):
        clusters = [list() for i in self.k_iter]

        for train_index, train_item in enumerate(self.train_data):
            min_distance = math.inf
            min_distance_index = 0

            for index, center in enumerate(self.centers):
                distance = numpy.linalg.norm(center - train_item)

                if min_distance > distance:
                    min_distance = distance
                    min_distance_index = index
            append(clusters[min_distance_index], train_item, train_index)
        return clusters
