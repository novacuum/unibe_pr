from collections import Counter
import numpy
from pr.algo.knn.item import Item
from pr.collection.prioritylist import PriorityList


class KNNAlgo:
    def __init__(self, k: int, distance: str = 'euclidian'):
        self.k = k
        self.ground_truth = None
        self.distance = distance

    def set_ground_truth(self, ground_truth):
        self.ground_truth = ground_truth
        return self

    def classify(self, item):
        result = PriorityList(self.k)
        distance = self.__getattribute__(self.distance)

        for source in self.ground_truth:
            d = distance(source, item)
            if d == 0.0:
                return source.label

            result.put((d, source.label))

        if result.empty():
            return None

        grouped = Counter()
        while not result.empty():
            grouped[result.get()[1]] += 1

        return grouped.most_common()[0][0]

    def condense(self):
        appended = True
        working_truth = self.ground_truth[1:]
        k = self.k
        self.k = 1
        self.ground_truth = self.ground_truth[0:1]
        iterations = 1

        while appended:
            appended = False
            size_before = len(self.ground_truth)
            correct_classified = []

            for item in working_truth:
                if item.label != self.classify(item):
                    self.ground_truth.append(item)
                    appended = True
                else:
                    correct_classified.append(item)

            working_truth = correct_classified
            size = len(self.ground_truth)
            print('run condense round {} width {} items, where {} added'.format(iterations, size, size - size_before))
            iterations += 1

        print('finish condense with {} items'.format(len(self.ground_truth)))
        self.k = k
        return self

    @staticmethod
    def euclidian(source_image: Item, target_image: Item):
        return numpy.sqrt(numpy.sum(numpy.square(numpy.subtract(source_image.image, target_image.image))))
