import numpy


class Item:
    def __init__(self, label, image):
        self.label = label
        self.image = image

    @staticmethod
    def from_csv(path: str = "../assets/knn/train.csv"):
        data = numpy.loadtxt(open(path, "rb"), dtype='i', delimiter=",", skiprows=0)
        result = []

        for row in data:
            result.append(Item(int(row[0]), row[1:]))

        print('successfully loaded {} lines as items'.format(len(result)))
        return result
