import numpy


class Item:
    def __init__(self, label, image):
        self.label = label
        self.image = image
        self.distance = None

    @staticmethod
    def from_csv(path: str = "../assets/knn/train.csv"):
        data = numpy.loadtxt(open(path, "rb"), dtype='i', delimiter=",", skiprows=0)
        result = []

        for row in data:
            result.append(Item(int(row[0]), row[1:]))

        print('successfully loaded {} lines as items'.format(len(result)))
        return result

    @staticmethod
    def to_csv(items, path: str = "../temp/knn/test.csv"):
        data = []
        for item in items:
            data.append(numpy.concatenate([[item.label], item.image]).astype(int))

        numpy.savetxt(path, numpy.array(data).astype(int), fmt="%i", delimiter=",")
