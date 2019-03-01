import numpy

from pr.algo.knn.item import Item
from pr.algo.knnalgo import KNNAlgo

train_set = Item.from_csv('../assets/knn/train.csv')
test_set = Item.from_csv('../assets/knn/test.csv')
kList = [1, 3, 5, 10, 15]

knn = KNNAlgo('manhattan')
knn.set_ground_truth(train_set)
knn.condense()

maxK = numpy.max(kList)
for test in test_set:
    knn.compute_distance(test, maxK)

for k in kList:
    success = 0
    error = 0

    for test in test_set:
        label = knn.classify_by_pre_computed_distance(test, k)

        if label == test.label:
            success += 1
        else:
            error += 1

    print('@{} success: {}\terror: {}\terror rate: {}'.format(k, success, error, error / (success + error)))
