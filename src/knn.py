from pr.algo.knn.item import Item
from pr.algo.knnalgo import KNNAlgo

train_set = Item.from_csv('../assets/knn/train.csv')
test_set = Item.from_csv('../assets/knn/test.csv')

knn = KNNAlgo(1)
knn.set_ground_truth(train_set)
knn.condense()
success = 0
error = 0

for test in test_set:
    label = knn.classify(test)

    if label == test.label:
        success += 1
    else:
        error += 1

print('success: {}\nerror: {}\nerror rate: {}'.format(success, error, error/(success+error)))
