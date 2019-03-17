from collections import Counter
import numpy
from pr.algo.cluster.kmeans import KMeans, append_train_index
from pr.algo.cluster.validators.cindex import CIndex
from pr.algo.cluster.validators.davisbouldinindex import DavisBouldinIndex
from pr.algo.cluster.validators.dunnindex import DunnIndex


def main():
    # dump multi processing protection
    train_data_raw = numpy.loadtxt(open('../assets/knn/train.csv', "rb"), dtype='i', delimiter=",", skiprows=0)
    train_data = train_data_raw[:, 1:]
    train_data_labels = train_data_raw[:, 0]
    kList = [5, 7, 9, 10, 12, 15]

    for k in kList:
        k_mean = KMeans(k)
        k_mean.add_validator(DavisBouldinIndex())
        # k_mean.add_validator(CIndex())
        k_mean.add_validator(DunnIndex())
        k_mean.set_train_data(train_data)

        print('----- result for k={}'.format(k))
        k_mean.cluster()
        clusters = k_mean.create_clusters(append_train_index)
        for cluster in clusters:
            counter = Counter()
            for label_index in cluster:
                counter[train_data_labels[label_index]] += 1
            print('cluster with labels {}'.format(counter))


if __name__ == '__main__':
    main()
