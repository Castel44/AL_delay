import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import LocalOutlierFactor


class BufferBase(object):
    def __init__(self):
        super(BufferBase, self).__init__()

        # self.values = []
        # self.logits = []

    def check_data(self):
        pass

    def add(self):
        # call subroutine add based on dimension of input. Has or not logit, etc etc
        pass

    def add_(self):
        pass


class IDBuffer(BufferBase):
    # TODO: cluster statistics
    def __init__(self, data, labels, input_data, n_class, k=20):
        super(IDBuffer, self).__init__()
        self.n_class = n_class
        # ID data buffer
        # TODO: add original data for re-training
        self.data_class = dict.fromkeys([x for x in range(self.n_class)])
        self.data_dist = dict.fromkeys([x for x in range(self.n_class)])
        self.data_input = dict.fromkeys([x for x in range(self.n_class)])
        # Reference density measures
        self.q_dist = dict.fromkeys([x for x in range(self.n_class)])

        # Init the buffer
        self.init(data, input_data, labels)
        self.k = k

    def init(self, data, input_data, labels):
        for i in range(self.n_class):
            self.data_class[i] = data[labels == i]
            self.data_input[i] = input_data[labels == i]
            self.q_dist[i] = dict.fromkeys(('low', '25', 'mid', '75', 'high'))
        return self

    def update(self, data, label, input_data):
        """
        Concatenate single sample to the data dictionary
        """
        # TODO: update with input data
        self.data_class[label] = np.concatenate((self.data_class[label], data[np.newaxis, :]), axis=0)
        self.data_input[label] = np.concatenate((self.data_input[label], input_data[np.newaxis, :]), axis=0)
        return self

    def compress(self):
        # TODO: prune some data
        pass

    def get_density_measure(self, low=0.05, mid=0.5, high=0.95):
        irq = dict()
        for i in range(self.n_class):
            data = self.data_class[i]
            pairwise_distances = cdist(data, data)
            kdist = np.take_along_axis(pairwise_distances, np.argpartition(pairwise_distances, self.k)[:, 1:self.k + 1],
                                       axis=1)
            dist_ = np.median(kdist, axis=1)
            self.data_dist[i] = dist_

            # Add IRQ
            self.q_dist[i]['low'], self.q_dist[i]['25'], self.q_dist[i]['mid'], self.q_dist[i]['75'], self.q_dist[i][
                'high'] = np.quantile(dist_, (low, 0.25, mid, 0.75, high))
            irq[i] = self.q_dist[i]['75'] - self.q_dist[i]['25']
        return self.q_dist, irq

    def get_data(self):
        data = np.concatenate([self.data_class[i] for i in range(self.n_class)])
        labels = np.array([i for i in range(self.n_class) for _ in self.data_class[i]])
        return data, labels


class OODBuffer(BufferBase):
    @staticmethod
    def knn_dist(x, reference, k=10):
        pairwise_distances = cdist(x[np.newaxis, :], reference)
        knn_idx = np.argpartition(pairwise_distances, k)[:, k]  # idx of closest neighbors reference data
        kdist = pairwise_distances.T[knn_idx].squeeze()
        dist = np.median(kdist)
        return dist, pairwise_distances

    def __init__(self):
        super(OODBuffer, self).__init__()

        self.data = []
        self.orig_data = []
        self.proba = []
        self.ood_label = []
        self.label = []
        self.idx = []
        self.pseudo_label = []

    def update(self, x, p, ood_label, idx, orig):
        self.data.append(x)
        self.proba.append(p)
        label = np.argmax(p)
        self.label.append(label)
        self.ood_label.append(ood_label)
        self.pseudo_label.append(label if ood_label == 0 or ood_label == 1 else -1)
        self.idx.append(idx)
        self.orig_data.append(orig)
        self.check_len()
        return self

    def check_len(self):
        assert len(self.data) == len(self.proba) == len(self.ood_label) == len(self.idx)

    def analyse(self, k):
        sample = np.asarray(self.data[-1])[np.newaxis, :]
        timestamp = self.idx[-1]
        data = np.asarray(self.data[:-1])
        labels = np.asarray(self.pseudo_label[:-1])
        timestamps = timestamp - np.asarray(self.idx[:-1])

        # Transient
        if len(data) < k:
            print('Not too much elements in buffer. Skipping...')
            return self

        # LOF to identify outliers
        clf = LocalOutlierFactor(n_neighbors=k, novelty=True)
        clf.fit(data)
        lof_label = clf.predict(sample)

        if lof_label == -1:
            # Sample is outlier
            print(f'Sample {timestamp} is Outlier. lof_score = {clf.score_samples(sample)}')
            return self

        # TODO: only use recent samples to get neighbors
        # TODO: solve new/class problem. Assign to outlier only. Not use all knn but just the closest.
        try:
            knn_dist, knn_idx = clf.kneighbors(sample, n_neighbors=k*2, return_distance=True)
        except:
            return self
        knn_t = timestamps[knn_idx]
        knn_idx = knn_idx[knn_t < 500]
        knn_labels = labels[knn_idx]

        value, count = np.unique(knn_labels, return_counts=True)
        if np.max(count) > k:
            frequent_value = value[np.argmax(count)]
            if frequent_value == -1:
                print('Frequent outlier')
                return self
            for i in np.append(knn_idx, -1):
                self.pseudo_label[i] = frequent_value
            return self
        else:
            return self

        if False:
            import matplotlib.pyplot as plt
            from src.utils.training_helper_v3 import MplColorHelper
            cmap = 'jet'

            ood = np.asarray(self.ood_label[:-1])
            sample = np.asarray(self.data[-1])[np.newaxis, :]
            data = np.asarray(self.data[:-1])
            labels = np.asarray(self.pseudo_label[:-1])

            COL = MplColorHelper(cmap, 0, len(np.unique(labels)))
            c = {0: 'green', 1: 'gold', 2: 'red'}
            plt.figure()
            plt.scatter(*data.T, c=COL.get_rgb(labels), alpha=0.5, edgecolor=[c[i] for i in ood])
            plt.scatter(*sample.T, c='red', marker='*', s=350)

    def statistics(self, min_size = 200):
        #n = len(self.data)
        idx = self.idx[-1]  # should be same as n-1
        ood = np.asarray(self.ood_label)
        labels = np.asarray(self.pseudo_label)
        #proba = np.asarray(self.proba)
        pred_labels = np.asarray(self.label)
        #classes = len(np.unique(pred_labels))
        # H = entropy(proba, axis=1, base=classes)

        outliers = np.sum(labels == -1)
        val, count = np.unique(ood, return_counts=True)
        s = 'ID/OOD stat: \t'  # 0: inliner, 1: almost outliner, 2: out of distribution
        for v, c in zip(val, count):
            s += f'{v} -> {c} \t'
        print(s)

        # OOD analysis. Sample for re-training.
        val, count = np.unique(labels[ood == 2], return_counts=True)
        s = 'OOD stat: \t'
        for v, c in zip(val, count):
            s += f'class:{v} -> {c} \t'
        print(s)

        # Non-outliers
        count_ = count[1:]
        end_flag = (count_ > min_size).all()
        return end_flag

        """        
        id = labels != -1
        import matplotlib.pyplot as plt
        data = np.asarray(self.data)
        plt.figure()
        plt.scatter(*data[~id].T, c='black', label='outliers', alpha=0.5)
        plt.scatter(*data[id].T, alpha=0.5)

        plt.figure()
        obj = plt.scatter(*data.T, c=H, cmap=plt.get_cmap('seismic'), alpha=0.5)
        plt.scatter(*data[most_info].T, c=H[most_info], marker='*', s=500, alpha=0.5)
        plt.colorbar(obj)

        plt.figure()
        plt.scatter(*data[id].T, c=pred_labels[id])

        plt.figure()
        plt.scatter(*data[id].T, c=labels[id])"""


# TODO: object to track class statistics
# TO detect label drift?!
class ClassMicroCluster(object):
    def __init__(self):
        self.n = 0
        self.linsum = 0
        self.quadsum = 0
