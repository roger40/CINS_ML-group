import numpy as np
from sklearn import metrics


class Choose(object):
    def __init__(self, data):
        """
        initialize data
        :param data: N X M, where N is the number of items, M is the dimensionality of feature
        """
        self.data = np.array(data)

    def distance(self, x_i, x_j):
        """
        compute the distance between items i, j with their features x_i, x_j
        :param x_i: np.array, 1 X dimensionality
        :param x_j:np.array, 1 X dimensionality
        :return: distance
        """
        # ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
        dist = metrics.pairwise_distances([x_i], [x_j], metric='cosine')
        return dist[0, 0]

    def centre(self, cluster):
        """
        compute the centre of cluster
        :param cluster: np.array or list, including items index
        :return: the feature of centre items
        """
        size = len(cluster) * 1.0
        cen = np.zeros_like(self.data[0])
        for item in cluster:
            cen = cen + self.data[item]
        return cen / size

    def avg_cen(self, cluster):
        size = len(cluster) * 1.0
        cen = self.centre(cluster)
        dist = 0
        for item in cluster:
            dist += self.distance(self.data[item], cen)
        return dist / size, cen

    def avg_cen0(self, cluster):
        size = len(cluster)
        cen = np.zeros_like(self.data[0])
        dists = 0
        for i in range(size):
            x_i = cluster[i]
            cen = cen + self.data[x_i]
            for j in range(i + 1, size):
                x_j = cluster[j]
                dist = self.distance(self.data[x_i], self.data[x_j])
                dists += dist
        avg = 2.0 * dists / (size * (size - 1.0))
        return avg, cen

    def avg_diam(self, cluster):
        """
        compute the average distance and maximum distance among items in this cluster
        :param cluster: np.array or list, including items index
        :return: avg, mean distance, and diam, the maximum distance between items among cluster
        """
        dists = 0
        size = len(cluster)
        diam = 0
        for i in range(size):
            x_i = cluster[i]
            for j in range(i + 1, size):
                x_j = cluster[j]
                dist = self.distance(self.data[x_i], self.data[x_j])
                dists += dist
                diam = max(diam, dist)
        avg = 2.0 * dists / (size * (size - 1.0))
        return avg, diam

    def min_cen(self, cluster1, cluster2):
        """
        compute the minimum distance between item in cluster1 and item in cluster2,
        and the distance between the centres of clusters
        :param cluster1: np.array or list, including items index
        :param cluster2: np.array or list, including items index
        :return: dmin, cen
        """
        dmin = np.inf
        cen1 = self.centre(cluster1)
        cen2 = self.centre(cluster2)
        cen = self.distance(cen1, cen2)
        for i in cluster1:
            for j in cluster2:
                dist = self.distance(self.data[i], self.data[j])
                dmin = min(dmin, dist)
        return dmin, cen

    def DB_index(self, dynamic_comms):
        comms = [k for k in dynamic_comms.keys()]
        num = len(comms) * 1.0
        db = 0
        for i in comms:
            dmax = 0
            cluster1 = dynamic_comms[i]
            for j in comms:
                if i == j:
                    continue
                cluster2 = dynamic_comms[j]

                # avg1, _ = self.avg_diam(cluster1)
                # avg2, _ = self.avg_diam(cluster2)
                # dmin, cen = self.min_cen(cluster1, cluster2)
                # dist = (avg1+avg2)/(cen*1.0)
                avg1, cen1 = self.avg_cen(cluster1)
                avg2, cen2 = self.avg_cen(cluster2)
                dist = (avg1 + avg2) / (1.0 * self.distance(cen1, cen2))
                dmax = max(dmax, dist)
            db += dmax
        return db / num
