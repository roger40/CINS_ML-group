import numpy as np
import networkx as nx
import scipy.io as scio
from sklearn import cluster as Cluster
from collections import defaultdict
import matplotlib
# matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt
from matplotlib import colors
import os

from chooseK import Choose
import PCA_eig as PCA

class DetectStat(object):
    def __init__(self, file, cluster_num, ftype='sparse'):
        if ftype=='sparse':
            sample = scio.mmread(file).todense()
        elif ftype == 'scipy':
            sample = scio.mmread(file)
        elif ftype == 'numpy':
            sample = np.mat(np.loadtxt(file))
        x, y = sample.shape
        sample.shape = 1, x*y
        print(self.detect_A(sample, cluster_num))
        print(self.detect_B(sample, cluster_num))

    def detect_A(self, sample_mat, cluster_num):
        represenation, _, _ = self.representation(sample_mat, cluster_num)
        represenation = np.array(represenation.T)
        cluster = Cluster.KMeans(n_clusters=cluster_num).fit(represenation)
        labels_list = cluster.labels_
        comms = defaultdict(list)
        for n, c in enumerate(labels_list):
            comms[c].append(str(n)+'-t0')
        return comms

    def detect_B(self, sample_mat, cluster_num):
        represenation, _, _ = self.representation(sample_mat, cluster_num)
        represenation = np.array(represenation.T)
        cluster = Cluster.SpectralClustering(n_clusters=cluster_num, gamma=0.1, affinity='rbf').fit(represenation)
        labels_list = cluster.labels_
        comms = defaultdict(list)
        for n, c in enumerate(labels_list):
            comms[c].append(str(n) + '-t0')
        return comms

    def representation(self, sample, dim, n=None):
        """
        reconstruction of sample to represent its nodes
        :return: representation, map_space, adjacent matrix with average
        """
        reconstructed_mat, eig_vector, adj_mat = PCA.pca1_6(sample, dim, n=n)
        self.dim, _ = eig_vector.shape
        return reconstructed_mat, eig_vector, adj_mat

    def representation_map(self, sample, eig_vector):
        """
        map and reconstruct sample with eig_vector
        :return: representation
        """
        reconstructed_mat, f = PCA.pca2(sample, eig_vector)
        x, y = reconstructed_mat.shape
        copy_mat = np.zeros((x, y))
        f = np.array(f)
        for i in range(x):
            for j in range(y):
                if isinstance(reconstructed_mat[i, j], complex):
                    copy_mat[i, j] = reconstructed_mat[i, j].real
                else:
                    copy_mat[i, j] = reconstructed_mat[i, j]
        return copy_mat, f

    def db_index(self, dynmaic, time_node, representation):
        choose = Choose(representation)
        num = len(time_node)
        tn_dict = dict([(time_node[i], i) for i in range(num)])
        dynmaic_comms = {}
        for c in dynmaic.keys():
            dynmaic_comms[c] = []
            for t_n in dynmaic[c]:
                dynmaic_comms[c].append(tn_dict[t_n])
        db = choose.DB_index(dynmaic_comms)
        return db

class DetectDyn(DetectStat):
    def __init__(self, file, data, cluster_num, gml=True, time=None, t_long=1, detect='alone',
                 method='kmeans', random_state=0, dim=None, modify=False):
        """
        dynamic community detection and save them with dictionary
        :param file: path of file
        :param data: the name of data
        :param cluster_num: the number of clusters
        :param gml: whether the file is .gml
        :param time: the name for time[0] to time[-1]
        :param detect: the representation method. e.g. "alone" is the alone PCA,
                        "time" is PCA of compression all snapshots, "join" is PCA of the X*X^T
        :param method: the cluster method
        """
        self.random_state = random_state
        self.data = data
        self.dim = dim
        if time is not None:
            # time = [0, 4]
            self.time = time
            path = file
            self.path = path
            if gml:
                graphs = {}
                self.nodes = {}
                i = 0
                s = 0
                for t in range(time[0], time[-1]):
                    if data[0] == 'r':
                        file = './gml/orignal/RC_2010-09_' + str(t) + '.gml'
                    else:
                        file = './gml/' + str(t-time[0]) + '.gml'
                    try:
                        graph = nx.read_gml(os.path.join(path, file))
                    except:
                        continue
                    # nx.draw(graph)
                    # plt.show()
                    for node in graph.nodes():
                        if node not in self.nodes.keys():
                            self.nodes[node] = i
                            i += 1
                    graphs[t] = graph
                    s += 1
                # print(s)
                if t_long == 2:
                    for t in range(time[0], time[-1]):
                        if data[0] == 'r':
                            file = './gml/orignal/RC_2010-10_' + str(t) + '.gml'
                        elif data == 'sbm':
                            file = './gml/' + str(t - time[0]) + '.gml'
                            print('data is error')
                            exit()
                        graph = nx.read_gml(os.path.join(path, file))
                        # nx.draw(graph)
                        # plt.show()
                        for node in graph.nodes():
                            if node not in self.nodes.keys():
                                self.nodes[node] = i
                                i += 1
                        graphs[t + s] = graph
                self.node_num = i
                print('total number of nodes: ', i, len(graphs))
                # print(self.nodes)
                sample = self.graph2mat(graphs, self.nodes)
                # exit()
            else:
                try:
                    sample = scio.mmread(path).todense()
                except AttributeError:
                    sample = scio.mmread(path)

                self.path = path.split('.')[0]
            if modify:
                comms_folder = os.path.join(self.path, './community/'+detect+'-modify')
            else:
                comms_folder = os.path.join(self.path, './community/'+detect)
            if os.path.exists(comms_folder) == False:
                os.makedirs(comms_folder)
            elif detect == 'comsp':
                if method == 'kmeans':
                    self.dyn_coms, self.db = self.detection_C(sample, cluster_num, method, dim=dim)
                    # if input is more than 90% Contribution rate, they are different
                    print('real dim and input dim', self.dim, dim)

                    print(self.dyn_coms)
                    if modify:
                        modified_coms = self.modification(self.dyn_coms, graphs)
                        self.writ_coms(os.path.join(comms_folder, 'sr(kmeans)_'+str(dim)
                                                +'.communities'), modified_coms)
                    else:
                        print(os.path.join(comms_folder, 'sr(kmeans)_'+str(dim) +'.communities'))
                        self.writ_coms(os.path.join(comms_folder, 'sr(kmeans)_'+str(dim)
                                                +'.communities'), self.dyn_coms)
                elif method == 'sc':

                    affinity = ['rbf', 'poly', 'sigmoid', 'nearest_neighbors']  #, 'linear', 'cosine']
                    gamma = [0.01, 0.1, 1.0, 10]
                    for a in affinity:
                        self.affinity = a
                        if a == 'nearest_neighbors':
                            self.dyn_coms, self.db = self.detection_C(sample, cluster_num, method, dim=dim)
                            self.writ_coms(os.path.join(comms_folder, 'sr(sc_' +
                                self.affinity + ')' + str(dim) + '.communities'), self.dyn_coms)
                            continue

                        for g in gamma:
                            self.gamma = g
                            print(self.affinity, self.gamma)
                            try:
                                self.dyn_coms, self.db = self.detection_C(sample, cluster_num, method, dim=dim)
                                self.writ_coms(os.path.join(comms_folder, 'sr(sc_'+
                                        self.affinity+str(self.gamma)+')'+str(dim)+'.communities'),
                                               self.dyn_coms)
                            except:
                                continue
                elif method == 'dbscan':
                    eps = [0.01, 0.1, 0.5]
                    metrics = ['euclidean', 'cosine', 'manhattan']
                    self.min_samples = 1
                    self.metric = metrics[2]
                    self.eps = eps[2]
                    self.dyn_coms, self.db = self.detection_C(sample, cluster_num, method)
                    self.writ_coms(os.path.join(comms_folder, 'sr(dbscan_' + str(self.eps)
                                                + self.metric + ').communities'), self.dyn_coms)
                else:
                    print('method is nonexistent')
                    exit()
                print(self.dyn_coms)
            elif detect == 'map':
                if method == 'kmeans':
                    self.dyn_coms = self.detection_E(sample, cluster_num, method, dim=-1)
                    self.writ_coms(os.path.join(comms_folder, 'sr(kmeans).communities'), self.dyn_coms)
                elif method == 'sc':

                    affinity = ['rbf', 'poly', 'sigmoid', 'nearest_neighbors']  #, 'linear', 'cosine']
                    gamma = [0.01, 0.1, 1.0, 10]
                    self.affinity = affinity[0]  # 0-2, 3
                    self.gamma = gamma[0]  # 0-3
                    print(self.affinity, self.gamma)
                    self.dyn_coms = self.detection_E(sample, cluster_num, method, dim=-1)
                    self.writ_coms(os.path.join(comms_folder, 'sr(sc_'+
                                                self.affinity+str(self.gamma)+').communities'), self.dyn_coms)
                elif method == 'dbscan':
                    eps = [0.01, 0.1, 0.5]
                    metrics = ['euclidean', 'cosine', 'manhattan']
                    self.min_samples = 1
                    self.metric = metrics[2]
                    self.eps = eps[2]
                    self.dyn_coms = self.detection_E(sample, cluster_num, method, dim=-1)
                    self.writ_coms(os.path.join(comms_folder, 'sr(dbscan_' + str(self.eps)
                                                + self.metric + ').communities'), self.dyn_coms)
                else:
                    print('method is nonexistent')
                    exit()
                print(self.dyn_coms)
            elif detect == 'surep':
                if method == 'kmeans':
                    self.dyn_coms, self.db = self.detection_G(sample, cluster_num, method, dim=dim)
                    print(self.dyn_coms)
                    self.writ_coms(os.path.join(comms_folder, 'sr(kmeans)_'+str(dim)
                                                +'.communities'), self.dyn_coms)
                elif method == 'sc':

                    affinity = ['rbf', 'poly', 'sigmoid', 'nearest_neighbors']  #, 'linear', 'cosine']
                    gamma = [0.01, 0.1, 1.0, 10]
                    for a in affinity:
                        self.affinity = a
                        if a == 'nearest_neighbors':
                            self.dyn_coms, self.db = self.detection_G(sample, cluster_num, method, dim=dim)
                            self.writ_coms(os.path.join(comms_folder, 'sr(sc_' +
                                self.affinity + ')' + str(dim) + '.communities'), self.dyn_coms)
                            continue

                        for g in gamma:
                            self.gamma = g
                            print(self.affinity, self.gamma)
                            try:
                                self.dyn_coms, self.db = self.detection_G(sample, cluster_num, method, dim=dim)
                                self.writ_coms(os.path.join(comms_folder, 'sr(sc_'+
                                        self.affinity+str(self.gamma)+')'+str(dim)+'.communities'),
                                               self.dyn_coms)
                            except:
                                continue
                elif method == 'dbscan':
                    eps = [0.01, 0.1, 0.5]
                    metrics = ['euclidean', 'cosine', 'manhattan']
                    self.min_samples = 1
                    self.metric = metrics[2]
                    self.eps = eps[2]
                    self.dyn_coms, self.db = self.detection_G(sample, cluster_num, method)
                    self.writ_coms(os.path.join(comms_folder, 'sr(dbscan_' + str(self.eps)
                                                + self.metric + ').communities'), self.dyn_coms)
                else:
                    print('method is nonexistent')
                    exit()
                print(self.dyn_coms)

            elif detect == 'none':
                if method == 'kmeans':
                    self.dyn_coms, self.db = self.detection_H(sample, cluster_num, method)
                    print(self.dyn_coms)
                    self.writ_coms(os.path.join(comms_folder, 'sr(kmeans)_'+str(dim)
                                                +'.communities'), self.dyn_coms)
            else:
                print('detection method is nonexistent')
        else:
            print('parameter is wrong')


    def cluster_A(self, node_embedding, nodes, cluster_num):
        """
        algorithm: KMeans
        :param node_embedding: each row is a sample, each column is a feature
        :param cluster_num:
        :return: dict {labels:[node1, node2,...]}
        """
        kmeans = Cluster.KMeans(n_clusters=cluster_num, random_state=self.random_state).fit(node_embedding)
        labels_list = kmeans.labels_
        labels = defaultdict(list)
        for i in range(len(labels_list)):
            labels[labels_list[i]].append(nodes[i])
        return labels

    def cluster_B(self, node_embedding, nodes, cluster_num, gamma=0.1, affinity='rbf'):
        """
        algorithm: SpectralClustering (sc)
        :param node_embedding: each row is a sample, each column is a feature
        :param cluster_num:
        :return: dict {labels:[node1, node2,...]}
        affinity:rbf, poly, sigmoid, laplacian and chi2
        """
        # rbf   poly   nearest_neighbors

        sc = Cluster.SpectralClustering(n_clusters=cluster_num, gamma=gamma, random_state=self.random_state, affinity=affinity).fit(node_embedding)
        # parameter as following is fit  R3
        # sc = Cluster.SpectralClustering(n_clusters=cluster_num, gamma=0.01, affinity='nearest_neighbors').fit(node_embedding)
        labels_list = sc.labels_
        labels = defaultdict(list)
        for i in range(len(labels_list)):
            labels[labels_list[i]].append(nodes[i])
        return labels

    def cluster_C(self, node_embedding, nodes):
        """
        algorithm: DBSCAN
        :param node_embedding: each row is a sample, each column is a feature
        :param cluster_num:
        :return: dict {labels:[node1, node2,...]}, metric='cosine''cityblock''euclidean'
        """
        dbscan = Cluster.DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric).fit(node_embedding)
        labels_list = dbscan.labels_
        labels = defaultdict(list)
        for i in range(len(labels_list)):
            labels[labels_list[i]].append(nodes[i])
        return labels


    def node_filter(self, clusters):
        comms = {}
        for n, c in enumerate(clusters):
            tf = n // self.node_num
            t = tf + self.time[0]
            node_id = n % self.node_num
            # print(tf, node_id)
            if self.index_node[t][node_id] not in self.hidden_nodes[tf]:
                try:
                    comms[c].append(str(self.index_node[t][node_id]) + "-t" + str(tf))
                except KeyError:
                    comms[c] = [str(self.index_node[t][node_id]) + "-t" + str(tf)]
        return comms

    def detection_C(self, sample, cluster_num, method, dim=None):
        """
        best plan for dynamic community detection [ComSP]
        :param sample:
        :param cluster_num:
        :param method:
        :return:
        """
        x, y = sample.shape
        n = int(np.sqrt(y))

        if dim == None:
            dim = cluster_num
        elif dim == -1:
            dim = n

        time_represent = []
        time_node = []
        adj_mat = np.zeros((n, n*x))
        for t in range(x):
            adj = sample[t]
            adj.shape = n, n
            adj_mat[0::, t*n:(t+1)*n] = adj
        adj_mat = adj_mat @ adj_mat.T
        print('M')
        print(adj_mat.shape)
        # self.drawMat(adj_mat)
        adj_mat.shape = 1, -1
        print(dim)
        represenation, eig_vector, _ = self.representation(adj_mat, dim)  # n
        print(dim)

        for t in range(x):
            sample_mat = sample[t]
            represenation, _ = self.representation_map(sample_mat, eig_vector)

            represenation = np.array(self.positionMat(represenation.T))  # better result
            for i in range(n):
                try:
                    if self.index_node[t][i] in self.hidden_nodes[t]:
                        continue
                    node = self.index_node[t][i]
                except:
                    node = str(i)
                time_represent.append(represenation[i, :])
                time_node.append(node + '-t' + str(t))
        time_represent = np.array(time_represent)
        print(time_represent.shape)
        print(method)
        if method == 'kmeans':
            dyn_coms = self.cluster_A(time_represent, time_node, cluster_num)
        elif method == 'sc':
            dyn_coms = self.cluster_B(time_represent, time_node, cluster_num, gamma=self.gamma, affinity=self.affinity)
        elif method == 'dbscan':
            dyn_coms = self.cluster_C(time_represent, time_node)
        # elif method == ''
        db = self.db_index(dyn_coms, time_node, np.mat(time_represent))
        # db = 0
        return dyn_coms, db


    def detection_E(self, sample, cluster_num, method, dim=None):
        """
        no reconstruction but mapping
        :param sample:
        :param cluster_num:
        :param method:
        :return:
        """
        x, y = sample.shape
        n = int(np.sqrt(y))

        if dim == None:
            dim = cluster_num
        elif dim == -1:
            dim = n
        time_represent = []
        time_node = []
        adj_mat = np.zeros((n, n*x))
        for t in range(x):
            adj = sample[t]
            adj.shape = n, n
            adj_mat[0::, t*n:(t+1)*n] = adj
        adj_mat = adj_mat @ adj_mat.T
        print(adj_mat.shape)
        adj_mat.shape = 1, -1
        print(dim)
        represenation, eig_vector, _ = self.representation(adj_mat, dim)  # n

        print(eig_vector.shape)
        for t in range(x):
            sample_mat = sample[t]
            represenation, f = self.representation_map(sample_mat, eig_vector)

            represenation = f

            for i in range(n):
                try:
                    if self.index_node[t][i] in self.hidden_nodes[t]:
                        continue
                    node = self.index_node[t][i]
                except:
                    node = str(i)
                time_represent.append(represenation[i, :])
                time_node.append(node + '-t' + str(t))
        time_represent = np.array(time_represent)
        print(time_represent.shape)
        print(method)
        if method == 'kmeans':
            dyn_coms = self.cluster_A(time_represent, time_node, cluster_num)
        elif method == 'sc':
            dyn_coms = self.cluster_B(time_represent, time_node, cluster_num, gamma=self.gamma, affinity=self.affinity)
        elif method == 'dbscan':
            dyn_coms = self.cluster_C(time_represent, time_node)
        # elif method == ''
        return dyn_coms


    def detection_G(self, sample, cluster_num, method, dim=None):
        """
        SuRep reconstruct
        :param sample:
        :param cluster_num:
        :param method:
        :return:
        """
        x, y = sample.shape
        n = int(np.sqrt(y))

        if dim == None:
            dim = cluster_num
        elif dim == -1:
            dim = n

        time_represent = []
        time_node = []
        adj_mat = sample
        print('shape:', adj_mat.shape)
        # self.drawMat(adj_mat)
        represenation, eig_vector, _ = self.representation(adj_mat, dim)  # n

        for t in range(x):
            sample_mat = sample[t]
            represenation, _ = self.representation_map(sample_mat, eig_vector)
            represenation = np.array(self.positionMat(represenation.T))  # better result
            for i in range(n):
                try:
                    if self.index_node[t][i] in self.hidden_nodes[t]:
                        continue
                    node = self.index_node[t][i]
                except:
                    node = str(i)
                time_represent.append(represenation[i, :])
                time_node.append(node + '-t' + str(t))
        time_represent = np.array(time_represent)
        print(time_represent.shape)
        print(method)
        if method == 'kmeans':
            dyn_coms = self.cluster_A(time_represent, time_node, cluster_num)
        elif method == 'sc':
            dyn_coms = self.cluster_B(time_represent, time_node, cluster_num, gamma=self.gamma, affinity=self.affinity)
        elif method == 'dbscan':
            dyn_coms = self.cluster_C(time_represent, time_node)
        db = self.db_index(dyn_coms, time_node, np.mat(time_represent))
        return dyn_coms, db

    def detection_H(self, sample, cluster_num, method):
        """
        nothing to do just adjacency matrix
        :param sample:
        :param cluster_num:
        :param method:
        :return:
        """
        x, y = sample.shape
        n = int(np.sqrt(y))

        time_represent = []
        time_node = []

        for t in range(x):
            sample_mat = sample[t]
            sample_mat.shape = n, n
            represenation = sample_mat  # none
            represenation = np.array(self.positionMat(represenation.T))  # better result
            for i in range(n):
                try:
                    if self.index_node[t][i] in self.hidden_nodes[t]:
                        continue
                    node = self.index_node[t][i]
                except:
                    node = str(i)
                time_represent.append(represenation[i, :])
                time_node.append(node + '-t' + str(t))
        time_represent = np.array(time_represent)
        print(time_represent.shape)
        print(method)
        if method == 'kmeans':
            dyn_coms = self.cluster_A(time_represent, time_node, cluster_num)
        elif method == 'sc':
            dyn_coms = self.cluster_B(time_represent, time_node, cluster_num, gamma=self.gamma, affinity=self.affinity)
        elif method == 'dbscan':
            dyn_coms = self.cluster_C(time_represent, time_node)
        db = self.db_index(dyn_coms, time_node, np.mat(time_represent))
        return dyn_coms, db


    def graph2mat(self, graphs, nodes):
        node_num = len(nodes)
        print(node_num)
        sample = []
        # save hidden nodes, {tf: nodes}
        self.hidden_nodes = defaultdict(list)
        # save {t:{index:node}}
        self.index_node = defaultdict(dict)
        for t in graphs.keys():
            graph = graphs[t]
            for node in nodes.keys():
                self.index_node[t][nodes[node]] = node
                if node not in list(graph.node()):
                    self.hidden_nodes[t].append(node)
            # plan B
            # a_mat = np.diag(np.ones(node_num)*1e-4)

            a_mat = np.zeros((node_num, node_num))
            for u, v, d in graph.edges(data=True):
                a_mat[nodes[u], nodes[v]] = 1
                a_mat[nodes[v], nodes[u]] = 1
            a_mat.shape = 1, node_num*node_num
            sample.append(np.array(a_mat)[0])
        return np.mat(sample)

    def reconstruct_graph(self, sample, cluster_num, graphs=None):
        if graphs == None:
            x, y = sample.shape
            for t in range(x):
                sample_mat = sample[t]
                if self.data[0] == 'r':
                    file = './surep/RC_2010-09_' + str(t) + '.gml'
                elif self.data == 'sbm':
                    file = './surep/' + str(t) + '.gml'
                reconstruct, _, _ = self.representation(sample_mat, cluster_num)
                graph = self.mat2graph(self.positionMat(reconstruct), t+self.time[0], graphs)
                nx.write_gml(graph, os.path.join(self.path, file))
        else:
            for t in range(self.time[0], self.time[1]):
                graph = graphs[t]
                if self.data[0] == 'r':
                    file = './surep(existent)/RC_2010-09_' + str(t) + '.gml'
                elif self.data == 'sbm':
                    file = './surep(existent)/' + str(t) + '.gml'
                node_index = dict([(node, i) for i, node in enumerate(graph.nodes())])
                index_node = dict([(node_index[node], node) for node in node_index.keys()])
                node_num = len(node_index)
                a_mat = np.zeros((node_num, node_num))
                for u, v, d in graph.edges(data=True):
                    try:
                        a_mat[node_index[u], node_index[v]] = d['weight']
                        a_mat[node_index[v], node_index[u]] = d['weight']
                    except KeyError:
                        a_mat[node_index[u], node_index[v]] = 1
                        a_mat[node_index[v], node_index[u]] = 1
                a_mat.shape = 1, node_num*node_num
                try:
                    reconstruct, _, _ = self.representation(a_mat, cluster_num)
                    graph.remove_edges_from(list(graph.edges()))
                    graph.add_edges_from([(index_node[i], index_node[j]) for i in range(node_num) for j in range(node_num)
                                      if reconstruct[i, j] != 0])
                except:
                    print('error in reconstruction')
                finally:
                    nx.write_gml(graph, os.path.join(self.path, file))

    def mat2graph(self, sample_mat, time, graphs=None):
        if graphs == None:
            graph = nx.Graph()
            graph.add_nodes_from([node for node in self.nodes.keys()])
        else:
            graph = graphs[time]
            edges = graph.edges()
            graph.remove_edges_from(edges)
        x, y = sample_mat.shape
        for i in range(x):
            for j in range(y):
                if sample_mat[i, j] != 0:
                    if self.index_node[time][i] in graph.nodes() and self.index_node[time][j] in graph.nodes():
                        graph.add_edge(self.index_node[time][i], self.index_node[time][j])
        return graph

    def positionMat(self, sample):
        tem = np.zeros_like(sample, dtype=float)
        x, y = sample.shape
        for i in range(x):
            for j in range(y):
                if sample[i, j] > 0:
                    tem[i, j] = 1
        return tem

    def drawMat(self, adj_mat, filename=None):
        # print(adj_mat)
        # plt.figure(figsize=[8, 7])
        norm = colors.Normalize(vmin=-1, vmax=1)
        cmp = plt.cm.RdBu  # coolwarm#bwr  # seismic
        cbar = plt.matshow(adj_mat, cmap=cmp)  # Greens
        cbar.set_norm(norm)
        plt.xticks([])
        plt.yticks([])
        cbar = plt.colorbar(cbar, cmap=cmp, norm=norm, ticks=[-1, 0, 1])
        cbar.ax.set_yticklabels(['-1', '0', '1'])  # vertically oriented colorbar

        def rect(pos):
            r = plt.Rectangle(pos - 0.5, 1, 1, facecolor="none", edgecolor="#C0C0C0", linewidth=0.5)
            plt.gca().add_patch(r)

        x, y = np.meshgrid(np.arange(adj_mat.shape[1]), np.arange(adj_mat.shape[1]))
        m = np.c_[x[adj_mat.astype(bool)], y[adj_mat.astype(bool)]]
        m_ = np.c_[x[~adj_mat.astype(bool)], y[~adj_mat.astype(bool)]]
        for pos in m:
            rect(pos)
        for pos in m_:
            rect(pos)
        if filename != None:
            plt.savefig(filename)
        else:
            plt.show()

    def writ_coms(self, file, groundtruth):
        with open(file, 'w') as fp:
            for k in groundtruth.keys():
                fp.write(str(k) + ': [')
                for value in groundtruth[k]:
                    fp.write(value + ', ')
                fp.write('],\n')

if __name__ == '__main__':
    # dynamic network
    t_long = 1
    # t_long = 2
    # data = 'r4'  # R-I
    # data = 'r0'  # R-II
    # data = 'r3'  # R-III
    data = 'sbm1000'
    if data == 'r4':
        if t_long == 1:
            file = "E:\\DATASet\\Reddit\\reddit\\2010-09(sampling ratio 1.0)\\\R4-\\undirected"
        elif t_long == 2:
            file = "E:\\DATASet\\Reddit\\reddit\\2010-09-10(sampling ratio 1.0)\\\R4-\\undirected"
        else:
            print('t_long is smaller than three')
        cluster_num = 3
        gml = True
        time = [0, 4]
        dim = 5
    elif data == 'r0':
        file = "E:\\DATASet\\Reddit\\reddit\\2010-09(sampling ratio 1.0)\\\R0-\\undirected"
        cluster_num = 2
        gml = True
        time = [0, 4]
        dim = None
    elif data == 'r3':
        if t_long == 1:
            file = "E:\\DATASet\\Reddit\\reddit\\2010-09(sampling ratio 1.0)\\\R3-\\undirected"
            dim = 14
        elif t_long == 2:
            file = "E:\\DATASet\\Reddit\\reddit\\2010-09-10(sampling ratio 1.0)\\\R3-\\undirected"
            dim = 15
        else:
            print('t_long is smaller than three')
        cluster_num = 17
        gml = True
        time = [0, 4]
    elif data == 'sbm1000':
        file = "G:\CodeSet\workspace\HGCN\sinmulateFordraw\SBM"
        cluster_num = 4
        gml = True
        time = [0, 4]
        dim = 3

    plan = ['comsp', 'surep', 'none']
    methods = ['kmeans', 'sc', 'dbscan']
    # detect = DetectDyn(file, data, cluster_num, gml, time, detect='comsp', method=methods[2])
    # detect = DetectDyn(file, data, cluster_num, gml, time, t_long, detect='none', method=methods[0], dim=dim)
    detect = DetectDyn(file, data, cluster_num, gml, time, t_long, detect='surep', method=methods[0], dim=dim)
