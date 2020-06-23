disp_avlbl = True
import os
if 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from scipy.io import mmread
from time import time

import json

from gensim.models import Word2Vec
import pandas as pd

from walker import RandomWalker

import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')


import sys
sys.path.append('..')
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from subprocess import call

from gem.embedding.static_graph_embedding import StaticGraphEmbedding
from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz


class node2vec(StaticGraphEmbedding):

    def __init__(self, *hyper_dict, **kwargs):
        ''' Initialize the node2vec class

        Args:
            d: dimension of the embedding
            max_iter: max iterations
            walk_len: length of random walk
            num_walks: number of random walks
            con_size: context size
            ret_p: return weight
            inout_p: inout weight
        '''
        hyper_params = {
            'method_name': 'node2vec_rw'
        }
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])
        for dictionary in hyper_dict:
            for key in dictionary:
                self.__setattr__('_%s' % key, dictionary[key])

    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return '%s_%d' % (self._method_name, self._d)

    def learn_embedding(self, graph=None, edge_f=None,
                        is_weighted=False, no_python=False):
        args = ["node2vec"]
        if not graph and not edge_f:
            raise Exception('graph/edge_f needed')
        if edge_f:
            graph = graph_util.loadGraphFromEdgeListTxt(edge_f)
        graph_util.saveGraphToEdgeListTxtn2v(graph, 'tempGraph.graph')
        args.append("-i:tempGraph.graph")
        args.append("-o:tempGraph.emb")
        args.append("-d:%d" % self._d)
        args.append("-l:%d" % self._walk_len)
        args.append("-r:%d" % self._num_walks)
        args.append("-k:%d" % self._con_size)
        args.append("-e:%d" % self._max_iter)
        args.append("-p:%f" % self._ret_p)
        args.append("-q:%f" % self._inout_p)
        args.append("-v")
        args.append("-dr")
        args.append("-w")
        t1 = time()
        try:
            call(args)
        except Exception as e:
            print(str(e))
            raise Exception('./node2vec not found. Please compile snap, place node2vec in the system path and grant executable permission')
        self._X = graph_util.loadEmbedding('tempGraph.emb')
        t2 = time()
        return self._X, (t2 - t1)

    def get_embedding(self):
        return self._X

    def get_edge_weight(self, i, j):
        return np.dot(self._X[i, :], self._X[j, :])

    def get_reconstructed_adj(self, X=None, node_l=None):
        if X is not None:
            node_num = X.shape[0]
            self._X = X
        else:
            node_num = self._node_num
        adj_mtx_r = np.zeros((node_num, node_num))
        for v_i in range(node_num):
            for v_j in range(node_num):
                if v_i == v_j:
                    continue
                adj_mtx_r[v_i, v_j] = self.get_edge_weight(v_i, v_j)
        return adj_mtx_r


class Node2Vec_n:

    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, workers=1):

        self.graph = graph
        self._embeddings = {}
        self.walker = RandomWalker(graph, p=p, q=q, )

        print("Preprocess transition probs...")
        self.walker.preprocess_transition_probs()

        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=1, iter=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model

        # self._embeddings = {}
        # for i in range(self.graph.number_of_nodes() + 1):
        #     self._embeddings[i] = model.wv[i]
        #     print(model.wv[i])

        # for i in self.graph.nodes():
        #     self._embeddings[i] = model.wv[i]

        return model

    def get_embeddings(self,):
        if self.w2v_model is None:
            print("model not train")
            return {}

        # print("length:", len(self.w2v_model.wv))
        # print("node 2:", self.w2v_model.wv[2])
        # print("node 3:", self.w2v_model.wv[3])
        # print("node 93:", self.w2v_model.wv[93])
        # print("node 94:", self.w2v_model.wv[94])
        # print("type:", type(self.w2v_model.wv))
        self._embeddings = {}
        self.all_embeddings = []
        l = self.graph.number_of_nodes()
        self.all_embeddings = self.w2v_model.wv[l].tolist()
        for i in range(l):
            self._embeddings[i+1] = self.all_embeddings[i]
        # for word in self.graph.nodes():
        #     # print("node:", self.w2v_model.wv[word])
        #     word = int(word) + 1
        #     self._embeddings[word] = self.w2v_model.wv[word].tolist()
        #     print("shape of ", word, "is", np.shape(self._embeddings[word]))

        return self._embeddings




class data_loarder():
    def __init__(self, filepath):
        self.filepath = filepath
        self.mtx_whole = mmread(self.filepath)
        self.num_snaps, self.num_nodes = np.shape(self.mtx_whole)
        self.num_nodes = int(np.sqrt(self.num_nodes))
        self.mtx_whole = self.mtx_whole.toarray()
        for i in range(self.mtx_whole.shape[0]):
            for j in range(self.mtx_whole.shape[1]):
                if self.mtx_whole[i][j] <= 0:
                    self.mtx_whole[i][j] = 0
                else:
                    self.mtx_whole[i][j] = 1

    def get_true_graph(self):
        temp = []
        for i in range(self.num_snaps):
            temp.append(np.reshape(self.mtx_whole[i], (self.num_nodes, self.num_nodes)))
        return temp

    def trans_adj_to_graph(self, adj):
        n = adj.shape[0]
        di_graph = nx.DiGraph()
        di_graph.add_nodes_from(range(n))
        for i in range(n):
            for j in range(n):
                if(i != j):
                    if(adj[i, j] > 0):
                        di_graph.add_edge(i, j, weight=1)

        # graph = di_graph.to_undirected()
        # di_graph = graph.to_directed()
        return di_graph


if __name__ == '__main__':
    # load Zachary's Karate graph
    # edge_f = 'data/karate.edgelist'
    # G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=False)
    # G = G.to_directed()
    # res_pre = 'results/testKarate'
    # graph_util.print_graph_stats(G)
    # t1 = time()
    # embedding = node2vec(2, 1, 80, 10, 10, 1, 1)
    # embedding.learn_embedding(graph=G, edge_f=None,
    #                           is_weighted=False, no_python=True)
    # print('node2vec:\n\tTraining time: %f' % (time() - t1))

    # viz.plot_embedding2D(embedding.get_embedding(),
    #                      di_graph=G, node_colors=None)
    # plt.show()



    filepath = "../data/simulant_24g_100n_a3a4a5.mtx"
    dl = data_loarder(filepath)
    temp = dl.get_true_graph()
    for i in range(len(temp)):
        G = dl.trans_adj_to_graph(temp[i])
        res_pre = './embedding_results/results_simulate_24g/graph%d.txt' % (i)
        print("# graph%d:" % i)
        graph_util.print_graph_stats(G)
        t1 = time()

        # print(G.edges())

        model=Node2Vec_n(G, walk_length = 10, num_walks = 80,
                   p = 0.25, q = 4, workers = 3)
        n2v_model = model.train(iter = 2000)
        embedding = model.get_embeddings()
        # print("type of embedding:", type(embedding))
        # n2v_model.save("node2vec_mit")
        # print(len(n2v_model.wv))
        print('node2vec-graph %d:\n\tTraining time: %f' % (i, time() - t1))
        # viz.plot_embedding2D(embedding,
        #                     di_graph=G, node_colors=None)
        # plt.show()
        
        fileObject = open(res_pre, 'w')
        jsObj = json.dump(embedding, fileObject)
        fileObject.close()



    # filepath = "../data/Sample_MIT.mtx"
    # dl = data_loarder(filepath)
    # temp = dl.get_true_graph()
    # for i in range(len(temp)):
    #     G = dl.trans_adj_to_graph(temp[i])
    #     res_pre = 'results/graph %d .txt' % (i)
    #     graph_util.print_graph_stats(G)
    #     t1 = time()

    #     embedding = node2vec(d=128, max_iter=2000, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1)
    #     embedding.learn_embedding(graph=G, edge_f=None,
    #                               is_weighted=False, no_python=True)
    #     print('node2vec-graph %d:\n\tTraining time: %f' % (i, time() - t1))
    #     viz.plot_embedding2D(embedding.get_embedding(),
    #                         di_graph=G, node_colors=None)
    #     plt.show()
    #     np.savetxt(res_pre, embedding.get_embedding())
