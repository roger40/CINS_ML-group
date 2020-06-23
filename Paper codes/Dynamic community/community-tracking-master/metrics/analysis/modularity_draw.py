# import matplotlib
# matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict
import os


def readlabel(file, sub_nodes=None):
        labels_dic = {}
        with open(file, 'r') as fp:
            for line in fp.readlines():
                lines = line.split(':')
                cid = int(lines[0].split()[-1])
                # print(cid)
                labels_dic[cid] = []
                temp1 = lines[-1].split(']')[0].split('[')[-1]
                nodes = temp1.split()
                for node in nodes:
                    # print(node)
                    if len(node) > 1:  # is string of node's name
                        node = node.split(',')[0]
                        if sub_nodes == None:
                            labels_dic[cid].append(node)
                        else:
                            if node in sub_nodes:
                                labels_dic[cid].append(node)
        return labels_dic

def get_time(labels_dict):
    times = defaultdict(dict)
    for c in labels_dict.keys():
        for n_t in labels_dict[c]:
            node, time = n_t.split('-t')
            time = int(time)
            times[time][node] = c
    return times

#python2.7
#type(graph) = networkx.Graph
def Q(graph, cluster):
    # graph = nx.Graph()
    nodes = [node for node in cluster.keys()]
    e = 0.0
    a_2 = 0.0
    cluster_degree_table = {}
    for vtx in nodes:
        label = cluster[vtx]
        adj = list(graph.neighbors(vtx))
        for neighbor in adj:
            if neighbor not in nodes:
                continue
            if label == cluster[neighbor]:
                e += 1
        if label not in cluster_degree_table:
            cluster_degree_table[label] = 0
        cluster_degree_table[label] += len(adj)
    e /= 2 * graph.number_of_edges()

    for label, cnt in cluster_degree_table.items():
        a = 0.5 * cnt / graph.number_of_edges()
        a_2 += a * a

    Q = e - a_2
    return Q

def draw_modular(path):
    return

def main(dataset, t_long, labels):
    import os
    if dataset == 'r4':
        cluster_num = 3
        if t_long == 1:
            tt = 4
            title = 'Reddit-II(a)'
            path = 'E:\\DATASet\\Reddit\\reddit\\2010-09(sampling ratio 1.0)\\R4-\\undirected'
        elif t_long == 2:
            tt = 8
            title = 'Reddit-II(b)'
            path = 'E:\\DATASet\\Reddit\\reddit\\2010-09-10(sampling ratio 1.0)\\R4-\\undirected'
        else:
            print('t_long is smaller than three')
    elif dataset == 'r0':
        title = 'Reddit-I'
        tt = 4
        cluster_num = 2
        path = 'E:\\DATASet\\Reddit\\reddit\\2010-09(sampling ratio 1.0)\\R0-\\undirected'
        if t_long > 1:
            print('t_long is smaller than three')
            exit()
    elif dataset == 'r3':
        cluster_num = 17
        if t_long == 1:
            tt = 4
            title = 'Reddit-III(a)'
            path = 'E:\\DATASet\\Reddit\\reddit\\2010-09(sampling ratio 1.0)\\R3-\\undirected'
        elif t_long == 2:
            tt = 8
            title = 'Reddit-III(b)'
            path = 'E:\\DATASet\\Reddit\\reddit\\2010-09-10(sampling ratio 1.0)\\R3-\\undirected'
        else:
            print('t_long is smaller than three')
    elif dataset == 'sbm':
        tt = 2
        title = 'SBM'
        cluster_num = 4
        path = 'G:\\CodeSet\\workspace\\HGCN\\sinmulateFordraw\\figure7'
    elif dataset == 'sbm2':
        tt = 4
        title = 'SBM'
        cluster_num = 4
        path = 'G:\\CodeSet\\workspace\\HGCN\\sinmulateFordraw\\test-figure7'
    elif dataset == 'test':
        tt = 4
        cluster_num = 17
        path = "E:\\DATASet\\Reddit\\reddit\\test-2010-10\\R3-\\undirected"
        cluster_num = 17
        title = '2010-10'

    sd = {}
    if dataset[0] == 'r' or dataset[0] == 't':
        s = 0
        for t in range(tt):
            # file = 'gml/RC_2010-09_' + str(t)+'.gml'
            file = 'gml/'+str(t)+'.gml'
            # file = 'surep(existent)/RC_2010-09_' + str(t) + '.gml'
            try:
                graph = nx.read_gml(os.path.join(path, file))
            except:
                continue
            sd[t] = graph
            s += 1
        print(s)
        # if t_long == 2:
        #     for t in range(tt):
        #         file = 'gml/RC_2010-10_' + str(t) + '.gml'
        #         file = 'gml/' + str(t + tt) + '.gml'
        #         # file = 'surep(existent)/RC_2010-09_' + str(t)+'.gml'
        #         graph = nx.read_gml(os.path.join(path, file))
        #         sd[t + s] = graph
    elif dataset == 'sbm':
        for t in range(tt):
            file = 'gml/' + str(t)+'.gml'
            # file = 'surep(existent)/' + str(t) + '.gml'
            graph = nx.read_gml(os.path.join(path, file))
            sd[t] = graph
    elif dataset == 'sbm2':
        for t in range(tt):
            file = 'gml/' + str(t)+'.gml'
            # file = 'surep(existent)/' + str(t) + '.gml'
            graph = nx.read_gml(os.path.join(path, file))
            sd[t] = graph

    plt.figure(figsize=(7, 6))
    mark = ['^', '*', '.', '|', 'x', 's']
    i = 0
    for method in labels:
        print(method)
        if method == 'NNTF':
            file = 'tf.communities'
        elif method == 'TR-NOC':
            file = 'trn.communities'
        elif method == 'PisCES':
            file = 'ps.communities'
        elif method == 'ts-Spect':
            file = 'twoStep(sc).communities'
        elif method == 'DynGEM':
            file = 'dg.communities'
        elif method == 'ComSP':
            file = 'srs.communities'
        elif method == 'Ground-Truth':
            file = 'gt.communities'
        comms = readlabel(os.path.join(path, 'community', file))
        times_comms = get_time(comms)
        times_comms = dict(sorted(times_comms.items(), key=lambda x:x[0]))
        print(times_comms)
        modular = []
        for t in sd.keys():
            # print(t)
            graph = sd[t]
            try:
                comms = times_comms[t]
            except KeyError:
                print('KeyError', t)
                print(times_comms)
                continue
            print(comms)
            if len(comms) == 0:
                modular.append(1)
                continue
            q = Q(graph, comms)
            print(q)
            modular.append(q)
        plt.plot(modular, marker=mark[i], label=method, markersize=14, linewidth=5)
        i += 1
    print(dataset)
    # plt.axvline(4, ls='--', c='k', linewidth=4)  #loc='lower left',
    plt.legend(loc='center right', bbox_to_anchor=(0.85, 0.3), prop={'size': 16, 'weight':'normal', 'family': 'Times New Roman'})
    plt.title(title, fontsize=23, family='Times New Roman')
    plt.xticks(range(len(times_comms)), [i+1 for i in list(times_comms.keys())], fontsize=22, family='Times New Roman')
    plt.yticks(fontsize=22, family='Times New Roman')
    # plt.xlabel('Timestamp', fontsize=14, family='Times New Roman')
    plt.ylabel('Modularity', fontsize=24, family='Times New Roman')
    file = os.path.join(path, 'community', 'modularity.png')
    # plt.savefig(file)

    plt.savefig('./'+title+'.eps', format='eps', dpi=1000)
    # plt.show()

def new_main(dataset, t_long, labels, color=None):
    import os
    if dataset == 'r4':
        cluster_num = 3
        if t_long == 1:
            tt = 4
            title = 'Reddit-I(a)'
            path = 'E:\\DATASet\\Reddit\\reddit\\2010-09(sampling ratio 1.0)\\R4-\\undirected'
        elif t_long == 2:
            tt = 8
            title = 'Reddit-I(b)'
            path = 'E:\\DATASet\\Reddit\\reddit\\2010-09-10(sampling ratio 1.0)\\R4-\\undirected'
        else:
            print('t_long is smaller than three')
    elif dataset == 'r0':
        title = 'Reddit-II'
        tt = 4
        cluster_num = 2
        path = 'E:\\DATASet\\Reddit\\reddit\\2010-09(sampling ratio 1.0)\\R0-\\undirected'
        if t_long > 1:
            print('t_long is smaller than three')
            exit()
    elif dataset == 'r3':
        cluster_num = 17
        if t_long == 1:
            tt = 4
            title = 'Reddit-III(a)'
            path = 'E:\\DATASet\\Reddit\\reddit\\2010-09(sampling ratio 1.0)\\R3-\\undirected'
        elif t_long == 2:
            tt = 8
            title = 'Reddit-III(b)'
            path = 'E:\\DATASet\\Reddit\\reddit\\2010-09-10(sampling ratio 1.0)\\R3-\\undirected'
        else:
            print('t_long is smaller than three')
    elif dataset == 'sbm1000':
        tt = 4
        title = 'SBM'
        cluster_num = 4
        path = 'G:\\CodeSet\\workspace\\HGCN\\sinmulateFordraw\\SBM'

    sd = {}
    for t in range(tt):
        file = 'gml/' + str(t)+'.gml'
        graph = nx.read_gml(os.path.join(path, file))
        sd[t] = graph

    plt.figure(figsize=(7, 6))
    mark = ['^', '+', '*', '.', 's', '|', 'x', 'v', 'o']
    i = 0
    for method in labels:
        # print(method)
        if method == 'PisCES':
            file = 'pisces(after).communities'
        elif method == 'ts-Spect':
            file = 'twoStep(sc).communities'
        elif method == 'GDG':
            file = 'gdg.communities'
        elif method == 'sE-NMF':
            file = 'enmf.communities'
        elif method == 'CCPSO':
            file = 'ccpso.communities'
        elif method == 'TR-NOC':
            file = 'trn.communities'
        elif method == 'Adj-Mat':
            file = 'adj(kmeans).communities'
        elif method == 'SuRep':
            file = 'surep(kmeans).communities'
        elif method == 'ComSP':
            file = 'comsp(kmeans).communities'
        elif method == 'Ground-Truth':
            file = 'gt.communities'
        comms = readlabel(os.path.join(path, 'community/best', file))
        times_comms = get_time(comms)
        times_comms = dict(sorted(times_comms.items(), key=lambda x:x[0]))
        # print(times_comms)
        modular = []
        node_num = []
        edge_num = []
        for t in sd.keys():
            # print(t)
            graph = sd[t]
            # graph = nx.Graph
            print(graph.number_of_nodes())
            node_num.append(graph.number_of_nodes())
            edge_num.append(graph.number_of_edges())
            try:
                comms = times_comms[t]
            except KeyError:
                print('KeyError', t)
                print(times_comms)
                continue
            # print(comms)
            if len(comms) == 0:
                modular.append(1)
                continue
            q = Q(graph, comms)
            # print(q)
            modular.append(q)
        if color == None:
            plt.plot(modular, marker=mark[i], label=method, markersize=14, linewidth=5)
        else:
            plt.plot(modular, marker=mark[i], color=color[i], label=method, markersize=14, linewidth=5)
        i += 1
        # print('node')
        # print('min:', min(node_num), 'max:', max(node_num), 'var:', np.var(node_num))
        # print('edge')
        # print('min:', min(edge_num), 'max:', max(edge_num), 'mean:', np.mean(edge_num),'var:', np.var(edge_num))
        # exit()
    print(dataset)
    # plt.axvline(4, ls='--', c='k', linewidth=4)  #loc='lower left',
    # plt.legend(loc='center left', bbox_to_anchor=(-0.02, 0.32), framealpha=0.5, prop={'size': 16, 'weight':'normal', 'family': 'Times New Roman'})
    # plt.legend(loc='center right', bbox_to_anchor=(0.85, 0.3), prop={'size': 16, 'weight':'normal', 'family': 'Times New Roman'})
    plt.title(title, fontsize=23, family='Times New Roman')
    plt.xticks(range(len(times_comms)), [i+1 for i in list(times_comms.keys())], fontsize=22, family='Times New Roman')
    plt.yticks(fontsize=22, family='Times New Roman')
    # plt.xlabel('Timestamp', fontsize=14, family='Times New Roman')
    plt.ylabel('Modularity', fontsize=24, family='Times New Roman')
    file = os.path.join(path, 'community', 'modularity.png')
    # plt.savefig(file)

    plt.savefig('./modularity_fig1/'+title+'.eps', format='eps', dpi=1000)
    # plt.show()

if __name__ == '__main__':

    labels = ['ts-Spect', 'GDG', 'PisCES', 'sE-NMF', 'CCPSO', 'TR-NOC', 'Adj-Mat', 'SuRep', 'ComSP']#, 'Ground-Truth']
    color = ['#1F77B4', '#BCBD22', '#FF7F0E', '#2CA02C', '#7F7F7F', '#9467BD', '#8C564B', '#E377C2', '#D62728']
    # labels = ['TR-NOC', 'PisCES', 'ts-Spect', 'DynGEM', 'ComSP']#, 'Ground-Truth']
    # for test
    # labels = ['ComSP']
    # labels = ['TR-NOC', 'PisCES', 'ts-Spect', 'ComSP', 'Ground-Truth']
    # t_long = 1
    t_long = 2
    # dataset = 'r4'
    # dataset = 'r0'
    dataset = 'r3'
    # dataset = 'sbm'
    # dataset = 'sbm2'
    # dataset = 'sbm1000'
    # dataset = 'senate'
    # dataset = 'mit'
    # dataset = 'test'
    new_main(dataset, t_long, labels, color)
    # new_main(dataset, t_long, labels)