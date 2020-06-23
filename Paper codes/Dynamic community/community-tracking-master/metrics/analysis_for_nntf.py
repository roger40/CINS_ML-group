from __future__ import division
import Omega, Bcubed
# import NMI
import NMI_lanlan
from collections import Counter, OrderedDict, defaultdict
import itertools
import math
import os
from sklearn import metrics


def unravel_tf(dynamic, tfs_len):
    """

    :param dynamic:
    :param tfs_len:
    :return: dictionary of dictionaries {timeframe:
                                                {
                                                community: [node1, node2...]}}
    """
    comms = {t: {} for t in range(tfs_len)}
    for c, dyn in dynamic.items():
        for node in dyn:
            # print(node)
            tf = int(node.split('-')[-1][1])
            # node = int(node.split('-')[0].split('\'')[-1])
            node = node.split('-')[0].split('\'')[-1]
            try:
                comms[tf][c].append(node)
            except KeyError:
                comms[tf][c] = [node]
    return remove_duplicate_coms(comms)


def remove_duplicate_coms(communities):
    """
    Removes duplicates from list of lists
    :param communities:
    :return:
    """
    new_comms = {tf: {} for tf in communities.keys()}
    for tf, comms in communities.items():
        unique_coms = [set(c) for c in comms.values()]
        unique_coms = list(comms for comms, _ in itertools.groupby(unique_coms))
        for i, com in enumerate(unique_coms):
            new_comms[tf][i] = list(com)
    return new_comms


def evaluate(ground_truth, method, name, eval, duration):
    # nmi = NMI.NMI(ground_truth, method).results
    nmi = NMI_lanlan.NMI(ground_truth, method).results
    # nmi = NMI(ground_truth, method)
    omega = Omega.Omega(ground_truth, method)
    bcubed = Bcubed.Bcubed(ground_truth, method)
    results = OrderedDict()
    # results["Method"] = [name]
    # results["Eval"] = [eval]
    # results['NMI'] = ["{0:.4f}".format(nmi['NMI<Max>'])]
    results['NMI'] = ["{0:.3f}".format(nmi)]
    results['Omega'] = ["{0:.3f}".format(omega.omega_score)]
    results['Bcubed-Recall'] = ["{0:.3f}".format(bcubed.recall)]
    results['Bcubed-Precision'] = ["{0:.3f}".format(bcubed.precision)]
    results['Bcubed-F1'] = ["{0:.3f}".format(bcubed.fscore)]
    results['Duration'] = [duration]
    return results


def get_results(ground_truth, method, name, tfs_len, eval="dynamic", duration=0):
    if eval == "dynamic":
        # print(ground_truth, method)
        results = evaluate(ground_truth, method, name, eval, duration)
    elif eval == "sets":
        new_comms1 = {i: set() for i in ground_truth.keys()}
        for i, comm in ground_truth.items():
            for node in comm:
                new_comms1[i].add(node.split('-')[0])
        new_comms2 = {i: set() for i in method.keys()}
        for i, comm in method.items():
            for node in comm:
                new_comms2[i].add(node.split('-')[0])
        results = evaluate(new_comms1, new_comms2, name, eval, duration)
    elif eval == "per_tf":
        new_comms1 = unravel_tf(ground_truth, tfs_len)
        new_comms2 = unravel_tf(method, tfs_len)
        per_tf = []
        per_tf_lanlan = []
        for t in range(tfs_len):
            per_tf.append(Counter(evaluate(new_comms1[t], new_comms2[t], name, eval, duration)))
            per_tf_lanlan.append(dict(evaluate(new_comms1[t], new_comms2[t], name, eval, duration)))
        # print(per_tf_lanlan)
        # results = sum(per_tf, Counter())
        results_lanlan = defaultdict(list)
        for dic in per_tf_lanlan:
            for k in dic.keys():
                results_lanlan[k].append(dic[k][0])
        results = results_lanlan
        for key in results:
            # print(results[key])
            if all(isinstance(x, str) for x in results[key]):
                results[key] = [results[key][0]]
            else:
                results[key] = [sum(results[key]) / len(per_tf)]
        # pprint.pprint(dict(f))
        # for k, v in res.iteritems():
        #     print "KEY ", k, " VALUE ", v
    return results


def readlabel(file):
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
                    labels_dic[cid].append(node.split(',')[0])
    return labels_dic

def filter(comms_gt, comms):
    """
    filter items in comms_gt which is not existent in comms
    :param comms_gt:
    :param comms:
    :return:
    """
    nodes = []
    for c in comms.keys():
        nodes = nodes + comms[c]
    nodes = set(nodes)
    comms_gt_f = defaultdict(list)
    for c in comms_gt.keys():
        for item in comms_gt[c]:
            if item in nodes:
                comms_gt_f[c].append(item)
    return comms_gt_f, comms

# def NMI(comms_gt, comms):
#         # convert dict into list
#         node_comms_gt = {}
#         node_comms = {}
#         nodes = []
#         for k in comms_gt.keys():
#             for node in comms_gt[k]:
#                 node_comms_gt[node] = k
#                 nodes.append(node)
#         for k in comms.keys():
#             for node in comms[k]:
#                 node_comms[node] = k
#                 nodes.append(node)
#         nodes = set(nodes)
#         result = []
#         label = []
#         for node in nodes:
#             result.append(node_comms[node])
#             label.append(node_comms_gt[node])
#
#         # 标准化互信息
#         total_num = len(label)
#         cluster_counter = Counter(result)
#         original_counter = Counter(label)
#         # 计算互信息量
#         MI = 0
#         eps = 1.4e-45  # 取一个很小的值来避免log 0
#         for k in cluster_counter:
#             for j in original_counter:
#                 count = 0
#                 for i in range(len(result)):
#                     if result[i] == k and label[i] == j:
#                         count += 1
#                 p_k = 1.0 * cluster_counter[k] / total_num
#                 p_j = 1.0 * original_counter[j] / total_num
#                 p_kj = 1.0 * count / total_num
#                 MI += p_kj * math.log(p_kj / (p_k * p_j) + eps, 2)
#         # 标准化互信息量
#         H_k = 0
#         for k in cluster_counter:
#             H_k -= (1.0 * cluster_counter[k] / total_num) * math.log(1.0 * cluster_counter[k] / total_num + eps, 2)
#         H_j = 0
#         for j in original_counter:
#             H_j -= (1.0 * original_counter[j] / total_num) * math.log(1.0 * original_counter[j] / total_num + eps, 2)
#
#         return 2.0 * MI / (H_k + H_j)

def dyn_main(path, code=1):
    file_gt = 'gt.communities'
    comms_gt = readlabel(os.path.join(path, file_gt))
    if code == 1:
        file_tf = 'tf(threshold 1e-1).communities'
        comms_tf = readlabel(os.path.join(path, file_tf))
        comms = comms_tf
        method = "NNTF"
    elif code == 2:
        file_tf = 'tf(threshold 1e-2).communities'
        comms_tf = readlabel(os.path.join(path, file_tf))
        comms = comms_tf
        method = "NNTF"
    elif code == 3:
        # change input with gml
        file_tf = 'tf(threshold 1e-3).communities'
        comms_tf = readlabel(os.path.join(path, file_tf))
        comms = comms_tf
        method = "NNTF"
    elif code == 4:
        file_tf = 'tf(threshold 1e-4).communities'
        comms_tf = readlabel(os.path.join(path, file_tf))
        comms = comms_tf
        method = "NNTF"
    elif code == 5:
        file_tr = 'tr_all(default_q true).communities'
        comms_tr = readlabel(os.path.join(path, file_tr))
        comms = comms_tr
        method = "Muturank(All)"
    elif code == 6:
        file_tr = 'tr_next(default_q true).communities'
        comms_tr = readlabel(os.path.join(path, file_tr))
        comms = comms_tr
        method = "Muturank(Next)"
    elif code == 7:
        file_tr = 'tr_all(0.5).communities'
        comms_tr = readlabel(os.path.join(path, file_tr))
        comms = comms_tr
        method = "Muturank(All)"
    elif code == 8:
        file_tr = 'tr_next(0.5).communities'
        comms_tr = readlabel(os.path.join(path, file_tr))
        comms = comms_tr
        method = "Muturank(Next)"
    elif code == 9:
        # kmeans
        file_sr = 'sr(kmeans).communities'
        comms_sr = readlabel(os.path.join(path, file_sr))
        comms = comms_sr
        method = "SuRep"

    elif code == 10:
        # SpectralClustering
        # file_sr = 'sr(sc).communities'
        file_sr = 'sr(sc_rbf0.1).communities'  # for test
        comms_sr = readlabel(os.path.join(path, file_sr))
        comms = comms_sr
        method = "SuRep"
    elif code == 11:
        # DBSCAN
        file_sr = 'sr(dbscan).communities'
        comms_sr = readlabel(os.path.join(path, file_sr))
        comms = comms_sr
        method = "SuRep"
    # elif code == 12:
    #         # DBSCAN
    #         file_sr = 'sr(all).communities'
    #         comms_sr = readlabel(os.path.join(path, file_sr))
    #         comms = comms_sr
    #         method = "SuRep"
    # elif code == 13:
    #         # DBSCAN
    #         file_sr = 'sr(next).communities'
    #         comms_sr = readlabel(os.path.join(path, file_sr))
    #         comms = comms_sr
    #         method = "SuRep"
    elif code == 14:
        # PisCES
        file_ps = 'a0.00.communities'
        comms_ps = readlabel(os.path.join(path, file_ps))
        comms = comms_ps
        method = 'PisCES'
    elif code == 15:
        # PisCES
        file_ps = 'a0.01.communities'
        comms_ps = readlabel(os.path.join(path, file_ps))
        comms = comms_ps
        method = 'PisCES'
    elif code == 16:
        # PisCES
        file_ps = 'a0.05.communities'
        comms_ps = readlabel(os.path.join(path, file_ps))
        comms = comms_ps
        method = 'PisCES'
    elif code == 17:
        # PisCES
        file_ps = 'a0.10.communities'
        comms_ps = readlabel(os.path.join(path, file_ps))
        comms = comms_ps
        method = 'PisCES'
    elif code == 18:
        # PisCES
        file_ps = 'a0.15.communities'
        comms_ps = readlabel(os.path.join(path, file_ps))
        comms = comms_ps
        method = 'PisCES'
    # evaluate and print
    get_evaluate(comms_gt, comms, method, eval="dynamic")


def new_dyn_evaluate(path, method='tr'):
    """
    evaluation
    :param path: data_folder/community
    :param method: 'tr'=TimeRank; 'tf'=NNTF; 'psb'=PisCES with filtering before;
                    'psa'=PisCES, filter after clustering; 'srsc'=SuRep with spectral clustering;
                    'srdb'= SuRep with DBSCAN; 'srkm'=SuRep with K-Means
    :return:
    """
    file_gt = 'gt.communities'
    comms_gt = readlabel(os.path.join(path, file_gt))
    if method == 'tr':
        folder = os.path.join(path, 'TimeRank')
        for file in os.listdir(folder):
            print(file + '------------------------')
            comms = readlabel(os.path.join(folder, file))
            get_evaluate(comms_gt, comms, method, eval="dynamic")
    elif method == 'tf':
        folder = os.path.join(path, 'NNTF')
        for file in os.listdir(folder):
            print(file + '------------------------')
            comms = readlabel(os.path.join(folder, file))
            get_evaluate(comms_gt, comms, method, eval="dynamic")
    elif method == 'psb':
        folder = os.path.join(path, 'PisCES/before')
        for file in os.listdir(folder):
            print(file + '------------------------')
            comms = readlabel(os.path.join(folder, file))
            get_evaluate(comms_gt, comms, method, eval="dynamic")
    elif method == 'psa':
        folder = os.path.join(path, 'PisCES/after')
        for file in os.listdir(folder):
            print(file + '------------------------')
            comms = readlabel(os.path.join(folder, file))
            get_evaluate(comms_gt, comms, method, eval="dynamic")
    elif method == 'srsc':
        folder = os.path.join(path, 'SuRep-SC')
        for file in os.listdir(folder):
            print(file + '------------------------')
            comms = readlabel(os.path.join(folder, file))
            get_evaluate(comms_gt, comms, method, eval="dynamic")
    elif method == 'srdb':
        folder = os.path.join(path, 'SuRep-DB')
        for file in os.listdir(folder):
            print(file + '------------------------')
            comms = readlabel(os.path.join(folder, file))
            get_evaluate(comms_gt, comms, method, eval="dynamic")
    elif method == 'srkm':
        print('K-Means' + '------------------------')
        file = 'sr(kmeans).communities'
        comms = readlabel(os.path.join(path, file))
        get_evaluate(comms_gt, comms, method, eval="dynamic")


def get_evaluate(comms_gt, comms, method, eval="dynamic"):
    all_res = []
    all_res.append(get_results(comms_gt, comms, method, 4, eval=eval))  # eval == dynamic, 4 does not work.
    # all_res.append(get_results(comms_gt, comms, method, 4, eval="sets"))
    # all_res.append(get_results(comms_gt, comms, method, 4, eval="per_tf"))
    results = OrderedDict()
    # results["Method"] = []
    # results['Eval'] = []
    results['NMI'] = []
    results['Omega'] = []
    results['Bcubed-Precision'] = []
    results['Bcubed-Recall'] = []
    results['Bcubed-F1'] = []
    results['Duration'] = []

    from tabulate import tabulate
    for res in all_res:
        for k, v in res.items():
            results[k].extend(v)
    # print(tabulate(results, headers="keys", tablefmt="fancy_grid").encode('utf8')+"\n")
    print(results)


def stat_main(path):
    file_gt = 'gt.communities'
    comms_gt = readlabel(os.path.join(path, file_gt))

    file_sr = 'sr(kmeans).communities'
    comms_sr = readlabel(os.path.join(path, file_sr))
    comms = comms_sr
    method = "SuRep"

    # print(method, NMI(comms_gt, comms))
    all_res = []
    all_res.append(get_results(comms_gt, comms, method, 4, eval="dynamic"))
    # all_res.append(get_results(comms_gt, comms, method, 4, eval="sets"))
    # all_res.append(get_results(comms_gt, comms, method, 4, eval="per_tf"))
    results = OrderedDict()
    results["Method"] = []
    results['Eval'] = []
    results['NMI'] = []
    results['Omega'] = []
    results['Bcubed-Precision'] = []
    results['Bcubed-Recall'] = []
    results['Bcubed-F1'] = []
    results['Duration'] = []

    from tabulate import tabulate
    for res in all_res:
        for k, v in res.items():
            results[k].extend(v)
    # print(tabulate(results, headers="keys", tablefmt="fancy_grid").encode('utf8')+"\n")
    print(results)


if __name__ == "__main__":
    # comms3 = {0: ['1-t0', '2-t0', '3-t0', '4-t0', '1-t1', '2-t1',  '3-t1', '4-t1', '1-t2', '2-t2', '3-t2', '4-t2'],
    #           1: ['11-t1', '12-t1', '13-t1'],
    #           2: ['5-t2', '6-t2', '7-t2', '5-t0', '6-t0', '7-t0']}
    # comms4 = {1: ['1-t0', '2-t0', '3-t0', '4-t0', '1-t1', '2-t1',  '3-t1', '4-t1', '1-t2', '2-t2', '3-t2','4-t2'],
    #           2: ['11-t1', '12-t1', '13-t1'],
    #           3: ['5-t2', '6-t2', '7-t2'],
    #           4: ['5-t0', '6-t0', '7-t0']}
    # comms5 = {5: ['5-t0', '6-t0', '7-t0'],
    #           1: ['1-t0', '2-t0', '3-t0', '4-t0', '1-t1', '2-t1',  '3-t1', '4-t1', '1-t2', '2-t2', '3-t2', '4-t2'],
    #           2: ['11-t1', '12-t1', '13-t1', '5-t0', '6-t0', '7-t0'],
    #           3: ['5-t0', '6-t0', '7-t0', '5-t2', '6-t2', '7-t2'],
    #           4: ['5-t0', '7-t0', '6-t0', ]}
    # all_res = []
    # all_res.append(get_results(comms4, comms5, "Muturank", 3, eval="dynamic"))
    # all_res.append(get_results(comms4, comms5, "Muturank", 3, eval="sets"))
    # all_res.append(get_results(comms4, comms5, "Muturank", 3, eval="per_tf"))
    # results = OrderedDict()
    # results["Method"] = []
    # results['Eval'] = []
    # results['NMI'] = []
    # results['Omega'] = []
    # results['Bcubed-Precision'] = []
    # results['Bcubed-Recall'] = []
    # results['Bcubed-F1'] = []
    # results['Duration'] = []
    #
    # from tabulate import tabulate
    # for res in all_res:
    #     for k, v in res.iteritems():
    #         results[k].extend(v)
    # print(tabulate(results, headers="keys", tablefmt="fancy_grid").encode('utf8')+"\n")

    # file_gt = 'E:\\DATASet\\Reddit\\reddit\\2010-09\\undirected\\groundtruth.communities'
    # comms_gt = readlabel(file_gt)
    #
    # file_tf = 'E:\\DATASet\\Reddit\\reddit\\2010-09\\undirected\\NNTF.communities'
    # comms_tf = readlabel(file_tf)
    # comms = comms_tf
    # method = "NNTF"
    #
    # file_tr = 'E:\\DATASet\\Reddit\\reddit\\2010-09\\undirected\\TimeRank.communities'
    # comms_tr = readlabel(file_tr)
    # comms = comms_tr
    # method = "Muturank"
    #
    # file_sr = 'E:\\DATASet\\Reddit\\reddit\\2010-09\\undirected\\SuRepA.communities'
    # comms_sr = readlabel(file_sr)
    # comms = comms_sr
    # method = "SuRep"

    # Reddit
    # path = 'E:\\DATASet\\Reddit\\reddit\\2010-09(sampling ratio 1.0)\\R4-\\undirected\\community'
    # path = 'E:\\DATASet\\Reddit\\reddit\\2010-09-10(sampling ratio 1.0)\\R4-\\undirected\\community'
    # path = 'E:\\DATASet\\Reddit\\reddit\\2010-09(sampling ratio 1.0)\\R0-\\undirected\\community'
    path = 'E:\\DATASet\\Reddit\\reddit\\2010-09(sampling ratio 1.0)\\R3-\\undirected\\community'
    # path = 'E:\\DATASet\\Reddit\\reddit\\2010-09-10(sampling ratio 1.0)\\R3-\\undirected\\community'
    # SBM
    # path = 'G:\\CodeSet\\workspace\\HGCN\\sinmulateFordraw\\figure7\\community'
    # MIT
    # path = 'E:/DATASet/RealityMining/RealityMining/community'

    # evalutaion for one method
    # print('TimeRank')
    # new_dyn_evaluate(path, method='tr')
    print('NNTF')
    new_dyn_evaluate(path, method='tf')
    # print('PisCES, filtering before clustering')
    # new_dyn_evaluate(path, method='psb')
    # print('PisCES, filtering after clustering')
    # new_dyn_evaluate(path, method='psa')
    # print('SuRep + K-Means')
    # new_dyn_evaluate(path, method='srkm')
    # print('SuRep + Spectral Clustering')
    # new_dyn_evaluate(path, method='srsc')
    # print('SuRep + DBSCAN')
    # new_dyn_evaluate(path, method='srdb')

    # alone evaluation

    # dyn_main(path, code=5)
    # print('kmeans')
    # dyn_main(path, code=9)
    # print('SpectralClustering')
    # dyn_main(path, code=10)
    # print('DBSCAN')
    # dyn_main(path, code=11)

    # print('SuRep all')
    # dyn_main(path, code=12)
    # print('SuRep next')
    # dyn_main(path, code=13)

    # print('nntf 1e-1')
    # dyn_main(path, code=1)
    # print('nntf 1e-2')
    # dyn_main(path, code=2)
    # print('nntf 1e-3')
    # dyn_main(path, code=3)
    # print('nntf 1e-4')
    # dyn_main(path, code=4)

    # print('timerank-noc-u')
    # dyn_main(path, code=6)
    # print('timerank-aoc-u')
    # dyn_main(path, code=5)
    # print('timerank-noc')
    # dyn_main(path, code=8)
    # print('timerank-aoc')
    # dyn_main(path, code=7)

    # print('PisCES alpha=0.00')
    # dyn_main(path, code=14)
    # print('PisCES alpha=0.01')
    # dyn_main(path, code=15)
    # print('PisCES alpha=0.05')
    # dyn_main(path, code=16)
    # print('PisCES alpha=0.10')
    # dyn_main(path, code=17)
    # print('PisCES alpha=0.15')
    # dyn_main(path, code=18)

    # # Karate
    # path = 'E:/DATASet/node classify/karate/communities'
    #
    # stat_main(path)