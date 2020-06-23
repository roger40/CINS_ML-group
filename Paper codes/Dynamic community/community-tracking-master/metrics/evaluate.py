
from __future__ import division
# import Omega
import Omega_lanlan as Omega
import Bcubed
# import NMI
import NMI_lanlan
from collections import Counter, OrderedDict, defaultdict
import itertools
import math
import os

class Metric(object):

    def unravel_tf(self, dynamic, tfs_len):
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
        return self.remove_duplicate_coms(comms)


    def remove_duplicate_coms(self, communities):
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

    def times_comms(self, communities):
        new_comms = defaultdict(dict)
        for c in communities.keys():
            for n_t in communities[c]:
                node, time = n_t.split('-t')
                time = int(time)
                try:
                    new_comms[time][c].append(node)
                except:
                    new_comms[time][c] = [node]
        return new_comms

    def evaluate(self, ground_truth, method, name, eval, duration):
        # nmi = NMI.NMI(ground_truth, method).results
        nmi = NMI_lanlan.NMI(ground_truth, method).results
        # nmi = NMI(ground_truth, method)
        omega = Omega.Omega(ground_truth, method)
        bcubed = Bcubed.Bcubed(ground_truth, method)
        results = OrderedDict()
        # results["Method"] = [name]
        # results["Eval"] = [eval]
        # results['NMI'] = ["{0:.4f}".format(nmi['NMI<Max>'])]
        results['NMI'] = [float("{0:.3f}".format(nmi))]
        results['Omega'] = [float("{0:.3f}".format(omega.omega_score))]
        results['Bcubed-Recall'] = [float("{0:.3f}".format(bcubed.recall))]
        results['Bcubed-Precision'] = [float("{0:.3f}".format(bcubed.precision))]
        results['Bcubed-F1'] = [float("{0:.3f}".format(bcubed.fscore))]
        results['Duration'] = [duration]
        return results


    def get_results(self, ground_truth, method, name, tfs_len, eval="dynamic", duration=0):
        if eval == "dynamic":
            # print(ground_truth, method)
            results = self.evaluate(ground_truth, method, name, eval, duration)
        elif eval == "sets":
            new_comms1 = {i: set() for i in ground_truth.keys()}
            for i, comm in ground_truth.items():
                for node in comm:
                    new_comms1[i].add(node.split('-')[0])
            new_comms2 = {i: set() for i in method.keys()}
            for i, comm in method.items():
                for node in comm:
                    new_comms2[i].add(node.split('-')[0])
            results = self.evaluate(new_comms1, new_comms2, name, eval, duration)
        elif eval == "per_tf":
            self.history = defaultdict(list)
            new_comms1 = self.unravel_tf(ground_truth, tfs_len)
            new_comms2 = self.unravel_tf(method, tfs_len)
            # new_comms1 = times_comms(ground_truth)
            # new_comms2 = times_comms(method)

            per_tf = []
            per_tf_lanlan = []
            for t in range(tfs_len):
                # print(new_comms1[t])
                # print(new_comms2[t])
                try:
                    per_tf.append(Counter(self.evaluate(new_comms1[t], new_comms2[t], name, eval, duration)))
                    temp_lanlan = dict(self.evaluate(new_comms1[t], new_comms2[t], name, eval, duration))
                    per_tf_lanlan.append(temp_lanlan)
                except:
                    continue
            # print(per_tf_lanlan)
            # results = sum(per_tf, Counter())
            results_lanlan = defaultdict(list)
            for dic in per_tf_lanlan:
                for k in dic.keys():
                    results_lanlan[k].append(dic[k][0])
            results = results_lanlan
            # print(results)
            for key in results:
                # print(results[key])
                if all(isinstance(x, str) for x in results[key]):
                    results[key] = [results[key][0]]
                else:
                    results[key] = ["{0:.3f}".format(sum(results[key])/tfs_len)]
            # pprint.pprint(dict(f))
            # for k, v in res.iteritems():
            #     print "KEY ", k, " VALUE ", v
        return results

    def get_evaluate(self, comms_gt, comms, method, time_num=4, eval="dynamic"):
        all_res = []
        all_res.append(self.get_results(comms_gt, comms, method, time_num, eval=eval))  # eval == dynamic, 4 does not work.
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

def readnode(file):
    nodes = []
    with open(file, 'r') as fp:
        for line in fp.readlines():
            nodes.append(line.split('\n')[0])
    return nodes

def dyn_main(path, code=1, time_num=4, eval="dynamic", outer=None):
    file_gt = 'gt.communities'
    comms_gt = readlabel(os.path.join(path, file_gt))

    if outer != None:
        # for dynGEM
        path = outer

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
            file_sr = 'sr(sc_sigmoid0.01).communities'  # for test
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
    elif code == 19:
            # DynGEM
            file_dg = 'sr(kmeans)_16.communities'
            comms_dg = readlabel(os.path.join(path, file_dg))
            comms = comms_dg
            method = 'DynGEM'
    elif code == 20:
            # DynGEM
            file_dg = 'sr(kmeans)_20.communities'
            comms_dg = readlabel(os.path.join(path, file_dg))
            comms = comms_dg
            method = 'DynGEM'
    elif code == 21:
            # DynGEM
            file_dg = 'sr(kmeans)_32.communities'
            comms_dg = readlabel(os.path.join(path, file_dg))
            comms = comms_dg
            method = 'DynGEM'
    elif code == 22:
            # DynGEM
            file_dg = 'sr(kmeans)_64.communities'
            comms_dg = readlabel(os.path.join(path, file_dg))
            comms = comms_dg
            method = 'DynGEM'
    elif code == 23:
            # DynGEM
            file_dg = 'sr(kmeans)_128.communities'
            comms_dg = readlabel(os.path.join(path, file_dg))
            comms = comms_dg
            method = 'DynGEM'
    # evaluate and print
    evaluate = Metric()
    # print(comms_gt)
    # print(comms)
    evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)

def new_dyn_evaluate(path, method='tr', time_num=4, top=False, sub=None, eval="dynamic"):
    """
    evaluation
    :param path: data_folder/community
    :param method: 'tr'=TimeRank; 'tf'=NNTF; 'psb'=PisCES with filtering before;
                    'psa'=PisCES, filter after clustering; 'srsc'=SuRep with spectral clustering;
                    'srdb'= SuRep with DBSCAN; 'srkm'=SuRep with K-Means
    :return:
    """
    if top:
        file_cof = sub + '-top0.1.label'
        nodes = readnode(os.path.join(path, file_cof))
        print(nodes)
    else:
        nodes = None
    file_gt = 'gt.communities'
    comms_gt = readlabel(os.path.join(path, file_gt), nodes)
    our_path = path
    # path = os.path.dirname(path)
    if method == 'tr':
        folder = os.path.join(path, 'TimeRank')
        for file in os.listdir(folder):
            print(file+'------------------------')
            comms = readlabel(os.path.join(folder, file), nodes)
            evaluate = Metric()
            evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
    elif method == 'tf':
        folder = os.path.join(path, 'NNTF')
        for file in os.listdir(folder):
            print(file+'------------------------')
            comms = readlabel(os.path.join(folder, file), nodes)
            evaluate = Metric()
            evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
    elif method == 'psb':
        # folder = os.path.join(path, 'PS-new/before')
        folder = os.path.join(path, 'PSbefore')
        for file in os.listdir(folder):
            name, suff = os.path.splitext(file)
            if suff == '.mat':
                continue
            print(file+'------------------------')
            comms = readlabel(os.path.join(folder, file), nodes)
            evaluate = Metric()
            evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
    elif method == 'psa':
        # folder = os.path.join(path, 'PS-new/after')
        folder = os.path.join(path, 'PSafter')
        for file in os.listdir(folder):
            name, suff = os.path.splitext(file)
            if suff == '.mat':
                continue
            print(file+'------------------------')
            comms = readlabel(os.path.join(folder, file), nodes)
            evaluate = Metric()
            evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
    elif method == 'srsc':
        folder = os.path.join(our_path, 'SuRep-SC')
        for file in os.listdir(folder):
            print(file+'------------------------')
            comms = readlabel(os.path.join(folder, file), nodes)
            evaluate = Metric()
            evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
    elif method == 'srdb':
        folder = os.path.join(our_path, 'SuRep-DB')
        for file in os.listdir(folder):
            print(file+'------------------------')
            comms = readlabel(os.path.join(folder, file), nodes)
            evaluate = Metric()
            evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
    elif method == 'srkm':
        print('K-Means'+'------------------------')
        file = 'sr(kmeans).communities'
        comms = readlabel(os.path.join(our_path, file), nodes)
        evaluate = Metric()
        evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
    elif method == 'twosc':
        print('Two-step SC'+'------------------------')
        file = 'twoStep(sc).communities'
        comms = readlabel(os.path.join(our_path, file), nodes)
        evaluate = Metric()
        evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
    elif method == 'joint':
        folder = os.path.join(our_path, method)
        for file in os.listdir(folder):
            print(file+'------------------------')
            comms = readlabel(os.path.join(folder, file), nodes)
            evaluate = Metric()
            evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
    elif method == 'cosine':
        folder = os.path.join(our_path, method)
        for file in os.listdir(folder):
            print(file+'------------------------')
            comms = readlabel(os.path.join(folder, file), nodes)
            evaluate = Metric()
            evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
    elif method == 'surep':
        folder = os.path.join(our_path, method)
        for file in os.listdir(folder):
            print(file+'------------------------')
            comms = readlabel(os.path.join(folder, file), nodes)
            evaluate = Metric()
            evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
    elif method == 'none':
        folder = os.path.join(our_path, method)
        for file in os.listdir(folder):
            print(file+'------------------------')
            comms = readlabel(os.path.join(folder, file), nodes)
            evaluate = Metric()
            evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
    elif method == 'ENMF':
        folder = os.path.join(path, method)
        for file in os.listdir(folder):
            name, suff = os.path.splitext(file)
            if suff == '.mat':
                continue
            # print(file+'------------------------')
            comms = readlabel(os.path.join(folder, file), nodes)
            evaluate = Metric()
            evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
    elif method == 'CCPSO':
        folder = os.path.join(our_path, method)
        for file in os.listdir(folder):
            print(file+'------------------------')
            comms = readlabel(os.path.join(folder, file), nodes)
            evaluate = Metric()
            evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
    elif method == 'GDG':
        folder = os.path.join(path, method)
        for file in os.listdir(folder):
            name, suff = os.path.splitext(file)
            if suff == '.mat':
                continue
            print(file+'------------------------')
            comms = readlabel(os.path.join(folder, file), nodes)
            evaluate = Metric()
            evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)

def dyn_folder_evaluate(path, time_num, top=False, sub=None, eval="dynamic"):
    folder = os.path.join(path, 'best')
    if top:
        file_cof = sub + '-top0.1.label'
        nodes = readnode(os.path.join(path, file_cof))
        # print(nodes)
    else:
        nodes = None
    file_gt = 'gt.communities'
    comms_gt = readlabel(os.path.join(path, file_gt), nodes)

    for file in os.listdir(folder):
        if file == file_gt:
            continue
        print(file + '------------------------')
        comms = readlabel(os.path.join(folder, file), nodes)
        evaluate = Metric()
        evaluate.get_evaluate(comms_gt, comms, file.split('.')[0], time_num, eval=eval)

def dyn_evaluate_sub(path, method='tr', time_num=4, top=False, sub=None, k=None, cluster_num=None, eval="dynamic"):
    """
    evaluation
    :param path: data_folder/community
    :param method: 'tr'=TimeRank; 'tf'=NNTF; 'psb'=PisCES with filtering before;
                    'psa'=PisCES, filter after clustering; 'srsc'=SuRep with spectral clustering;
                    'srdb'= SuRep with DBSCAN; 'srkm'=SuRep with K-Means
    :return:
    """
    if top:
        file_cof = sub + '-top' + str(k) + '.label'
        nodes = readnode(os.path.join(path, file_cof))
        # print(nodes)
    else:
        nodes = None
    file_gt = 'gt.communities'
    comms_gt = readlabel(os.path.join(path, file_gt), nodes)
    our_path = path
    # path = os.path.dirname(path)
    if method == 'tra':
        file = os.path.join(path, 'tra.communities')
        comms = readlabel(file, nodes)
        evaluate = Metric()
        evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
    elif method == 'trn':
        file = os.path.join(path, 'trn.communities')
        comms = readlabel(file, nodes)
        evaluate = Metric()
        evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
    elif method == 'tf':
        file = os.path.join(path, 'tf.communities')
        comms = readlabel(file, nodes)
        # print(comms)
        evaluate = Metric()
        evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
    elif method == 'ps':
        file = os.path.join(path, 'ps.communities')
        comms = readlabel(file, nodes)
        evaluate = Metric()
        evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
    elif method == 'srs':
        file = os.path.join(our_path, 'srs.communities')
        comms = readlabel(file, nodes)
        evaluate = Metric()
        evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
    elif method == 'srkm':
        # print('K-Means'+'------------------------')
        file = 'sr(kmeans).communities'
        comms = readlabel(os.path.join(our_path, file), nodes)
        evaluate = Metric()
        evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
    elif method == 'two':
        # print('Two-step SC'+'------------------------')
        file = 'twoStep(sc).communities'
        comms = readlabel(os.path.join(our_path, file), nodes)
        evaluate = Metric()
        evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
    elif method == 'joint':
        folder = os.path.join(our_path, method)
        for file in os.listdir(folder):
            if file.split('(')[-1].split(')')[0] != 'kmeans':
                # print(file)
                continue
            try:
                if abs(cluster_num - int(file.split('_')[-1].split('.')[0])) <= 3:
                    print(cluster_num, '\t', file+'------------------------')
                    comms = readlabel(os.path.join(folder, file), nodes)
                    evaluate = Metric()
                    evaluate.get_evaluate(comms_gt, comms, method, time_num, eval=eval)
            except:
                continue




def stat_main(path):
    file_gt = 'gt.communities'
    comms_gt = readlabel(os.path.join(path, file_gt))

    file_sr = 'sr(kmeans).communities'
    comms_sr = readlabel(os.path.join(path, file_sr))
    comms = comms_sr
    method = "SuRep"

    # print(method, NMI(comms_gt, comms))
    all_res = []
    evaluate = Metric()

    all_res.append(evaluate.get_results(comms_gt, comms, method, 4, eval="dynamic"))
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
    # print(results)
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

    # -----------------------------------------------------------------------------------------------
    # 实验分割线
    # -----------------------------------------------------------------------------------------------

    # Reddit
    cluster_num = 3
    path = 'E:\\DATASet\\Reddit\\reddit\\2010-09(sampling ratio 1.0)\\R4-\\undirected\\community'
    outer = 'E:/DATASet/Reddit/reddit/2020-4-2/reddit4_4'
    time_num = 4
    # path = 'E:\\DATASet\\Reddit\\reddit\\2010-09-10(sampling ratio 1.0)\\R4-\\undirected\\community'
    # outer = 'E:/DATASet/Reddit/reddit/2020-4-2/reddit4_8'
    # time_num = 8

    # cluster_num = 2
    # path = 'E:\\DATASet\\Reddit\\reddit\\2010-09(sampling ratio 1.0)\\R0-\\undirected\\community'
    # outer = 'E:/DATASet/Reddit/reddit/2020-4-2/reddit0_4'
    # time_num = 4

    # cluster_num = 17
    # path = 'E:\\DATASet\\Reddit\\reddit\\2010-09(sampling ratio 1.0)\\R3-\\undirected\\community'
    # outer = 'E:/DATASet/Reddit/reddit/2020-4-2/reddit3_4'
    # time_num = 4
    # path = 'E:\\DATASet\\Reddit\\reddit\\2010-09-10(sampling ratio 1.0)\\R3-\\undirected\\community'
    # outer = 'E:/DATASet/Reddit/reddit/2020-4-2/reddit3_8'
    # time_num = 8

    # SBM

    # cluster_num = 4
    # path = 'G:\\CodeSet\\workspace\\HGCN\\sinmulateFordraw\\SBM\\community'
    # time_num = 4

    # cluster_num = 4
    # path = 'G:\\CodeSet\\workspace\\HGCN\\sinmulateFordraw\\figure7\\community'
    # time_num = 2
    # path = 'G:\\CodeSet\\workspace\\HGCN\\sinmulateFordraw\\test-figure7\\community'
    # outer = 'E:/DATASet/Reddit/reddit/2020-4-2/figure7_4'
    # time_num = 4
    # MIT
    # path = 'E:/DATASet/RealityMining/RealityMining/community'

    # ---------------------------------------------------------
    # choose top-k node for evaluate
    # ---------------------------------------------------------
    # methods = ['tf', 'tra', 'trn', 'ps', 'two', 'joint']
    # # top = False
    # # sub = 'deg'
    # sub = 'cof'
    # # k = 0.1
    # k = 0.001
    # top = True
    # top = False
    # sub = None
    # k = None
    # for method in methods:
    #     print(method)
    #     dyn_evaluate_sub(path, method=method, time_num=time_num, top=top, sub=sub, k=k, cluster_num=cluster_num, eval='per_tf')

    # top = True
    # # dyn_evaluate_sub(path, method='joint', time_num, top=False, cluster_num=cluster_num)
    # subs = ['deg', 'cof']
    # k = 0.1
    # for sub in subs:
    #     print(sub + '****************************')
    #     dyn_evaluate_sub(path, method='tf', time_num=time_num, top=top, sub=sub, k=k, cluster_num=cluster_num)

    # ---------------------------------------------------------
    # evalutaion for one method
    # ---------------------------------------------------------
    # print(path)
    # print('choose....')
    # print(path)

    # new_dyn_evaluate(path, method='cosine', time_num=time_num)

    # print('TimeRank')
    # new_dyn_evaluate(path, method='tr', time_num=time_num)

    # print('NNTF')
    # new_dyn_evaluate(path, method='tf', time_num=time_num)

    # print('PisCES, filtering before clustering')
    # new_dyn_evaluate(path, method='psb', time_num=time_num)
    # print('PisCES, filtering after clustering')
    # new_dyn_evaluate(path, method='psa', time_num=time_num)

    # path = path + '\joint'  # 'map'
    # print('SuRep + K-Means')
    # new_dyn_evaluate(path, method='srkm', time_num=time_num)
    # print('SuRep + Spectral Clustering')
    # new_dyn_evaluate(path, method='srsc', time_num)
    # print('SuRep + DBSCAN')
    # new_dyn_evaluate(path, method='srdb', time_num)

    # print('Two-Step SC')
    # new_dyn_evaluate(path, method='twosc', time_num=time_num, top=None, sub=None)
    # print('add, SuRep')
    # new_dyn_evaluate(path, method='surep', time_num=time_num, top=False, sub=None)
    # print('none, adjancy')
    # new_dyn_evaluate(path, method='none', time_num=time_num, top=False, sub=None)
    # print('ENMF')
    # new_dyn_evaluate(path, method='ENMF', time_num=time_num, top=False, sub=None)
    # print('ComSP')
    # new_dyn_evaluate(path, method='joint', time_num=time_num)
    # print('CCPSO')
    # new_dyn_evaluate(path, method='CCPSO', time_num=time_num)
    # print('GDG')
    # new_dyn_evaluate(path, method='GDG', time_num=time_num, eval='per_tf')

    # for folder
    # dyn_folder_evaluate(path, time_num, top=False, sub=None, eval="dynamic")
    dyn_folder_evaluate(path, time_num, top=False, sub=None, eval="per_tf")
    # print('cof')
    # dyn_folder_evaluate(path, time_num, top=True, sub='cof', eval="dynamic")
    # print('deg')
    # dyn_folder_evaluate(path, time_num, top=True, sub='deg', eval="dynamic")

    # alone evaluation

    # dyn_main(path, code=5, time_num)
    # print('kmeans')
    # dyn_main(path, code=9, time_num)
    # print('SpectralClustering')
    # dyn_main(path, code=10, time_num)
    # print('DBSCAN')
    # dyn_main(path, code=11, time_num)

    # print('SuRep all')
    # dyn_main(path, code=12, time_num)
    # print('SuRep next')
    # dyn_main(path, code=13, time_num)

    # print('nntf 1e-1')
    # dyn_main(path, code=1, time_num)
    # print('nntf 1e-2')
    # dyn_main(path, code=2, time_num)
    # print('nntf 1e-3')
    # dyn_main(path, code=3, time_num)
    # print('nntf 1e-4')
    # dyn_main(path, code=4, time_num)

    # print('timerank-noc-u')
    # dyn_main(path, code=6, time_num)
    # print('timerank-aoc-u')
    # dyn_main(path, code=5, time_num)
    # print('timerank-noc')
    # dyn_main(path, code=8, time_num)
    # print('timerank-aoc')
    # dyn_main(path, code=7, time_num)


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

    # DynGEM
    # print('DynGEM')
    # for code in range(19, 24):
    #     dyn_main(path=path, code=code, time_num=time_num, outer=outer)

    # # Karate
    # path = 'E:/DATASet/node classify/karate/communities'
    #
    # stat_main(path)


    # -----------------------------------------------------------------------------------------------
    # 检验分割线
    # -----------------------------------------------------------------------------------------------
    # gt = {0: ["1-t0", "5-t0", "3-t0", "7-t0",
    #           "1-t1", "2-t1", "3-t1", "4-t1",
    #           "6-t2", "2-t2", "8-t2", "4-t2"],
    #       1: ["2-t0", "6-t0", "4-t0", "8-t0",
    #           "5-t2", "1-t2", "7-t2", "3-t2"]}
    #
    # ct = {0: ["1-t0", "2-t0", "3-t0", "4-t0",
    #           "1-t1", "2-t1", "3-t1", "4-t1",
    #           "1-t2", "2-t2", "3-t2", "4-t2"],
    #       1: ["5-t0", "6-t0", "7-t0", "8-t0",
    #           "5-t2", "6-t2", "7-t2", "8-t2"]}
    #
    # get_evaluate(gt, ct, 'origin', eval="dynamic")

    # n1 = {0: ["1-t0", "2-t0", "3-t0", "4-t0",
    #           "1-t1", "2-t1", "3-t1", "4-t1"],
    #       2: ["1-t2", "2-t2", "3-t2", "4-t2"],
    #       1: ["5-t0", "6-t0", "7-t0", "8-t0",
    #           "5-t2", "6-t2", "7-t2", "8-t2"
    #           ]}
    # get_evaluate(gt, n1, 'number', eval="dynamic")
    # n2 = {0: ["1-t0", "2-t0", "3-t0", "4-t0",
    #           "1-t1", "2-t1", "3-t1", "4-t1"],
    #       2: ["1-t2", "2-t2", "3-t2", "4-t2"],
    #       1: ["5-t0", "6-t0", "7-t0", "8-t0"],
    #       3: ["5-t2", "6-t2", "7-t2", "8-t2"]}
    # get_evaluate(gt, n2, 'number', eval="dynamic")
    #
    # s1 = {0: ["1-t0", "2-t0", "3-t0", "4-t0",
    #           "1-t1", "2-t1", "3-t1", "4-t1",
    #           "1-t2", "2-t2", "3-t2", "4-t2",
    #           "5-t2", "6-t2","7-t2", "8-t2"
    #           ],
    #       1: [
    #           "5-t0", "6-t0", "7-t0", "8-t0"
    #           ]}
    # get_evaluate(gt, s1, 'size', eval="dynamic")
    # s2 = {0: [
    #           "1-t0", "2-t0", "3-t0", "4-t0",
    #           "1-t1", "2-t1", "3-t1", "4-t1",
    #           "1-t2", "2-t2", "3-t2", "4-t2",
    #           "5-t2", "6-t2", "7-t2", "8-t2",
    #           "5-t0", "6-t0", "7-t0", "8-t0"
    #         ]}
    # get_evaluate(gt, s2, 'size', eval="dynamic")
    #
    # sub1 = {0: ["1-t0", "2-t0", "3-t0", "4-t0",
    #             "1-t1", "2-t1", "3-t1", "4-t1"],
    #         1: ["5-t0", "6-t0", "7-t0", "8-t0",
    #             "5-t2", "6-t2", "7-t2", "8-t2"]}
    # get_evaluate(gt, sub1, 'subset', eval="dynamic")
    # sub2 = {0: ["1-t0", "2-t0", "3-t0", "4-t0",
    #             "1-t1", "2-t1", "3-t1", "4-t1"],
    #         1: ["5-t2", "6-t2", "7-t2", "8-t2"]}
    # get_evaluate(gt, sub2, 'subset', eval="dynamic")

    # ns1 = {0: ["1-t0", "2-t0", "3-t0", "4-t0",
    #           "1-t1", "2-t1", "3-t1", "4-t1",
    #           "3-t2", "4-t2"],
    #       2: ["1-t2", "2-t2", "5-t0", "6-t0", ],
    #       1: ["7-t0", "8-t0",
    #           "5-t2", "6-t2", "7-t2", "8-t2"]}
    # get_evaluate(gt, ns1, 'subset', eval="dynamic")
    #
    # ns2 = {0: ["1-t0", "2-t0", "3-t0", "4-t0",
    #           "1-t1", "2-t1", "3-t1", "4-t1",
    #           ],
    #       2: ["1-t2", "2-t2", "5-t0", "6-t0", "7-t0", "8-t0", "3-t2", "4-t2"],
    #       1: [
    #           "5-t2", "6-t2", "7-t2", "8-t2"]}
    # get_evaluate(gt, ns2, 'subset', eval="dynamic")