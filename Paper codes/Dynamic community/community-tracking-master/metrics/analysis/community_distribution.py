# from __future__ import division

from collections import Counter, OrderedDict, defaultdict
import os
# import matplotlib
# matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt
import numpy as np
from NMI_lanlan import NMI


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

# ----------------------------------------------------------------------
# comparision communities between the continuous timestamp
# ----------------------------------------------------------------------
def get_time(labels_dict):
    times = defaultdict(dict)
    for c in labels_dict.keys():
        for n_t in labels_dict[c]:
            node, time = n_t.split('-t')
            time = int(time)
            try:
                times[time][c].append(node)
            except:
                times[time][c] = [node]

    return times

def compare(time_labels, labels):
    times = sorted([t for t in time_labels.keys()])
    time_num = len(times)
    nmi = []
    for t in range(time_num):
        comms1 = labels[t]
        comms2 = time_labels[t]
        # print(t)
        # print(comms1)
        # print(comms2)
        nmi.append(NMI(comms1, comms2).results)
    return nmi

def visual_nmi(path, method_list, labels, title):
    gt = get_time(method_list[-1])
    l = len(method_list)
    mark = ['^', '*', '|', 'x']
    plt.figure()
    i = 0
    for labels_dict in method_list:
        print(labels_dict)
        times_comms = get_time(labels_dict)
        nmi = compare(times_comms, gt)
        plt.plot(nmi, marker=mark[i], label=labels[i])
        i += 1
        if i == l - 1:
            break
    plt.title(title)
    plt.legend()
    plt.xticks(range(len(gt)), [i+1 for i in list(gt.keys())])
    plt.xlabel('TimeStamp')
    plt.ylabel('NMI')
    plt.title(title)
    plt.savefig(os.path.join(path, 'nmi.png'))
    # plt.show()


# ----------------------------------------------------------------------
# statics for distribution of size of communities
# ----------------------------------------------------------------------

def stat(labels_dict):
    """
    X: count of items, Y: count of communities with X(i) items
    :param labels_dict: {commsID: [items]}
    :return: X, Y
    """
    count = [len(labels_dict[c]) for c in labels_dict.keys()]
    return count

def draw(comms_dict, labels, title):
    """
    draw communities distribution of dataset, detected by each methods
    :param comms_dict: {method: labels_dict}
    :return: None
    """
    # font = {'family': 'Times New Roman'}
    plt.figure(figsize=(9, 6))
    counts = []
    i = 0
    s = 1
    for labels_dict in comms_dict:
        count = np.array(stat(labels_dict))
        # if i == 0:
        #     s = sum(count)*1.0
        counts.append(count/s)
        print(labels[i])
        print(np.median(sorted(count)))
        median = np.median(count)
        i += 1

    bplot = plt.boxplot(counts, labels=labels, sym='k+',  # 异常值
                        boxprops={'linewidth': 2},  # 箱体
                        medianprops={'linewidth': 2},   # 中位数, 'color': '#1F77B4'
                        capprops={'linewidth': 2},  # 两端横线, 'color': '#1F77B4'
                        whiskerprops={'linewidth': 2},  # 连接线, 'color': '#1F77B4'
                        vert=False)#patch_artist=True,
    colors = ['k' for c in range(i-1)] + ['#D62728']
    print(colors)
    # for patch, color in zip(bplot['whiskers'], colors):
    #     patch.set_color(color)
    # for patch, color in zip(bplot['caps'], colors):
    #     patch.set_color(color)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_color(color)
    # colors = ['#1F77B4' for c in range(i-1)] + ['#D62728']
    for patch, color in zip(bplot['medians'], colors):
        patch.set_color(color)

    # plt.grid(True, axis='y')

    plt.axvline(median, ls='--', c='#D62728', linewidth=2)
    plt.plot()
    plt.title(title, fontsize=23, family='Times New Roman')
    # plt.xlabel('Methods', fontsize=15, family='Times New Roman')
    # plt.xticks(range(len(labels)+1), ['']+labels, fontsize=24, family='Times New Roman')
    plt.yticks(fontsize=23, family='Times New Roman')
    plt.xticks(fontsize=16, family='Times New Roman')
    plt.xlabel('Size of Community', fontsize=24, family='Times New Roman')
    # plt.legend(fontsize=13, family='Times New Roman')
    # plt.xticks()
    # plt.yticks()
    # 设置坐标标签字体大小
    # plt.set_xlabel(..., fontsize=20)
    # plt.set_ylabel(..., fontsize=20)
    # 设置图例字体大小
    # plt.savefig(os.path.join(path, 'statics.png'))
    plt.savefig('./distribution_fig1/'+title+'.eps', format='eps', dpi=1000)
    # plt.close()
    # plt.show()

def read_data(path):
    print('read_data')
    method_list = []
    # each file is the best result in table
    # TimeRank-NOC
    trn_comms_dict = readlabel(os.path.join(path, 'trn.communities'))
    method_list.append(trn_comms_dict)
    # # TimeRank-AOC
    # tra_comms_dict = readlabel(os.path.join(path, 'tra.communities'))
    # method_list.append(tra_comms_dict)
    # # NNTF
    # tf_comms_dict = readlabel(os.path.join(path, 'tf.communities'))
    # method_list.append(tf_comms_dict)
    # PisCES
    ps_comms_dict = readlabel(os.path.join(path, 'ps.communities'))
    method_list.append(ps_comms_dict)
    # TwoStep Spectral Clustering
    two_comms_dict = readlabel(os.path.join(path, 'twoStep(sc).communities'))
    method_list.append(two_comms_dict)
    # DynGEM
    two_comms_dict =readlabel(os.path.join(path, 'dg.communities'))
    method_list.append(two_comms_dict)
    # SuRep K-Means
    srk_comms_dict = readlabel(os.path.join(path, 'srs.communities'))
    method_list.append(srk_comms_dict)
    # groundtruth
    gt_comms_dict = readlabel(os.path.join(path, 'gt.communities'))
    method_list.append(gt_comms_dict)
    return method_list

def draw_main(path, title):
    labels = ['TR-NOC', 'TR-AOC', 'NNTF', 'PisCES', 'TwoSC', 'DyCD', 'GroundTruth']
    method_list = read_data(path)
    # draw(path, method_list, labels, title)
    visual_nmi(path, method_list, labels, title)

def new_read_data(path, labels):
    print('read_data')
    method_list = []
    for method in labels:
        if method == 'TS':
            file_path = os.path.join(path, 'twoStep(sc).communities')
        elif method == 'GD':
            file_path = os.path.join(path, 'gdg.communities')
        elif method == 'PS':
            file_path = os.path.join(path, 'pisces(after).communities')
        elif method == 'SE':
            file_path = os.path.join(path, 'enmf.communities')
        elif method == 'CC':
            file_path = os.path.join(path, 'ccpso.communities')
        elif method == 'TR':
            file_path = os.path.join(path, 'trn.communities')
        elif method == 'AM':
            file_path = os.path.join(path, 'adj(kmeans).communities')
        elif method == 'SR':
            file_path = os.path.join(path, 'surep(kmeans).communities')
        elif method == 'CP':
            file_path = os.path.join(path, 'comsp(kmeans).communities')
        elif method == 'GT':
            file_path = os.path.join(path, 'gt.communities')

        method_list.append(readlabel(file_path))
    return method_list

def draw_main_new(path, title):
    # labels = ['TR', 'PS', 'TS', 'DG', 'CS', 'GT']
    labels = ['TS', 'GD', 'PS', 'SE', 'CC', 'TR', 'AM', 'SR', 'CS', 'GT']
    print(labels)
    # method_list = read_data(path)
    method_list = new_read_data(path, labels)
    draw(method_list, labels, title)
    # visual_nmi(path, method_list, labels, title)

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
    path = 'E:\\DATASet\\Reddit\\reddit\\2010-09(sampling ratio 1.0)\\R4-\\undirected\\community\\best'
    title = 'Reddit-I(a)'
    # path = 'E:\\DATASet\\Reddit\\reddit\\2010-09-10(sampling ratio 1.0)\\R4-\\undirected\\community\\best'
    # title = 'Reddit-I(b)'
    # path = 'E:\\DATASet\\Reddit\\reddit\\2010-09(sampling ratio 1.0)\\R0-\\undirected\\community\\best'
    # title = 'Reddit-II'
    # path = 'E:\\DATASet\\Reddit\\reddit\\2010-09(sampling ratio 1.0)\\R3-\\undirected\\community\\best'
    # title = 'Reddit-III(a)'
    # path = 'E:\\DATASet\\Reddit\\reddit\\2010-09-10(sampling ratio 1.0)\\R3-\\undirected\\community\\best'
    # title = 'Reddit-III(b)'

    # SBM
    # path = 'G:\\CodeSet\\workspace\\HGCN\\sinmulateFordraw\\figure7\\community'
    # path = 'G:\\CodeSet\\workspace\\HGCN\\sinmulateFordraw\\test-figure7\\community'

    # SBM1000
    # path = 'G:\\CodeSet\\workspace\\HGCN\\sinmulateFordraw\\SBM\\community\\best'
    # title = 'SBM'
    # MIT
    # path = 'E:/DATASet/RealityMining/RealityMining/community'

    # draw_main(path, title)
    print(path)
    draw_main_new(path, title)