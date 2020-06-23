import bcubed

class Bcubed:
    def __init__(self, truth, coms):
        self.ground_truth = self.process_input(truth)
        self.communities = self.process_input(coms)
        # print(self.ground_truth)
        # FIXME: fix keyerror for nodes not in ground truth
        try:
            self.precision = bcubed.precision(self.communities, self.ground_truth)
            self.recall = bcubed.recall(self.communities, self.ground_truth)
            self.fscore = bcubed.fscore(self.precision, self.recall)
        except KeyError:
            self.precision = 0
            self.recall = 0
            self.fscore = 0

    def process_input(self, coms):
        itemset = {}
        for id, com_list in coms.items():
            for node in com_list:
                try:
                    itemset[node].add(id)
                except KeyError:
                    itemset[node] = set([id])
        return itemset

def readlabel(file):
        labels_dic = {}
        with open(file, 'r') as fp:
            for line in fp.readlines():
                lines = line.split(':')
                cid = float(lines[0].split()[-1])
                # print(cid)
                labels_dic[cid] = []
                temp1 = lines[-1].split(']')[0].split('[')[-1]
                nodes = temp1.split()
                for node in nodes:
                    # print(node)
                    if len(node) > 1:  # is string of node's name
                        labels_dic[cid].append(node.split(',')[0])
        return labels_dic

if __name__=='__main__':
    # coms = {0: ['11-t1', '12-t1', '13-t1'],
    #         1: ['1-t0',
    #             '2-t0',
    #             '3-t0',
    #             '4-t0',
    #             '1-t1',
    #             '2-t1',
    #             '3-t1',
    #             '4-t1',
    #             '1-t2',
    #             '2-t2',
    #             '3-t2',
    #             '4-t2'],
    #         2: ['5-t0', '6-t0', '7-t0', '5-t2', '6-t2', '7-t2']}
    #
    # coms1 = {23: ['11-t1', '12-t1', '13-t1','1-t0'],
    #          50: ['1-t0',
    #               '2-t0',
    #               '3-t0',
    #               '4-t0',
    #               '1-t1',
    #               '2-t1',
    #               '3-t1',
    #               '4-t1',
    #               '1-t2',
    #               '2-t2',
    #               '3-t2',
    #               '4-t2'],
    #          2: ['5-t0', '6-t0', '7-t0', '5-t2', '6-t2', '7-t2']}
    # b = Bcubed(coms, coms1)
    # print(b.fscore)
    import os
    # path = 'E:/DATASet/node classify/karate/communities'
    # path = 'E:/DATASet/node classify/cora/communities'
    path = 'E:/DATASet/node classify/email-EuAll/communities'

    ground_truth = readlabel(os.path.join(path, 'ground_truth.label'))
    print('\n******************** k_clique **********************')
    for k in range(2, 10):
        print(k)
        file_non = os.path.join(path, str(k) + '_clique.label')
        file_sr = os.path.join(path, str(k) + '_clique_surep.label')
        print('method\tprecision\trecall\tfscore')
        try:
            comms_non = readlabel(file_non)
            b_non = Bcubed(ground_truth, comms_non)
            print('before\t', "{0:.4f}".format(b_non.precision), '\t', "{0:.4f}".format(b_non.recall), '\t', "{0:.4f}".format(b_non.fscore))
        except:
            print(file_non + ' is not existent.')
        try:
            comms_sr = readlabel(file_sr)
            b_sr = Bcubed(ground_truth, comms_sr)
            print('after\t', "{0:.4f}".format(b_sr.precision), '\t', "{0:.4f}".format(b_sr.recall), '\t', "{0:.4f}".format(b_sr.fscore))
        except:
            print(file_sr + ' is not existent.')


    print('\n******************** greedy_modularity **********************')
    file_non = os.path.join(path, 'greedy_modularity.label')
    file_sr = os.path.join(path, 'greedy_modularity_surep.label')
    print('method\tprecision\trecall\tfscore')
    try:
        comms_non = readlabel(file_non)
        b_non = Bcubed(ground_truth, comms_non)
        print('before\t', "{0:.4f}".format(b_non.precision), '\t', "{0:.4f}".format(b_non.recall), '\t', "{0:.4f}".format(b_non.fscore))
    except:
        print(file_non + ' is not existent.')
    try:
        comms_sr = readlabel(file_sr)
        b_sr = Bcubed(ground_truth, comms_sr)
        print('after\t', "{0:.4f}".format(b_sr.precision), '\t', "{0:.4f}".format(b_sr.recall), '\t', "{0:.4f}".format(b_sr.fscore))
    except:
        print(file_sr + ' is not existent.')

    print('\n******************** label_propagation **********************')
    file_non = os.path.join(path, 'label_propagation.label')
    file_sr = os.path.join(path, 'label_propagation_surep.label')
    print('method\tprecision\trecall\tfscore')
    try:
        comms_non = readlabel(file_non)
        b_non = Bcubed(ground_truth, comms_non)
        print('before\t', "{0:.4f}".format(b_non.precision), '\t', "{0:.4f}".format(b_non.recall), '\t', "{0:.4f}".format(b_non.fscore))
    except:
        print(file_non + ' is not existent.')
    try:
        comms_sr = readlabel(file_sr)
        b_sr = Bcubed(ground_truth, comms_sr)
        print('after\t', "{0:.4f}".format(b_sr.precision), '\t', "{0:.4f}".format(b_sr.recall), '\t', "{0:.4f}".format(b_sr.fscore))
    except:
        print(file_sr + ' is not existent.')

    print('\n******************** girvan_newman **********************')
    file_non = os.path.join(path, 'girvan_newman.label')
    file_sr = os.path.join(path, 'girvan_newman_surep.label')
    print('method\tprecision\trecall\tfscore')
    try:
        comms_non = readlabel(file_non)
        b_non = Bcubed(ground_truth, comms_non)
        print('before\t', "{0:.4f}".format(b_non.precision), '\t', "{0:.4f}".format(b_non.recall), '\t', "{0:.4f}".format(b_non.fscore))
    except:
        print(file_non + ' is not existent.')
    try:
        comms_sr = readlabel(file_sr)
        b_sr = Bcubed(ground_truth, comms_sr)
        print('after\t', "{0:.4f}".format(b_sr.precision), '\t', "{0:.4f}".format(b_sr.recall), '\t', "{0:.4f}".format(b_sr.fscore))
    except:
        print(file_sr + ' is not existent.')

    print('\n******************** asyn_fluidc **********************')
    file_non = os.path.join(path, 'asyn_fluidc.label')
    file_sr = os.path.join(path, 'asyn_fluidc_surep.label')
    print('method\tprecision\trecall\tfscore')
    try:
        comms_non = readlabel(file_non)
        b_non = Bcubed(ground_truth, comms_non)
        print('before\t', "{0:.4f}".format(b_non.precision), '\t', "{0:.4f}".format(b_non.recall), '\t', "{0:.4f}".format(b_non.fscore))
    except:
        print(file_non + ' is not existent.')
    try:
        comms_sr = readlabel(file_sr)
        b_sr = Bcubed(ground_truth, comms_sr)
        print('after\t', "{0:.4f}".format(b_sr.precision), '\t', "{0:.4f}".format(b_sr.recall), '\t', "{0:.4f}".format(b_sr.fscore))
    except:
        print(file_sr + ' is not existent.')