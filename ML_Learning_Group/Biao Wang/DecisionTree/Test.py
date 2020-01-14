import numpy as np
import pandas as pd
import math

_746Data = '../LogisticRegression/newHIV-1_data/746Data.txt'
_1625Data = '../LogisticRegression/newHIV-1_data/1625Data.txt'
_impensData = '../LogisticRegression/newHIV-1_data/impensData.txt'
_schillingData = '../LogisticRegression/newHIV-1_data/schillingData.txt'
_x_all = 'ARNDCQEGHILKMFPSTWYV'

labels = ['index1', 'index2', 'index3', 'index4',
          'index5', 'index6', 'index7', 'index8', 'label']

def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况', '类别']
    # 返回数据集和每个维度的名称
    return datasets, labels

def loadData_2(data_path):
    # 用那个基因数据集
    data = pd.read_csv(data_path, header=None)
    X_train = data.values[:, 0]
    Y_train = data.values[:, 1]
    X_train_list = []
    # 需要对属性进行数值化
    for i in range(len(X_train)):
        X_train_list.append([a for a in X_train[i]])
        # X_train_list.append([_x_all.index(a) + 1 for a in X_train[i]])  # 从1到20进行编号, 字符需要转化为数值,
    X_train = np.array(X_train_list)
    for i in range(len(Y_train)):
        if Y_train[i] == -1:
            Y_train[i] = 'non-cleaved'
        else:
            Y_train[i] = 'cleaved'
    _X = np.hstack((X_train, np.array(Y_train).reshape((len(Y_train), 1))))
    # print('_X:', _X)
    df = pd.DataFrame(data=_X, columns=labels)
    # print('df:', df)
    return _X, df, X_train

# 定义节点类 二叉树
class Node:
    def __init__(self, root=True, label=None, feature_name=None, feature=None):
        self.root = root
        self.label = label  # 节点的类别
        self.feature_name = feature_name  # 特征名
        self.feature = feature  # 特征下标
        self.tree = {}  # 保存一个二叉树
        self.result = {
            'label:': self.label,
            'feature': self.feature,
            'tree': self.tree
        }

    # 直接调用的是这个函数
    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self, val, node):
        self.tree[val] = node

    def predict(self, features):
        if self.root is True:
            # print('root:', self.root)
            # print(self.label)
            return self.label
        # print('False的情形:', self.tree[features[self.feature]].predict(features))
        return self.tree[features[self.feature]].predict(features)

class DTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    # 计算熵
    @staticmethod
    def calc_ent(datasets):
        data_length = len(datasets)
        label_count = {}
        for i in range(data_length):
            label = datasets[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        ent = -sum([(p / data_length) * math.log(p / data_length, 2)
                    for p in label_count.values()])
        return ent

    # 经验条件熵
    def cond_ent(self, datasets, axis=0):  # axis选择特征
        data_length = len(datasets)
        feature_sets = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(datasets[i])
        cond_ent = sum([(len(p) / data_length) * self.calc_ent(p) for p in feature_sets.values()])
        return cond_ent

# 信息增益
    @staticmethod
    def info_gain(ent, cond_ent):
        return ent - cond_ent

    def info_gain_train(self, datasets):
        count = len(datasets[0]) - 1  # 特征数
        ent = self.calc_ent(datasets)
        best_feature = []  # 保存特征下标及该特征下的信息增益
        for c in range(count):
            c_info_gain = self.info_gain(ent, self.cond_ent(datasets, axis=c))
            best_feature.append((c, c_info_gain))
        # 比较大小
        best_ = max(best_feature, key=lambda x: x[-1])  # 返回的是特征下标及该特征下对应的信息增益
        return best_

    def train(self, train_data):
        """
        input:数据集D(DataFrame格式)，特征集A，阈值eta
        output:决策树T
        """
        _, y_train, features = train_data.iloc[:, :-1], \
                               train_data.iloc[:, -1], \
                               train_data.columns[:-1]
        '''_为数据的属性值，y_train为数据的标签值， features为数据的属性名'''
        # print(y_train.value_counts()['cleaved'])
        # cleaved      402
        # non-cleaved  344
        # 1,若D中实例属于同一类Ck，则T为单节点树，并将类Ck作为结点的类标记，返回T
        if len(y_train.value_counts()) == 1:
            return Node(root=True, label=y_train.iloc[0])

        # 2, 若A为空，则T为单节点树，将D中实例树最大的类Ck作为该节点的类标记，返回T
        if len(features) == 0:
            return Node(root=True,
                        label=y_train.value_counts().sort_values(ascending=False).index[0])
        # 上述的label就是样本数比较多的类
        # test = y_train.value_counts().sort_values(ascending=False)

        # 3,计算最大信息增益 同5.1,Ag为信息增益最大的特征
        max_feature, max_info_gain = self.info_gain_train(np.array(train_data))
        max_feature_name = features[max_feature]

        # 4,Ag的信息增益小于阈值eta,则置T为单节点树，并将D中是实例数最大的类Ck作为该节点的类标记，返回T
        if max_info_gain < self.epsilon:  # 预剪枝操作
            return Node(root=True,
                        label=y_train.value_counts().sort_values(ascending=False).index[0])

        # 5,构建Ag子集
        node_tree = Node(root=False,
                         feature_name=max_feature_name, feature=max_feature)
        feature_list = train_data[max_feature_name].value_counts().index
        # 统计这个属性的各个取值
        # Index(['L', 'Y', 'F', 'S', 'N', 'A', 'V', 'M', 'G', 'W', 'T', 'K', 'C', 'R',
        #       'Q', 'I', 'D', 'H', 'E', 'P']

        # 遍历max_feature下的所有取值
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name]
                                          == f].drop([max_feature_name], axis=1)
            # print(train_data.loc[train_data[max_feature_name] == f])
            # exit()
            # 6, 递归生成树
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)
        return node_tree

    def fit(self, train_data):
        self._tree = self.train(train_data)
        return self._tree

    def predict(self, X_test):
        return self._tree.predict(X_test)

if __name__ == '__main__':
    '''
    datasets, labels = create_data()
    dataframe = pd.DataFrame(data=datasets, columns=labels)
    dic = Node(label='是')
    print(dic)
    exit()
    '''

    '''datasets以及dataframe都包含有数据标签信息'''
    datasets, dataframe, data_no_label = loadData_2(_746Data)  # 数组, datafram, 训练集
    # datasets_test, dataframe_test, data_no_label_test = loadData_2(_1625Data)  # 测试集
    datasets_test, dataframe_test, data_no_label_test = loadData_2(_impensData)  # 测试集
    # datasets_test, dataframe_test = loadData_2(_schillingData)  # 测试集

    print('labels:', labels)
    print('datasets:', datasets)
    print('dataframe:', dataframe)
    print('---------------------------------------')
    print('index_1', set(list(dataframe['index1'].values)))
    print('index_2', set(list(dataframe['index2'].values)))
    print('index_3', set(list(dataframe['index3'].values)))
    print('index_4', set(list(dataframe['index4'].values)))
    print('index_5', set(list(dataframe['index5'].values)))
    print('index_6', set(list(dataframe['index6'].values)))
    print('index_7', set(list(dataframe['index7'].values)))
    print('index_8', set(list(dataframe['index8'].values)))
    print('---------------------------------------')


    dt = DTree()
    tree = dt.fit(dataframe)
    true_label = list(dataframe['label'].values)
    print('true_label:', true_label)
    print('------predict------')
    pre_label = []
    for data in data_no_label:
        print('---------')
        print(data)
        label_pre = dt.predict(list(data))
        print(label_pre)
        print('---------')
        pre_label.append(label_pre)

    # 计算准确率
    total = len(true_label)
    right = 0
    for i in range(total):
        if true_label[i] == pre_label[i]:
            right += 1
    print('accuracy:', right/total)


