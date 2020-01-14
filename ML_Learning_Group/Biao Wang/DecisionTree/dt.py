from math import log
import operator
import numpy as np
import pandas as pd

_746Data = '../LogisticRegression/newHIV-1_data/746Data.txt'
_1625Data = '../LogisticRegression/newHIV-1_data/1625Data.txt'
_impensData = '../LogisticRegression/newHIV-1_data/impensData.txt'
_schillingData = '../LogisticRegression/newHIV-1_data/schillingData.txt'
labels = ['index1', 'index2', 'index3', 'index4',
          'index5', 'index6', 'index7', 'index8', 'label']

def calcShannonEnt(dataSet):  # 计算数据的熵(entropy)
    numEntries=len(dataSet)  # 数据条数
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1] # 每行数据的最后一个字（类别）
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1  # 统计有多少个类以及每个类的数量
    shannonEnt=0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries # 计算单个类的熵值
        shannonEnt-=prob*log(prob,2) # 累加每个类的熵值
    return shannonEnt

def createDataSet1():    # 创造示例数据
    dataSet = [['长', '粗', '男'],
               ['短', '粗', '男'],
               ['短', '粗', '男'],
               ['长', '细', '女'],
               ['短', '细', '女'],
               ['短', '粗', '女'],
               ['长', '粗', '女'],
               ['长', '粗', '女']]
    labels = ['头发','声音']  #两个特征
    return dataSet,labels

def splitDataSet(dataSet,axis,value): # 按某个特征分类后的数据
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec =featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):  # 选择最优的分类特征
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)  # 原始的熵
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob =len(subDataSet)/float(len(dataSet))
            newEntropy +=prob*calcShannonEnt(subDataSet)  # 按特征分类后的熵
        infoGain = baseEntropy - newEntropy  # 原始熵与按特征分类后的熵的差值
        if (infoGain>bestInfoGain):   # 若按某特征划分后，熵值减少的最大，则次特征为最优分类特征
            bestInfoGain=infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):    #按分类后类别数量排序，比如：最后分类为2男1女，则判定为男；
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]  # 类别：男或女
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet) #选择最优特征
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}} #分类结果以字典形式保存
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet\
                            (dataSet,bestFeat,value),subLabels)
    return myTree

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

def predict(x, tree):
    _labels = ['index1', 'index2', 'index3', 'index4',
               'index5', 'index6', 'index7', 'index8']
    x = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'F']

    print('tree', tree)


    return


if __name__=='__main__':
    # dataSet, labels=createDataSet1()  # 创造示列数据
    datasets, dataframe, _datasets = loadData_2(_746Data)  # 数组, datafram, 训练集
    # datasets_test, dataframe_test, data_no_label_test = loadData_2(_1625Data)  # 测试集
    datasets_test, dataframe_test, _datasets_test = loadData_2(_impensData)  # 测试集
    # datasets_test, dataframe_test = loadData_2(_schillingData)  # 测试集
    _labels = ['index1', 'index2', 'index3', 'index4',
              'index5', 'index6', 'index7', 'index8']
    print(datasets)
    _data = []
    for data in datasets:
        _data.append(list(data))
    tree = createTree(_data, _labels)  # 输出决策树模型结果

    # 进行预测
    predict(['A','R','N','D','C','Q', 'E', 'F'], tree)