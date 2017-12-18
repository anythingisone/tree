#coding:utf8
from math import log
import operator

def calcShannonEnt(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    # 为所有可能分类创建字典
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

# 建立数据集
def creatDataSet():
    dataset = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no'],
               ]
    labels = ['no surfacing','flippers']
    return dataset,labels

mydata,labels = creatDataSet()
shannonEnt = calcShannonEnt(mydata)
# print shannonEnt

# 划分数据集
def splitDataSet(dataset,axis,value):
    # 创建新的list对象,防止修改原数据
    retDataSet = []
    for featVec in dataset:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

splitData = splitDataSet(mydata,0,1)
print splitData

# 遍历整个数据集,选取最好的数据集划分方式
def chooseBestFeattureTosplit(dataset):
    numFeatures = len(dataset[0]) - 1
    baseEntropy = calcShannonEnt(dataset)
    bestInfoGain = 0.0;bestFeatture = -1
    for i in range(numFeatures):
        # 创建唯一分类标签列表
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 计算每种划分方式信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataset,i,value)
            prob = len(subDataSet)/float(len(dataset))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        # 计算最优信息增益
        if (infoGain >bestInfoGain):
            bestInfoGain = infoGain
            bestFeatture = i
    return bestFeatture
# print (chooseBestFeattureTosplit(mydata))

# 递归构建决策树
def majorityCnt(classList):
    classCount = {}
    for vote in classCount:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClasscount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse = True)
    return sortedClasscount[0][0]

def createTree(dataSet,labels):
    # 递归结束第一个条件:类别完全相同则停止划分
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 递归结束第二个条件:所有特征都用完,仍无法划分成仅包含同一类别的分组,返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeattureTosplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

# print(createTree(mydata,labels))
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearTate']
lensesTree = createTree(lenses,lensesLabels)
print(lensesTree)