# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:57:23 2024

@author: Jerome
"""

from math import log
import operator


# 创建演示数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # 读取数据集中所有的可能性
    for featVec in dataSet: #the the number of unique elements and their occurance
        #获取对应事件 
        currentLabel = featVec[-1]
        # 第一次获取
        if currentLabel not in labelCounts.keys(): 
            labelCounts[currentLabel] = 1
        else:
            labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        # 求取事件的可能性
        prob = float(labelCounts[key])/numEntries
        # 求以2为底的对数
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt


# 划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        # 定位数据集中axis列数据为value的值
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            # 延展列表，相当于去掉了axis列
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    #最后一列存放标签
    numFeatures = len(dataSet[0]) - 1      
    # 计算原始香农熵
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    # 遍历所有特征
    for i in range(numFeatures):        
        # 获取当前列的所有数值
        featList = [example[i] for example in dataSet]
        # 建立集合，获取唯一值组成的集合
        uniqueVals = set(featList)       
        newEntropy = 0.0
        # 计算每种划分方式的熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy  
        # 判断当前是否为最好的划分方式
        if (infoGain > bestInfoGain):       
            bestInfoGain = infoGain         
            bestFeature = i
    return bestFeature                     


# 返回出现次数最多的分类名称
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 1
        else:
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 创建决策树
def createTree(dataSet,labels):
    # 提取所有的类别
    classList = [example[-1] for example in dataSet]
    # 如果类别都相等，停止划分
    if classList.count(classList[0]) == len(classList): 
        return classList[0]
    # 仅剩单列可能时返回其出现次数最多的情况
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 获取最佳特征划分的序列
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 获取对应特征字符
    bestFeatLabel = labels[bestFeat]
    # 创建树
    myTree = {bestFeatLabel:{}}
    # 去除当前标签
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    # 递归所有的特征
    for value in uniqueVals:
        subLabels = labels[:]       
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree               


# 使用决策树进行分类
def classify(inputTree,featLabels,testVec):
    # 获取当前判断特征名称
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # 获取特征的索引
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    # 判断对应的是字典还是值
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: 
        classLabel = valueOfFeat
    return classLabel


# 决策树的存储
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

# 决策树的读取
def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

















