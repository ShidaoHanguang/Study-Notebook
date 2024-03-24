# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 18:40:05 2024

@author: Jerome
"""

import numpy as np


# 创建原始数据集
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 给定是否是侮辱性文字，0为正常，1为侮辱
    classVec = [0,1,0,1,0,1]   
    return postingList,classVec


#  创建词表
def createVocabList(dataSet):
    # 创建空集合
    vocabSet = set([])
    for document in dataSet:
        # 获取两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 集合转向量
def setOfWords2Vec(vocabList, inputSet):
    # 创建原始空向量
    returnVec = [0]*len(vocabList)
    # 将输入数据中存在的文本在向量中表示出来
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: 
            print ("the word: {} is not in my Vocabulary!".format(word))
    return returnVec


# 构建朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):
    # 获取数目
    numTrainDocs = len(trainMatrix)
    # 获取字典数目
    numWords = len(trainMatrix[0])
    # 获取总体概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 创建初值
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0                  
    # 根据分类结果训练，p0、p1分别为两种分类方式的概率
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = (p1Num/p1Denom)          
    p0Vect = (p0Num/p0Denom)          
    return p0Vect,p1Vect,pAbusive


# 构建朴素贝叶斯分类器训练函数
def trainNB1(trainMatrix,trainCategory):
    # 获取数目
    numTrainDocs = len(trainMatrix)
    # 获取字典数目
    numWords = len(trainMatrix[0])
    # 获取总体概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 创建初值
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0                  
    # 根据分类结果训练，p0、p1分别为两种分类方式的概率
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)          
    p0Vect = np.log(p0Num/p0Denom)          
    return p0Vect,p1Vect,pAbusive


# 分类
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 对数加，对应的是各概率相乘
    # 根据各词分类为1的概率为句中分类1中各词出现概率*分类为1的概率/各词出现概率
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
    

# 测试用例
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB1(trainMat,listClasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))