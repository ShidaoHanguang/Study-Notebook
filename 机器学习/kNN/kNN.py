# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:49:25 2024

@author: Jerome
"""

import numpy as np
import operator
import os

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    # 获取数据集大小
    dataSetSize = dataSet.shape[0]
    # np.tile用于沿指定方向复制数组。它的语法是 np.tile(array, reps)，其中 
    # array 是要复制的数组，reps 是指定沿每个轴重复的次数的元组。
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    # 这里实现的就是计算输入点距各数据集中点的距离
    sqDistance = sqDiffMat.sum(axis = 1)
    distances = sqDistance ** 0.5
    # 返回数组中元素排序后的索引
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 获取前k种可能的情况
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # 根据情况对应次数排序
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    #获取文本文件的行数
    numberOfLines = len(fr.readlines())         
    returnMat = np.zeros((numberOfLines,3))        
    classLabelVector = []                       #prepare labels return   
    index = 0
    # 执行fr.readlines()后文件指针指向文档尾部，需要fr.seek(0)重新回到开头
    fr.seek(0)
    dict_label = []
    # dict_label = {'didntLike' : 1,
    #               'smallDoses' : 2,
    #               'largeDoses' : 3
    #               }
    for line in fr.readlines():
        # .strip()可以截去回车字符
        line = line.strip()
        listFromLine = line.split('\t')
        # 矩阵读取文本每行的前三个数值
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    fr.close()
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

def datingClassTest(filepath,hoRatio,k):
    # 获取数据集
    datingDataMat,datingLabels = file2matrix(filepath)       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 整个数据集长度
    m = normMat.shape[0]
    # 设定检测数据集范围
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 进行kNN算法检测
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],k)
        print ("the classifier came back with: {}, the real answer is: {}".format(classifierResult, datingLabels[i]))
        # 统计错误次数
        if classifierResult != datingLabels[i]: 
            errorCount += 1.0
    print ("the total error rate is: {}" .format(errorCount/float(numTestVecs)))
    print (errorCount)

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("请输入用于玩游戏的时间比列："))
    ffMiles = float(input("请输入每年获取的飞行常客里程数："))
    iceCream = float(input("请输入每年冰淇淋的消耗量："))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("你可能会喜欢这个人：{}".format(resultList[classifierResult - 1]))
    
    
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
    
def handwritingClassTest():
    hwLabels = []
    # 读取训练数据文件夹中的文件
    trainingFileList = os.listdir('trainingDigits')           
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    # 读取为矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/' + fileNameStr)
    testFileList = os.listdir('testDigits')        
    errorCount = 0.0
    # 读取测试数据文件夹中的文件
    mTest = len(testFileList)
    errorlist = {}
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/' + fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: {}, the real answer is: {}".format(classifierResult, classNumStr))
        if classifierResult != classNumStr : 
            errorCount += 1.0
            errorlist[fileNameStr] = classifierResult
    print("the total number of errors is: {}".format(errorCount))
    print("the total error rate is: {}".format(errorCount/float(mTest)))
    return errorlist
    








