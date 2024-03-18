# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:21:11 2024

@author: Jerome
"""

import kNN
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# group, labels = kNN.createDataSet()
# res = kNN.classify0([0,0], group, labels, 3)
# datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
# plt.show()
# normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
# kNN.datingClassTest()
# kNN.classifyPerson()
errorlist = kNN.handwritingClassTest()
