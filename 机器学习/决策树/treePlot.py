# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:14:47 2024

@author: Jerome
"""

import matplotlib.pyplot as plt
# 指定中文字符集
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False)   
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    # plotTree.totalW = float(getNumLeafs(inTree))
    # plotTree.totalD = float(getTreeDepth(inTree))
    # plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    # plotTree(inTree, (0.5,1.0), '')
    plotNode(U'决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode(U'叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()
    
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )