# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:06:05 2024

@author: Jerome
"""

import mytree
import treePlot


dataset, labels = mytree.createDataSet()
a = mytree.createTree(dataset, labels)

# 绘制树形图
treePlot.createPlot()