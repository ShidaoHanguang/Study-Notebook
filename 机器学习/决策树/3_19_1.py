# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:06:05 2024

@author: Jerome
"""

import mytree
import treePlot

fr = open('眼镜.txt')
lense = [line.strip().split('\t') for line in fr.readlines()]
labels = ['年龄', '症状', '是否散光', '眼泪频率']
tree = mytree.createTree(lense, labels)
treePlot.createPlot(tree)
