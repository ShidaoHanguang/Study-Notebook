# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 18:47:15 2024

@author: Jerome
"""

import mybayes
import feedparser
np = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
a, b, c = mybayes.localWords(np, sf)