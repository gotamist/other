#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 18:26:45 2018

@author: gotamist
"""
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.feature_selection import SelectKBest
def KBestSelector(data_dict, features_list, k):
       
    data_array = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data_array)

    kbest = SelectKBest(k=k)
    kbest.fit(features, labels)
    scores = kbest.scores_
    tuples = zip(features_list[1:], scores)
    top_k_features = sorted(tuples, key=lambda x: x[1], reverse=True)

    return top_k_features[:k]