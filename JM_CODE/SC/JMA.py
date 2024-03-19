import random
import pickle as pkl
import numpy as np
import networkx as nx
import math
import more_itertools

import numpy as np
import pandas as pd

import itertools
import math
from operator import itemgetter
import time
import functools
import operator

import sys
from itertools import repeat
from collections import ChainMap

def pre_processing(ego_list):
    graph_list = []
    edges_dic = {}
    for g in ego_list:
        graph = []
        for e in g.edges:
            if tuple(sorted(e)) not in edges_dic:
                edges_dic[tuple(sorted(e))] = len(edges_dic)
            graph.append(edges_dic[tuple(sorted(e))])
        graph_list.append(graph)
    return graph_list,edges_dic


def JMA(segmentation):

    M = len(segmentation)
    size_list = [len(graph) for graph in segmentation]
    index_list = [0] * M
    index_dic_list = [{}]* M

    for j,g in enumerate(segmentation):
        ege_dic = {}
        for e in g:
            ege_dic[e] = j
        index_dic_list[j] = ege_dic
   
    two_list = []
    for gid,g in enumerate(segmentation):
        size_g = len(g)
        intersection_list = np.array([0] * M)
        for e in g:
            temporary_index_list = [0] * M
            for i,j in enumerate(index_dic_list):
                if e in j:
                    temporary_index_list[i] = temporary_index_list[i] + 1
            intersection_list = intersection_list + temporary_index_list
        two_list.append(np.sum(1 - (intersection_list/ ((np.array(size_list) + size_g) - intersection_list))))
        
    return min(two_list), segmentation[two_list.index(min(two_list))]

def jaccard_distance_list(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return 1 - float(intersection) / union

def compute_JD_list(sequence,representative):
    JD = 0
    edge_set = [list(g) for g in sequence]
    for g in edge_set:
        JD = JD + jaccard_distance_list(g, representative) 
    return JD

if __name__ == '__main__':

    data_path = sys.argv[1]

    output_path = sys.argv[2]

    with open (data_path, 'rb') as fp:
        ego_list = pkl.load(fp)

    ego_list,edges_dic = pre_processing(ego_list)

    start = time.time()
    V, summary = JMA(ego_list)
    end = time.time()

    JD = compute_JD_list(ego_list,summary)

    with open(output_path,'a') as f:
        f.write('time is : ' +  str(end - start) + '\n')
        f.write('JD is : ' +  str(JD) + '\n')
        f.write('summary graph is ' + str(summary) + '\n')



