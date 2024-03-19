from hyperminhash import *
import random
import pickle as pkl
import numpy as np
import networkx as nx
import math
import more_itertools
import pandas as pd
from operator import itemgetter
import time
import sys
from metrics import clustering_metrics

def jaccard_similarity(list1, list2):
    intersection = len(list1.intersection(list2))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def compute_JS(sequence,representative):
    JS = 0
    for g in sequence:
        JS = JS + jaccard_similarity(g, representative) 
    return JS

def jaccard_distance(list1, list2):
    intersection = len(list1.intersection(list2))
    union = (len(list1) + len(list2)) - intersection
    return 1 - float(intersection) / union

def compute_JD(sequence,representative):
    JD = 0
    for g in sequence:
        JD = JD + jaccard_distance(g, representative) 
    return JD


def HEU(segmentation):
    all_edges_score = {}
    segmentation_set = []
    for j,g in enumerate(segmentation):
        segmentation_set.append(set(g))
        g_size = len(g)
        for e in g:
            if e in all_edges_score:
                all_edges_score[e] = all_edges_score[e] + 1/g_size
            else:
                all_edges_score[e] = 0

    N = len(all_edges_score)

    add_edges_indes = [2**i for i in range(math.ceil(math.log(N,2)))]
    add_edges_indes.append(N)
    sorted_edge_list = [k for k, v in sorted(all_edges_score.items(), key=lambda item: item[1], reverse=True)]
    
    sorted_edge_set = [set(sorted_edge_list[:i]) for i in add_edges_indes]
    
    value_list = [compute_JS(segmentation_set,i) for i in sorted_edge_set]
    best_index = np.argmax(value_list)
    best_val = np.max(value_list)

    mid_index = add_edges_indes[best_index]


    final_best_val = 0
    final_best_index = 0
    phase_3 = []

    if mid_index == add_edges_indes[0]:

        phase_3 = [1]
        final_best_index = 0
        final_best_val = best_val

    elif mid_index == add_edges_indes[-1] and mid_index - add_edges_indes[best_index-1] == 1:
        phase_3 = [1]
        final_best_index = 0
        final_best_val = best_val

    elif mid_index == add_edges_indes[-1]:

        left_index = add_edges_indes[best_index-1]+1
        phase_3 = [mid_index,left_index]

        for i in [2**i for i in range(math.ceil(math.log(mid_index - left_index,2)))]:
            phase_3.append(mid_index-i)

        phase_3 = sorted(phase_3)
        
        sorted_edge_set = [set(sorted_edge_list[:i]) for i in phase_3]

        final_value_list = [compute_JS(segmentation_set,i) for i in sorted_edge_set]
        final_best_index = np.argmax(final_value_list)
        final_best_val = np.max(final_value_list)

    elif mid_index - add_edges_indes[best_index-1] == 1 and add_edges_indes[best_index+1] - mid_index == 1:

        phase_3 = [1]
        final_best_index = 0
        final_best_val = best_val       

    else:

        if mid_index - add_edges_indes[best_index-1] > 1 and add_edges_indes[best_index+1] - mid_index > 1:

            left_index = add_edges_indes[best_index-1]+1
            right_index = add_edges_indes[best_index+1]-1
            phase_3 = [mid_index,left_index,right_index]

            for i in [2**i for i in range(math.ceil(math.log(mid_index - left_index,2)))]:
                phase_3.append(mid_index-i)

            for i in [2**i for i in range(math.ceil(math.log(right_index - mid_index,2)))]:
                phase_3.append(mid_index+i)

            phase_3 = sorted(phase_3)
            sorted_edge_set = [set(sorted_edge_list[:i]) for i in phase_3]
            
            final_value_list = [compute_JS(segmentation_set,i) for i in sorted_edge_set]
            final_best_index = np.argmax(final_value_list)
            final_best_val = np.max(final_value_list)

        elif mid_index - add_edges_indes[best_index-1] == 1:
            right_index = add_edges_indes[best_index+1]-1
            phase_3 = [mid_index,right_index]

            for i in [2**i for i in range(math.ceil(math.log(right_index - mid_index,2)))]:
                phase_3.append(mid_index+i) 

            phase_3 = sorted(phase_3)
            sorted_edge_set = [set(sorted_edge_list[:i]) for i in phase_3]
            
            final_value_list = [compute_JS(segmentation_set,i) for i in sorted_edge_set]
            final_best_index = np.argmax(final_value_list)
            final_best_val = np.max(final_value_list)

        elif add_edges_indes[best_index+1] - mid_index == 1:
        #add_edges_indes[best_index+1] - mid_index == 1:
            left_index = add_edges_indes[best_index-1]+1
            phase_3 = [mid_index,left_index]
            for i in [2**i for i in range(math.ceil(math.log(mid_index - left_index,2)))]:
                phase_3.append(mid_index-i)
            phase_3 = sorted(phase_3)    
            sorted_edge_set = [set(sorted_edge_list[:i]) for i in phase_3]
            final_value_list = [compute_JS(segmentation_set,i) for i in sorted_edge_set]
            final_best_index = np.argmax(final_value_list)
            final_best_val = np.max(final_value_list)
    
    return final_best_val,sorted_edge_list[:phase_3[final_best_index]]


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
    V, summary = HEU(ego_list)
    end = time.time()

    JD = compute_JD_list(ego_list,summary)

    with open(output_path,'a') as f:
        f.write('time is : ' +  str(end - start) + '\n')
        f.write('JD is : ' +  str(JD) + '\n')
        f.write('summary graph is ' + str(summary) + '\n')




















