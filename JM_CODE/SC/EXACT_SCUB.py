import pickle as pkl
import numpy as np
import networkx as nx
import pandas as pd
from operator import itemgetter
import time
from metrics import clustering_metrics
import sys


def jaccard_similarity_LB_list(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2)))
    return float(intersection) / union

def compute_JS_LB_list(sequence,representative):
    JS = 0
    edge_set = [list(g) for g in sequence]
    for g in edge_set:
        JS = JS + jaccard_similarity_LB_list(g, representative) 
    return JS

def EXACT_SCUB(segmentation):

    EXACT_SCUB_list = [] 

    M = len(segmentation)
    summary_edges_distribution_list = [0] * M
    
    size_list = [len(graph) for graph in segmentation]
    dic_edge_graphIDs = {}
    all_edges = []
    for i,g in enumerate(segmentation):
        for e in g:
            all_edges.append(e)
            if e not in dic_edge_graphIDs:
                dic_edge_graphIDs[e] = [i]
            else:
                dic_edge_graphIDs[e].append(i)  

    all_edges = list(set(all_edges))

    for iteration in range(1,len(all_edges)):
        
        candidate_value = []
        candidate_edge = []
        
        for e, value in dic_edge_graphIDs.items():
                
            objective_value = sum([float(1/(size_list[gid] + iteration)) for gid in value]) 
            candidate_value.append(objective_value)
            candidate_edge.append(e)
            
        temp = np.argpartition(-np.array(candidate_value), iteration)
        result_args = temp[:iteration]
        
        output = [candidate_edge[i] for i in result_args]
        
        EXACT_SCUB_list.append((sum([candidate_value[i] for i in result_args]),output))
        
    EXACT_SCUB_list.append((compute_JS_LB_list(segmentation,all_edges),all_edges))

    return sorted(EXACT_SCUB_list,key=itemgetter(0),reverse=True)[0][0], list(sorted(EXACT_SCUB_list,key=itemgetter(0),reverse=True)[0][1])

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
    V, summary = EXACT_SCUB(ego_list)
    end = time.time()

    JD = compute_JD_list(ego_list,summary)

    with open(output_path,'a') as f:
        f.write('time is : ' +  str(end - start) + '\n')
        f.write('JD is : ' +  str(JD) + '\n')
        f.write('summary graph is ' + str(summary) + '\n')