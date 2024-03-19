import pickle as pkl
import numpy as np
import networkx as nx
import pandas as pd
from operator import itemgetter
import time
from metrics import clustering_metrics
import sys

def GSCUB (segmentation):
    
    output = []
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

    all_candidate_value = []
    all_candidate_output = []
    edge_index_in_summary = []
    
    for iteration in range(len(all_edges)):
        candidate_value = []
        candidate_edge = []
        
        for e, value in dic_edge_graphIDs.items():
            temporary_index_list = [*summary_edges_distribution_list]
            for j in value:
                temporary_index_list[j] = temporary_index_list[j]  + 1
            
            objective_value = sum([float(temporary_index_list[i]/(size_list[i] + iteration+1)) for i in range(M)]) 

            candidate_value.append(objective_value)
            candidate_edge.append(e)
            
            
        largest_value_index = np.argmax(candidate_value)
        selected_edge = candidate_edge[largest_value_index]
        output.append(selected_edge)
        
        for w in dic_edge_graphIDs[selected_edge]:
            summary_edges_distribution_list[w] = summary_edges_distribution_list[w]  + 1
        
        del dic_edge_graphIDs[selected_edge]
        
        all_candidate_value.append(candidate_value[largest_value_index])
        all_candidate_output.append(output)

    
    final_largest_value_index = np.argmax(all_candidate_value)
    final_output = all_candidate_output[final_largest_value_index]

    return all_candidate_value[final_largest_value_index], final_output


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
    V, summary = GSCUB(ego_list)
    end = time.time()

    JD = compute_JD_list(ego_list,summary)

    with open(output_path,'a') as f:
        f.write('time is : ' +  str(end - start) + '\n')
        f.write('JD is : ' +  str(JD) + '\n')
        f.write('summary graph is ' + str(summary) + '\n')