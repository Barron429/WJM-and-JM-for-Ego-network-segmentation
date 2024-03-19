import numpy as np
import sys
import pickle as pkl
import pandas as pd
import collections
import time
from operator import itemgetter

def weighted_pre_processing(ego_list):
    graph_list = []
    edges_dic = {}
    for g in ego_list:
        graph = []
        for e in g:
            if tuple(sorted(e)) not in edges_dic:
                edges_dic[tuple(sorted(e))] = len(edges_dic)
            graph.append(edges_dic[tuple(sorted(e))])
        graph_list.append(graph)
    return graph_list,edges_dic

def weighted_make_data(segment):
    all_edges = []
    for g in segment:
        for e in g:
            all_edges.append(e)
    all_edges = sorted(set(all_edges))

    all_edges_dict = {}
    
    for i, j in enumerate(all_edges):
        all_edges_dict[i] = j
        
    reversed_all_edges_dict = dict(map(reversed, all_edges_dict.items()))
        
    new_index_sequence = []    
    for g in segment:
        new_graph = []
        for e in g:
            new_graph.append(reversed_all_edges_dict[e])
        new_index_sequence.append(new_graph)

    all_edges = list(range(len(all_edges)))
    
    I = len(segment)
    J = len(all_edges)

    df = pd.DataFrame(np.zeros((I,J)),columns=all_edges)

    for i, row in df.iterrows():
        segment_edges = []
        frequency_dict = dict(collections.Counter(segment[i]))
        for key, value in frequency_dict.items():
            df.at[i, reversed_all_edges_dict[key]] = value    
            
    a = df.to_numpy()
    return a,new_index_sequence

def weighted_jaccard_distance(x,y):
    q = np.array([x,y])
    return 1 - np.sum(np.amin(q,axis=0))/np.sum(np.amax(q,axis=0))

def compute_WJD(sequence,representative):
    WJD = 0
    for g in sequence:
        WJD = WJD + weighted_jaccard_distance(g,representative) 
    return WJD

def WJMA(sequence):
    all_results = []
    for i in range(len(sequence)):
        all_results.append(compute_WJD(sequence,sequence[i]))
        

    return min(all_results), sequence[all_results.index(min(all_results))]

if __name__ == '__main__':

    data_path = sys.argv[1]

    with open (data_path, 'rb') as fp:
        data = pkl.load(fp)

    graph_list,edges_dic = weighted_pre_processing(data)
    sparse_matrix_data, sparse_set_data =  weighted_make_data(graph_list)

    print('data size is: ', sparse_matrix_data.shape)

    start = time.time()
    WJD, summary = WJMA(sparse_matrix_data)
    end = time.time()
    runtime_time = end - start

    print('runtime is: ', runtime_time)
    print('WJD is: ', WJD)
    print('summary is: ', summary,flush=True)
