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


def EXACT_WSCUB (matrix_data,segmentation,J,K):
    
    edges_dic_multiplicity_graphID = {}
    multiplicity_dic = {}

    size_list = [len(graph) for graph in segmentation]

    for i,g in enumerate(segmentation):
        multiplicity_dic[i] = dict(collections.Counter(segmentation[i]))
        for e in pd.unique(g):
            if e not in edges_dic_multiplicity_graphID:
                edges_dic_multiplicity_graphID[e] = [(i,multiplicity_dic[i][e])]
            else:
                edges_dic_multiplicity_graphID[e].append((i,multiplicity_dic[i][e]))    
                

    numerator_list = []
    for key, value in edges_dic_multiplicity_graphID.items():
        multiplicity_list = np.array([i[1] for i in value])
        graph_index_list = np.array([i[0] for i in value])
        while len(multiplicity_list) > 0 :
            h = len(multiplicity_list)
            frequency = min(multiplicity_list)
            numerator_list.append([key,frequency,h,[size_list[gi] for gi in graph_index_list]])
            multiplicity_list = multiplicity_list - frequency
            delete_list = multiplicity_list != 0
            multiplicity_list = multiplicity_list[delete_list]
            graph_index_list = graph_index_list[delete_list]
                        
            
    weighted_EXACT_SCUB_list = [] 
    
    for k in range(1,K):

        all_scores = []
        edge_position_list = []
        weighted_S_index = 0
        for s in numerator_list:
            score_list = s[:3]
            score = sum([float(1/(k+denominator)) for denominator in s[-1]])
            score_list.append(score)

            for repetition in range(s[1]):
                all_scores.append(score)
                edge_position_list.append(s[0])
            
        output = [0] * J

        temp = np.argpartition(-np.array(all_scores), k)
        result_args = temp[:k]


        for i in result_args:
            output[edge_position_list[i]] = output[edge_position_list[i]] + 1
            weighted_S_index = weighted_S_index + all_scores[i]
            
        weighted_EXACT_SCUB_list.append((weighted_S_index,output))
        


    all_scores = []
    edge_position_list = []
    weighted_S_index = 0
    for s in numerator_list:
        score_list = s[:3]
        score = sum([float(1/(K+denominator)) for denominator in s[-1]])
        score_list.append(score)

        for repetition in range(s[1]):
            all_scores.append(score)
            edge_position_list.append(s[0])

    output = [0] * J

    for i in range(K):
        output[edge_position_list[i]] = output[edge_position_list[i]] + 1
        weighted_S_index = weighted_S_index + all_scores[i]

    weighted_EXACT_SCUB_list.append((weighted_S_index,output))

    
    summary = list(sorted(weighted_EXACT_SCUB_list,key=itemgetter(0),reverse=True)[0][1])
    
            
    return compute_WJD(matrix_data,summary), summary

if __name__ == '__main__':

    data_path = sys.argv[1]

    with open (data_path, 'rb') as fp:
        data = pkl.load(fp)

    graph_list,edges_dic = weighted_pre_processing(data)
    sparse_matrix_data, sparse_set_data =  weighted_make_data(graph_list)

    print('data size is: ', sparse_matrix_data.shape)

    I = sparse_matrix_data.shape[0]
    J = sparse_matrix_data.shape[1]
    K = int(np.sum(np.max(sparse_matrix_data, axis=0)))

    start = time.time()
    WJD, summary = EXACT_WSCUB(sparse_matrix_data,sparse_set_data,J,K)
    end = time.time()
    runtime_time = end - start

    print('runtime is: ', runtime_time)
    print('WJD is: ', WJD)
    print('summary is: ', summary,flush=True)
