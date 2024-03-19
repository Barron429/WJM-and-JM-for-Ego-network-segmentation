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


def zeta_function(edge_id,X,Ei):
    if Ei[edge_id] == 0 :
        return X[edge_id]
    elif Ei[edge_id] != 0 and max(Ei[edge_id],X[edge_id]) == Ei[edge_id]:
        return 0
    else:
        return -Ei[edge_id] + X[edge_id]

def incrementally_compute_WJD(I,previous_numerator_list,previous_denominator_list,sequence,representative,edge_id,data,summary,sum_of_Ei):    
    
    summary[edge_id] = representative
    
    numer_list = np.array([min(i,representative) for i in sequence]) 
    
    denom_list_3 = np.array([zeta_function(edge_id,summary,data[i,:]) for i in range(I)]) 
    
    WJD = I - np.sum((previous_numerator_list + numer_list)/(sum_of_Ei + previous_denominator_list + denom_list_3))
    
    if edge_id == 0:

        previous_denominator_list = np.array(denom_list_3[:])
    else:
        previous_denominator_list = previous_denominator_list + denom_list_3[:]

    summary[edge_id] = 0
    
    return WJD, numer_list, previous_denominator_list


def Greedy_WSC_q(data,l):

    I = data.shape[0]
    J = data.shape[1]
    summary = np.array([0] * J)
    
    quantile_matrix = np.quantile(a = data, q = [i/l for i in range(1,l)], axis = 0,  interpolation='nearest')
    
    previous_numerator_list = np.array([0] * I) 
    previous_denominator_list = [0] * I 
    sum_of_Ei = np.sum(data, axis=1)
    
    for edge_id in range(J):
        temporary_wjd_list = []
        temporary_numer_list = []
        temporary_denom_list = []
        one_slice = data[:,edge_id]
        candidate_set = pd.unique(quantile_matrix[:,edge_id])
        for candidate_id in candidate_set:
            temporary_wjd, temporary_numer, temporary_denom = incrementally_compute_WJD(I,previous_numerator_list,previous_denominator_list,one_slice,
                                                                                        candidate_id,edge_id,data,summary,sum_of_Ei)
            temporary_wjd_list.append(temporary_wjd)
            temporary_numer_list.append(temporary_numer)
            temporary_denom_list.append(temporary_denom)
        candidate_index = np.argsort(temporary_wjd_list)[0]
        summary[edge_id] = candidate_set[candidate_index]
        previous_numerator_list = np.array(temporary_numer_list[candidate_index]) + np.array(previous_numerator_list)
        previous_denominator_list = temporary_denom_list[candidate_index]
                
    return temporary_wjd_list[candidate_index],summary

if __name__ == '__main__':

    data_path = sys.argv[1]

    l = int(sys.argv[2])

    with open (data_path, 'rb') as fp:
        data = pkl.load(fp)

    graph_list,edges_dic = weighted_pre_processing(data)
    sparse_matrix_data, sparse_set_data =  weighted_make_data(graph_list)

    print('data size is: ', sparse_matrix_data.shape)

    start = time.time()
    WJD, summary = Greedy_WSC_q(sparse_matrix_data,l)
    end = time.time()
    runtime_time = end - start

    print('runtime is: ', runtime_time)
    print('WJD is: ', WJD)
    print('summary is: ', summary,flush=True)






