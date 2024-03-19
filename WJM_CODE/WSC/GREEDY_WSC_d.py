import numpy as np
import sys
import pickle as pkl
import pandas as pd
import collections
import time
import math
from operator import itemgetter

def weighted_jaccard_distance(x,y):
    q = np.array([x,y])
    return 1 - np.sum(np.amin(q,axis=0))/np.sum(np.amax(q,axis=0))

def compute_WJD(sequence,representative):
    WJD = 0
    for g in sequence:
        WJD = WJD + weighted_jaccard_distance(g,representative) 
    return WJD

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


def chunks(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

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

def Greedy_WSC_d(data,l,r):
    
    if r > 1:
    
        I = data.shape[0]
        J = data.shape[1]
        
        first_level_data_index = list(chunks(list(range(I)), r))
        
        all_index_list = []
        two_level_data = []
        for index in first_level_data_index:
            part_of_data = data[index,:]
            
            index_list = np.all(part_of_data==0, axis=0)
            all_index_list.append(index_list)
            part_of_data_no_zero = part_of_data[:,~index_list]
            error,summary = Greedy_WSC_q(part_of_data_no_zero,l)
            two_level_data.append(summary)
            
        new_two_level_data = []
        for i,j in zip(all_index_list,two_level_data):
            summary = np.array([0] * J)
            true_list = np.where(~i)[0]
            summary[true_list] = j
            new_two_level_data.append(summary)

        new_two_level_data = np.array(new_two_level_data)
        new_index_list = np.all(new_two_level_data==0, axis=0)
        new_part_of_data_no_zero = new_two_level_data[:,~new_index_list]
        if new_part_of_data_no_zero.shape[1] > 0:

            error,summary = Greedy_WSC_q(new_part_of_data_no_zero,l)

            final_summary = np.array([0] * J)
            new_true_list = np.where(~new_index_list)[0]
            final_summary[new_true_list] = summary

            return compute_WJD(data,final_summary),final_summary
        
        else:
            return I, []
        
    if r == 1:
        error,summary = Greedy_WSC_q(data,l)
        return error,summary
        
    else:
        print('Error. That is not a valid r')
        return False 


if __name__ == '__main__':

    data_path = sys.argv[1]

    l = int(sys.argv[2])

    with open (data_path, 'rb') as fp:
        data = pkl.load(fp)

    graph_list,edges_dic = weighted_pre_processing(data)
    sparse_matrix_data, sparse_set_data =  weighted_make_data(graph_list)

    print('data size is: ', sparse_matrix_data.shape)

    r = math.ceil(math.sqrt(sparse_matrix_data.shape[0]))

    start = time.time()
    WJD, summary = Greedy_WSC_d(sparse_matrix_data,l,r)
    end = time.time()
    runtime_time = end - start

    print('runtime is: ', runtime_time)
    print('WJD is: ', WJD)
    print('summary is: ', summary,flush=True)






