import numpy as np
import pandas as pd
import math
from numpy import inf
import time
import pickle as pkl
import collections
from operator import itemgetter
import random
import sys
from metrics import clustering_metrics



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

def GREEDY_WSC(data):
    I = data.shape[0]
    J = data.shape[1]
    summary = np.array([0] * J)
    
    
    previous_numerator_list = np.array([0] * I) 
    previous_denominator_list = [0] * I 
    
    sum_of_Ei = np.sum(data, axis=1)
    
    for edge_id in range(J):
        temporary_wjd_list = []
        temporary_numer_list = []
        temporary_denom_list = []
        one_slice = data[:,edge_id]
        candidate_set = np.unique(one_slice)
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
                
    return I-temporary_wjd_list[candidate_index],summary


def prepare_ksegments(series):
    '''
    '''
    N = series.shape[0]

    dists = np.zeros((N,N))
            
    mean_dict = {}    

    for i in range(N):
        mean_dict[(i,i)] = [i for i in series[i]]

    for i in range(N):
        for j in range(N-i):
            r = i+j
            if i != r:
                sub_segment = series[i:r+1,:]
                error, representative = GREEDY_WSC(sub_segment)

                mean_dict[(i,r)] = representative
                dists[i][r] = error
                
    return dists, mean_dict

def k_segments(series, k):
    '''
    '''
    N = series.shape[0]

    k = int(k)

    dists, means = prepare_ksegments(series)

    k_seg_dist = np.zeros((k,N+1))
    k_seg_path = np.zeros((k,N))
    k_seg_dist[0,1:] = dists[0,:]    

    for i in range(k):
        k_seg_path[i,:] = i

    for i in range(1,k):
        for j in range(i,N):
            choices = k_seg_dist[i-1, :j] + dists[:j, j]
            best_index = np.argmax(choices)
            best_val = np.max(choices)

            k_seg_path[i,j] = best_index
            k_seg_dist[i,j+1] = best_val
            
    reg = np.zeros(N)
    rhs = len(reg)-1
    range_list = []
    for i in reversed(range(k)):
        if i == k-1:
            lhs = k_seg_path[i,rhs]
            range_list.append((int(lhs),int(rhs+1)))
            reg[int(lhs):rhs+1] = int(i)
            rhs = int(lhs)
        else:
            lhs = k_seg_path[i,rhs]
            range_list.append((int(lhs),int(rhs)))
            reg[int(lhs):rhs] = int(i)
            rhs = int(lhs)   
            
            
    c_list = []
    for i in range_list:
        c_list.append(i[0])
        c_list.append(i[1])
        
    c_list = sorted(set(c_list))
    c_list = c_list[1:-1]
    

    means_index = []
    for i,j in enumerate(sorted(c_list)):
        if i == 0:
            means_index.append((0,j-1))
        else:
            means_index.append((sorted(c_list)[i-1],j-1))
            
    if N > 1:
        means_index.append((sorted(c_list)[-1],N-1))  

    summary_graphs = {}

    for i in means_index:
        summary_graphs[i] = means[i]
        
    return list([int(i) for i in reg]), summary_graphs


if __name__ == '__main__':

    data_path = sys.argv[1]

    output_path = sys.argv[2]

    true_lables_path = sys.argv[3]

    k = int(sys.argv[4])

    with open (data_path, 'rb') as fp:
        data = pkl.load(fp)


    with open(true_lables_path) as f:
        true_lables = [int(line.rstrip('\n')) for line in f]

    true_cut_points = []
    for i,j in enumerate(true_lables):
        if i != len(true_lables)-1:
            if true_lables[i] != true_lables[i+1]:
                true_cut_points.append(i)

    graph_list,edges_dic = weighted_pre_processing(data)

    reversed_edges_dic = dict(map(reversed, edges_dic.items()))

    sparse_matrix_data, sparse_set_data =  weighted_make_data(graph_list)

    start = time.time()
    segmentation,summary_graphs = k_segments(sparse_matrix_data,k)
    end = time.time()

    cm = clustering_metrics(true_lables, segmentation)
    acc, nmi, f1_macro = cm.evaluationClusterModelFromLabel()


    with open(output_path,'a') as f:
        f.write('time is : ' +  str(end - start) + '\n')
        f.write('segmentation is : ' + str(segmentation) + '\n')
        f.write('acc is : ' + str(acc) + '\n')
        f.write('nmi is : ' + str(nmi) + '\n')
        f.write('f1_macro is : ' + str(f1_macro) + '\n')
        for i in summary_graphs.keys():
            f.write('summary graph ' + str(i) + ' is ' + str([reversed_edges_dic[e] for e in summary_graphs[i]])   + '\n')


