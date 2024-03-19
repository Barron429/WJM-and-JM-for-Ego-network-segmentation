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

def chunks(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


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

def prepare_ksegments(series,l):
    '''
    '''
    N = series.shape[0]

    dists = np.zeros((N,N))
            
    mean_dict = {}    

    for i in range(N):
        mean_dict[(i,i)] = [i for i in series[i]]

    for i in range(N):
        for j in range(N-i):
            R = i+j
            if i != R:
                sub_segment = series[i:R+1,:]
                m = sub_segment.shape[0]
                r = math.ceil(math.sqrt(m))
                error, representative = Greedy_WSC_d(sub_segment,l,r)

                mean_dict[(i,R)] = representative
                dists[i][R] = error
                
    return dists, mean_dict

def k_segments(series, k,l):
    '''
    '''
    N = series.shape[0]

    k = int(k)

    dists, means = prepare_ksegments(series,l)

    k_seg_dist = np.zeros((k,N+1))
    k_seg_path = np.zeros((k,N))
    k_seg_dist[0,1:] = dists[0,:]
    
    for i in range(k):
        k_seg_path[i,:] = i

    for i in range(1,k):
        for j in range(i,N):
            choices = k_seg_dist[i-1, :j] + dists[:j, j]
            best_index = np.argmin(choices)
            best_val = np.min(choices)

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

    l = int(sys.argv[5])

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
    segmentation,summary_graphs = k_segments(sparse_matrix_data,k,l)
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


