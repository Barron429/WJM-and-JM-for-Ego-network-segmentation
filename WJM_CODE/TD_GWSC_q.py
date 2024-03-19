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
                
    return I-temporary_wjd_list[candidate_index],summary


def Memoization(sub_segment,all_dis_dict,index_sub_segment,l):

    if len(sub_segment) == 1:
        best_index = index_sub_segment[0]
        best_val = 0.5

    if len(sub_segment) == 2:
        best_index = index_sub_segment[0]
        best_val = 1

    if len(sub_segment) == 3:
        error_list = []
        options = [[(index_sub_segment[0],index_sub_segment[0]),(index_sub_segment[1],index_sub_segment[2])],[(index_sub_segment[0],index_sub_segment[1]),(index_sub_segment[2],index_sub_segment[2])]]
        for i,o in enumerate(options):
            if i == 0:
                if (o[1][0],o[1][1]) not in all_dis_dict:
                    error, representative = Greedy_WSC_q(sub_segment[1:,:],l)
                    all_dis_dict[(o[1][0],o[1][1])] = [error, representative]
                error_list.append(all_dis_dict[(o[0][0],o[0][1])][0]+all_dis_dict[(o[1][0],o[1][1])][0])
            if i == 1:
                if (o[0][0],o[0][1]) not in all_dis_dict:
                    error, representative = Greedy_WSC_q(sub_segment[0:2,:],l)
                    all_dis_dict[(o[0][0],o[0][1])] = [error, representative]
                error_list.append(all_dis_dict[(o[0][0],o[0][1])][0]+all_dis_dict[(o[1][0],o[1][1])][0]) 
        best_index = options[np.argmax(error_list)][0][1]
        best_val = np.max(error_list)
        
    if len(sub_segment) > 3:
        start = index_sub_segment[0]
        end = index_sub_segment[-1]
        index_list = []
        error_list = []
        if (index_sub_segment[1],end) not in all_dis_dict:
            error, representative = Greedy_WSC_q(sub_segment[1:,:],l)
            all_dis_dict[(index_sub_segment[1],end)] = [error, representative]
        index_list.append([(start,start),(index_sub_segment[1],end)])
        error_list.append(0.5+all_dis_dict[(index_sub_segment[1],end)][0])
        for j in index_sub_segment[1:-2]:
            anchor_index = index_sub_segment.index(j)
            if (start,j) not in all_dis_dict:
                error, representative = Greedy_WSC_q(sub_segment[0:anchor_index+1,:],l)
                all_dis_dict[(start,j)] = [error, representative]
            if (j+1,end) not in all_dis_dict:    
                error, representative = Greedy_WSC_q(sub_segment[anchor_index+1:],l)
                all_dis_dict[(j+1,end)] = [error, representative]
            index_list.append([(start,j),(j+1,end)])
            error_list.append(all_dis_dict[(start,j)][0]+all_dis_dict[(j+1,end)][0])
        if (start,index_sub_segment[-2]) not in all_dis_dict:
            error, representative = Greedy_WSC_q(sub_segment[0:len(index_sub_segment)-1,:],l)
            all_dis_dict[(start,index_sub_segment[-2])] = [error, representative]
        index_list.append([(start,index_sub_segment[-2]),(end,end)])
        error_list.append(all_dis_dict[(start,index_sub_segment[-2])][0]+0.5)

        best_index = index_list[np.argmax(error_list)][0][1]
        best_val = np.max(error_list)
    return best_index, best_val


def top_down(sequence,k,l):
    all_error = 0
    all_dis_dict = {}
    N = sequence.shape[0]
    index_segment = list(range(N))
    for i in range(N):
        all_dis_dict[(i,i)] = [0.5, sequence[i,:]]
    ob_value_list = []
    cut_point_list = []
    for run in range(k-1):
        if run == 0:
            cut_point, ob_value = Memoization(sequence,all_dis_dict,index_segment,l)
            index_segment = [index_segment[:cut_point+1],index_segment[cut_point+1:]]
            ob_value_list.append(ob_value)
            cut_point_list.append(cut_point)
        else:
            temporary_ob_value_list = []
            temporary_cut_point = []
            cut_which_segment = []
            for i in index_segment:
                cut_point, ob_value = Memoization(sequence[i[0]:i[-1]+1,:],all_dis_dict,i,l)
                temporary_ob_value_list.append(ob_value)
                temporary_cut_point.append(cut_point)
                cut_which_segment.append(i)
            ob_value = np.max(temporary_ob_value_list)
            cut_point = temporary_cut_point[np.argmax(temporary_ob_value_list)]
            index_cut_point = cut_which_segment[np.argmax(temporary_ob_value_list)].index(cut_point)
            index_segment.remove(cut_which_segment[np.argmax(temporary_ob_value_list)])
            index_segment.append(cut_which_segment[np.argmax(temporary_ob_value_list)][:index_cut_point+1])
            index_segment.append(cut_which_segment[np.argmax(temporary_ob_value_list)][index_cut_point+1:])
            index_segment = sorted(index_segment[:])
            ob_value_list.append(ob_value)
            cut_point_list.append(cut_point)
            
    flat_reaults = [0] * N
    for a,i in enumerate(sorted(index_segment)):
        for j in i:
            flat_reaults[j] = a   

#     N = len(sequence)

    means_index = []
    for i,j in enumerate(sorted(cut_point_list)):
        if i == 0:
            means_index.append((0,j))
        else:
            means_index.append((sorted(cut_point_list)[i-1]+1,j))
            
    if N > 1:
        means_index.append((sorted(cut_point_list)[-1]+1,N-1))  

    summary_graphs = {}

    for i in means_index:
        summary_graphs[i] = all_dis_dict[i][1]


    return flat_reaults,summary_graphs


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
    segmentation,summary_graphs = top_down(sparse_matrix_data,k,l)
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


