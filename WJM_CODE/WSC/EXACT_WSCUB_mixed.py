import numpy as np
import sys
import pickle as pkl
import pandas as pd
import collections
import time
from operator import itemgetter
import copy
import math
from queue import PriorityQueue

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
    numerator = 0
    for commonkey in list(x.keys() & y.keys()):
        numerator = numerator + min(x[commonkey],y[commonkey])
    
    denominator = 0
    for allkey in list(x.keys() | y.keys()):
        denominator = denominator + max(x.get(allkey,0),y.get(allkey,0))
    
    return 1 - numerator/denominator

def compute_WJD(sequence,representative):
    WJD = 0
    multiplicity_dic = {}
    for i,g in enumerate(segmentation):
        multiplicity_dic[i] = dict(collections.Counter(segmentation[i]))
    representative_distribution =  dict(collections.Counter(representative))
    
    for key in list(multiplicity_dic.keys()):
        WJD = WJD + weighted_jaccard_distance(multiplicity_dic[key],representative_distribution) 
    return WJD

def EXACT_WSCUB (segmentation):
    
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
                
    rho = 0
    numerator_list = {}
    numerator_list_2 = []
    for key, value in edges_dic_multiplicity_graphID.items():
        rho = max(edges_dic_multiplicity_graphID[key],key=lambda x:x[1])[1] + rho
        multiplicity_list = np.array([i[1] for i in value])
        graph_index_list = np.array([i[0] for i in value])
        numerator_list[key] = []
        while len(multiplicity_list) > 0 :
            h = len(multiplicity_list)
            frequency = min(multiplicity_list)
            numerator_list[key].append([key,frequency,h,[size_list[gi] for gi in graph_index_list]])
            numerator_list_2.append([key,frequency,h,[size_list[gi] for gi in graph_index_list]])
            multiplicity_list = multiplicity_list - frequency
            delete_list = multiplicity_list != 0
            multiplicity_list = multiplicity_list[delete_list]
            graph_index_list = graph_index_list[delete_list]
            
    numerator_list_permanent = copy.deepcopy(numerator_list)
    n = len(numerator_list_permanent)
    m = len(segmentation)
    
    if rho == 1:     
        PQ = PriorityQueue(maxsize=n)
        for key in list(numerator_list.keys()):
            c_value = sum([float(1/(1+denominator)) for denominator in numerator_list[key][0][-1]])
            PQ.put([-c_value,key])
            if len(numerator_list[key]) > 1:
                if numerator_list[key][0][1] > 1:
                    numerator_list[key][0][1] = numerator_list[key][0][1] - 1
                else:
                    numerator_list[key].pop(0)
            else:
                if numerator_list[key][0][1] > 1:
                    numerator_list[key][0][1] = numerator_list[key][0][1] - 1
                else: 
                    del numerator_list[key]

        first_output = PQ.get()   
        output = []   
        output.append(first_output[1])
        return output
    
    else:
        weighted_EXACT_SCUB_list = [] 
        for iteration in range(1,rho):
            if (iteration+1) >= (rho-n-(n/m)*math.log(n, 2))/math.log(n, 2):

                s,out = FIXED_SIZE_GREEDY_SCUB_A(rho,iteration,numerator_list_2) 
                weighted_EXACT_SCUB_list.append((s,out))
            else:
                s,out = FIXED_SIZE_GREEDY_SCUB_S(iteration,numerator_list_permanent,
                             copy.deepcopy(numerator_list),n) 
                weighted_EXACT_SCUB_list.append((s,out))
        return list(sorted(weighted_EXACT_SCUB_list,key=itemgetter(0),reverse=True)[0][1])
    
def FIXED_SIZE_GREEDY_SCUB_A (rho,iteration,numerator_list_2):
    
    if iteration+1 !=  rho:
        all_scores = []
        edge_position_list = []
        weighted_S_index = 0
        for s in numerator_list_2:
            score_list = s[:3]
            score = sum([float(1/(iteration+1+denominator)) for denominator in s[-1]])
            score_list.append(score)

            for repetition in range(s[1]):
                all_scores.append(score)
                edge_position_list.append(s[0])

        output = []

        temp = np.argpartition(-np.array(all_scores), iteration+1)
        result_args = temp[:iteration+1]


        for i in result_args:
            output.append(edge_position_list[i]) 
            weighted_S_index = weighted_S_index + all_scores[i]

        return weighted_S_index, output
    
    else:
        all_scores = []
        edge_position_list = []
        weighted_S_index = 0
        for s in numerator_list_2:
            score_list = s[:3]
            score = sum([float(1/(iteration+1+denominator)) for denominator in s[-1]])
            score_list.append(score)

            for repetition in range(s[1]):
                all_scores.append(score)
                edge_position_list.append(s[0])

        for i in range(rho):
            weighted_S_index = weighted_S_index + all_scores[i]
            
        return weighted_S_index, edge_position_list

def FIXED_SIZE_GREEDY_SCUB_S (iteration,numerator_list_permanent,numerator_list,n):
    
    PQ = PriorityQueue(maxsize=n)
    for key in list(numerator_list.keys()):
        c_value = sum([float(1/(iteration+1+denominator)) for denominator in numerator_list[key][0][-1]])
        PQ.put([-c_value,key])
        if len(numerator_list[key]) > 1:
            if numerator_list[key][0][1] > 1:
                numerator_list[key][0][1] = numerator_list[key][0][1] - 1
            else:
                numerator_list[key].pop(0)
        else:
            if numerator_list[key][0][1] > 1:
                numerator_list[key][0][1] = numerator_list[key][0][1] - 1
            else: 
                del numerator_list[key]

    output = []     
    output_dict = {}
    first_output = PQ.get()
    output.append(first_output[1])
    selected_edge = first_output[1]
    output_dict[selected_edge] = 1

    for k in range(iteration):
        temporary_output = []
        if selected_edge in numerator_list:
            c_value = sum([float(1/(iteration+1+denominator)) for denominator in numerator_list[selected_edge][0][-1]])
            PQ.put([-c_value,selected_edge])

            if len(numerator_list[selected_edge]) > 1:
                if numerator_list[selected_edge][0][1] > 1:
                    numerator_list[selected_edge][0][1] = numerator_list[selected_edge][0][1] - 1
                else:
                    numerator_list[selected_edge].pop(0)
            else:
                if numerator_list[selected_edge][0][1] > 1:
                    numerator_list[selected_edge][0][1] = numerator_list[selected_edge][0][1] - 1
                else: 
                    del numerator_list[selected_edge]    

            next_select = PQ.get()
            output.append(next_select[1])

            if next_select[1] in output_dict:
                output_dict[next_select[1]] = output_dict[next_select[1]] + 1
            else:
                output_dict[next_select[1]] = 1

            selected_edge = next_select[1]

        else:
            next_select = PQ.get()
            output.append(next_select[1])
            selected_edge = next_select[1]

            if next_select[1] in output_dict:
                output_dict[next_select[1]] = output_dict[next_select[1]] + 1
            else:
                output_dict[next_select[1]] = 1

    temporary_score = 0
    for key, value in output_dict.items():
        accumulated_value = 0
        for list_computation in numerator_list_permanent[key]:
            for repetition in range(list_computation[1]):
                temporary_score = temporary_score + sum([float(1/(iteration+1+denominator)) for denominator in list_computation[-1]])
                accumulated_value  = accumulated_value + 1
                if accumulated_value == value:
                    break
            if accumulated_value  == value:
                break

    return temporary_score,output
    



if __name__ == '__main__':

    data_path = sys.argv[1]

    with open (data_path, 'rb') as fp:
        data = pkl.load(fp)

    graph_list,edges_dic = weighted_pre_processing(data)
    sparse_matrix_data, sparse_set_data =  weighted_make_data(graph_list)

    print('data size is: ', sparse_matrix_data.shape)

    start = time.time()
    summary = EXACT_WSCUB(sparse_set_data)
    end = time.time()
    runtime_time = end - start

    print('runtime is: ', runtime_time)
    print('WJD is: ', compute_WJD(sparse_set_data,summary))
    print('summary is: ', summary,flush=True)






