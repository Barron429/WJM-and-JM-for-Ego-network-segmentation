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


def Memoization(sub_segment,all_dis_dict,index_sub_segment):

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
                    error, representative = EXACT_SCUB(sub_segment[1:])
                    all_dis_dict[(o[1][0],o[1][1])] = [error, representative]
                error_list.append(all_dis_dict[(o[0][0],o[0][1])][0]+all_dis_dict[(o[1][0],o[1][1])][0])
            if i == 1:
                if (o[0][0],o[0][1]) not in all_dis_dict:
                    error, representative = EXACT_SCUB(sub_segment[0:2])
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
            error, representative = EXACT_SCUB(sub_segment[1:])
            all_dis_dict[(index_sub_segment[1],end)] = [error, representative]
        index_list.append([(start,start),(index_sub_segment[1],end)])
        error_list.append(0.5+all_dis_dict[(index_sub_segment[1],end)][0])
        for j in index_sub_segment[1:-2]:
            anchor_index = index_sub_segment.index(j)
            if (start,j) not in all_dis_dict:
                error, representative = EXACT_SCUB(sub_segment[0:anchor_index+1])
                all_dis_dict[(start,j)] = [error, representative]
            if (j+1,end) not in all_dis_dict:    
                error, representative = EXACT_SCUB(sub_segment[anchor_index+1:])
                all_dis_dict[(j+1,end)] = [error, representative]
            index_list.append([(start,j),(j+1,end)])
            error_list.append(all_dis_dict[(start,j)][0]+all_dis_dict[(j+1,end)][0])
        if (start,index_sub_segment[-2]) not in all_dis_dict:
            error, representative = EXACT_SCUB(sub_segment[0:len(index_sub_segment)-1])
            all_dis_dict[(start,index_sub_segment[-2])] = [error, representative]
        index_list.append([(start,index_sub_segment[-2]),(end,end)])
        error_list.append(all_dis_dict[(start,index_sub_segment[-2])][0]+0.5)

        best_index = index_list[np.argmax(error_list)][0][1]
        best_val = np.max(error_list)
    return best_index, best_val


def top_down(sequence,k):
    all_error = 0
    all_dis_dict = {}
    index_segment = list(range(len(sequence)))
    for i in range(len(sequence)):
        all_dis_dict[(i,i)] = [0.5, sequence[i]]
    ob_value_list = []
    cut_point_list = []
    for run in range(k-1):
        if run == 0:
            cut_point, ob_value = Memoization(sequence,all_dis_dict,index_segment)
            index_segment = [index_segment[:cut_point+1],index_segment[cut_point+1:]]
            ob_value_list.append(ob_value)
            cut_point_list.append(cut_point)
        else:
            temporary_ob_value_list = []
            temporary_cut_point = []
            cut_which_segment = []
            for i in index_segment:
                cut_point, ob_value = Memoization(sequence[i[0]:i[-1]+1],all_dis_dict,i)
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
            
    flat_reaults = [0] * len(sequence)
    for a,i in enumerate(sorted(index_segment)):
        for j in i:
            flat_reaults[j] = a   

    N = len(sequence)

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


if __name__ == '__main__':

    data_path = sys.argv[1]

    output_path = sys.argv[2]

    true_lables_path = sys.argv[3]

    k = int(sys.argv[4])

    with open (data_path, 'rb') as fp:
        ego_list = pkl.load(fp)


    with open(true_lables_path) as f:
        true_lables = [int(line.rstrip('\n')) for line in f]

    true_cut_points = []
    for i,j in enumerate(true_lables):
        if i != len(true_lables)-1:
            if true_lables[i] != true_lables[i+1]:
                true_cut_points.append(i)

    ego_list,edges_dic = pre_processing(ego_list)

    reversed_edges_dic = dict(map(reversed, edges_dic.items()))
    
    start = time.time()
    segmentation,summary_graphs = top_down(ego_list,k)
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































