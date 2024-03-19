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


def prepare_ksegments(series):
    '''
    '''
    N = len(series)

    dists = np.zeros((N,N))
            
    mean_dict = {}    

    for i in range(N):
        mean_dict[(i,i)] = [i for i in series[i]]

    for i in range(N):
        for j in range(N-i):
            r = i+j
            if i != r:
                sub_segment = series[i:r+1]
                error, representative = EXACT_SCUB(sub_segment)

                mean_dict[(i,r)] = representative
                dists[i][r] = error
                
    return dists, mean_dict

def k_segments(series, k):
    '''
    '''
    N = len(series)

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
    segmentation,summary_graphs = k_segments(ego_list,k)
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
