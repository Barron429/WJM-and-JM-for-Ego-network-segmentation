import networkx as nx
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB,LinExpr, quicksum as qsum
import time
import sys


def EXACT_SC(segment):
    
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as m:

            all_edges = []
            for g in segment:
                for e in g.edges:
                    all_edges.append(tuple(sorted(e)))
            all_edges = sorted(set(all_edges))

            all_edges_dict = {}
            for i, j in enumerate(all_edges):
                all_edges_dict[i] = j

            reversed_all_edges_dict = dict(map(reversed, all_edges_dict.items()))

            all_edges = list(range(len(all_edges)))

            I = len(segment)
            J = len(all_edges)

            all_edges.insert(0,'a_0')

            df = pd.DataFrame(np.zeros((I,J+1)),columns=all_edges)

            for i, row in df.iterrows():
                segment_edges = []
                for e in segment[i].edges:
                    segment_edges.append(reversed_all_edges_dict[tuple(sorted(e))])
                df.at[i, segment_edges] = 1

            x = m.addVars(1, J,vtype=GRB.BINARY, name="x")

            a = df.to_numpy()

            b = 1 - a

            b[:, 0] = [len(graph.edges) for graph in segment]

            y = m.addVars(1, I, vtype=GRB.CONTINUOUS, name="y", ub = 1, lb = 0)

            for i in range(I):
                y[0,i].setAttr("ub", 1/b[i, 0])

            for i in range(I):
                y[0,i].setAttr("lb", 1/(b[i, 0]+J))

            z = m.addMVar((I, J), vtype=GRB.CONTINUOUS, name="z", ub = 1, lb = 0)

            for i in range(I):
                for j in range(J):
                    z[i,j].setAttr("ub", 1/(b[i, 0]))

            obj= LinExpr()
            for i in range(I):
                obj.addTerms(a[i,1:], z[i,:].tolist())
            m.setObjective(obj, GRB.MAXIMIZE)

            m.addConstrs( (b[i,0] * y[0,i] + qsum(b[i,j+1] * z[i,j] for j in range(J)) == 1 for i in range(I) ), "23b_constraint")

            m.addConstrs( (z[i,j] <= 1/b[i, 0] * x[0,j] for j in range(J) for i in range(I)) , "23c_constraint")

            m.addConstrs( (z[i,j] <= y[0,i] + 1/(b[i, 0] + J) * (x[0,j] - 1) for j in range(J) for i in range(I)), "23d_constraint")

            m.addConstrs( (z[i,j] >= 1/(b[i, 0] + J) * x[0,j] for j in range(J) for i in range(I)) , "23e_constraint")

            m.addConstrs( (z[i,j] >= y[0,i] + 1/b[i, 0] * (x[0,j] - 1) for j in range(J) for i in range(I)), "23f_constraint")

            m.update()

            m.optimize()

            all_edges.pop(0)

            all_edges = np.array(all_edges)

            edges = all_edges[np.array([int(v.X) for v in x.values()], dtype=bool)]

            real_edges = []

            for i in edges:
                real_edges.append(all_edges_dict[i])

            return I-m.objVal, sorted(real_edges)


def prepare_ksegments(series):
    '''
    '''
    N = len(series)

    dists = np.zeros((N,N))
            
    mean_dict = {}    

    for i in range(N):
        mean_dict[(i,i)] = [tuple(sorted(i)) for i in series[i].edges]

    for i in range(N):
        for j in range(N-i):
            r = i+j
            if i != r:
                sub_segment = series[i:r+1]
                error, representative = EXACT_SC(sub_segment)

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

    # the below data is from the Fig1 of our paper
    G1= nx.Graph([(0,1),(0,2),(0,3),(1,2)])
    G2= nx.Graph([(0,1),(0,2),(0,3),(1,2),(0,4),(3,4)])
    G3= nx.Graph([(0,1),(0,3),(0,4)])
    G4= nx.Graph([(0,1),(0,6),(0,5),(5,6),(1,5)])
    G5= nx.Graph([(0,1),(0,5),(0,6),(0,7),(5,7)])

    ego_list = [G1,G2,G3,G4,G5]

    segmentation,summary_graphs = k_segments(ego_list, 2)

    print('segmentation is : ', segmentation)
    print('summary_graphs are : ', summary_graphs)































