import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB,LinExpr, quicksum as qsum
import time
import sys
from numpy import inf


def EXACT_WSC(a):
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as m:
            
            I = a.shape[0]
            J = a.shape[1]


            xlb = a.min(axis=0)
            xub = a.max(axis=0)

            K = np.floor(np.log2(a-xlb))+1

            K[K == -inf] = 2

            K[K == 0] = 2

            K = K.astype(int)

            K = K.tolist()

            Nij = list(xub - xlb)
            N = []
            for i in range(I):
                N.append(Nij)

            N = np.asarray(N)

            N = np.floor(np.log2(N))+1

            N[N == -inf] = 2

            N[N == 0] = 2

            N = N.astype(int)

            N = N.tolist()

            x = m.addVars(1, J,vtype=GRB.INTEGER, name="x", lb = 0)

            for j in range(J):
                x[0,j].setAttr("lb", xlb[j])

            for j in range(J):
                x[0,j].setAttr("ub", xub[j])     

            v = m.addVars(I, J,vtype=GRB.INTEGER, name="v", lb = 0)

            w = m.addVars(I, J,vtype=GRB.INTEGER, name="w", lb = 0)

            p = m.addVars(1, I,vtype=GRB.CONTINUOUS, name="p")

            piL = 1/xub.sum()

            for i in range(I):
                p[0,i].setAttr("lb", piL)

            for i in range(I):
                p[0,i].setAttr("ub", 1/a[i].sum())               

            u = np.zeros((I,J)).tolist()
            s = np.zeros((I,J)).tolist()
            o = np.zeros((I,J)).tolist()
            t = np.zeros((I,J)).tolist()
            for i in range(I):
                for j in range(J):
                    u[i][j] = m.addVars(1,K[i][j], vtype=GRB.BINARY)
                    t[i][j] = m.addVars(1,N[i][j], vtype=GRB.BINARY)
                    s[i][j] = m.addVars(1,K[i][j], vtype=GRB.CONTINUOUS)
                    o[i][j] = m.addVars(1,N[i][j], vtype=GRB.CONTINUOUS)

            
            m.setObjective( qsum(xlb[j] * p[0,i] for i in range(I) for j in range(J)) + qsum(2**k * s[i][j][0,k] for i in range(I) for j in range(J) for k in range(K[i][j])) , GRB.MAXIMIZE)


            c = [0] * J
            for j, column in enumerate(a.T):
                candidate_set = np.unique(column)
                candidate_set_length  = len(candidate_set)
                c[j] = m.addVars(1,len(candidate_set), vtype=GRB.BINARY, name="c%s" % j)
                m.addConstr( qsum(c[j][0,OO] for OO in range(candidate_set_length)) == 1 )
                m.addConstr( qsum(c[j][0,oo] * candidate for oo,candidate in enumerate(candidate_set)) == x[0,j] )
            
            m.addConstrs( (v[i,j] == gp.min_(x[0,j],constant= a[i,j]) for i in range(I) for j in range(J)), "linearize_min()_constraint")
    
            m.addConstrs( (w[i,j] == gp.max_(x[0,j],constant= a[i,j]) for i in range(I) for j in range(J)), "linearize_max()_constraint")

            m.addConstrs( ( qsum(xlb[j] * p[0,i] for j in range(J)) + qsum(2**l * o[i][j][0,l] for j in range(J) for l in range(N[i][j])) - 1 == 0 for i in range(I) ), "Fractional_to_MILP_constraint")

            m.addConstrs( ( xlb[j] + qsum(2**l * t[i][j][0,l] for l in range(N[i][j])) == w[i,j] for i in range(I) for j in range(J)), "integer_to_binary_wij_constraint")

            m.addConstrs( ((piL * t[i][j][0,l]) <= o[i][j][0,l] for i in range(I) for j in range(J) for l in range(N[i][j])), "linearize_binary_wij_continuous_case")

            m.addConstrs( (o[i][j][0,l] <= (t[i][j][0,l] * 1/a[i].sum()) for i in range(I) for j in range(J) for l in range(N[i][j])), "linearize_binary_wij_continuous_case")

            m.addConstrs( ((piL * (1 - t[i][j][0,l])) <= p[0,i] - o[i][j][0,l] for i in range(I) for j in range(J) for l in range(N[i][j])), "linearize_binary_wij_continuous_case")

            m.addConstrs( (p[0,i] - o[i][j][0,l] <= 1/a[i].sum() * (1 - t[i][j][0,l]) for i in range(I) for j in range(J) for l in range(N[i][j])), "linearize_binary_wij_continuous_case")

            m.addConstrs( ( xlb[j] + qsum(2**k * u[i][j][0,k] for k in range(K[i][j])) == v[i,j] for i in range(I) for j in range(J)), "integer_to_binary_vij_constraint")

            m.addConstrs( ((piL * u[i][j][0,k]) <= s[i][j][0,k] for i in range(I) for j in range(J) for k in range(K[i][j])), "linearize_binary_vij_continuous_case")

            m.addConstrs( (s[i][j][0,k] <= u[i][j][0,k] * 1/a[i].sum() for i in range(I) for j in range(J) for k in range(K[i][j])), "linearize_binary_vij_continuous_case")

            m.addConstrs( ((piL * (1 - u[i][j][0,k])) <= p[0,i] - s[i][j][0,k] for i in range(I) for j in range(J) for k in range(K[i][j])), "linearize_binary_vij_continuous_case")

            m.addConstrs( ((p[0,i] - s[i][j][0,k]) <= 1/a[i].sum() * (1 - u[i][j][0,k]) for i in range(I) for j in range(J) for k in range(K[i][j])), "linearize_vij_binary_continuous_case")

            m.update()
            
            m.optimize()
            
            summary = []
            for vv in x.values():
                summary.append(int(vv.X))

            return I-m.objVal, summary


if __name__ == '__main__':

    # the below data is from the Fig2a of our paper

    # [(0, 1), (0, 2), (0, 3), (1, 2), (0, 4)]
    data = np.array([[10., 10., 1., 10., 0.], [10., 10., 0., 10., 1.]])

    WJD,summary = EXACT_WSC(data)

    print('WJD is : ', WJD)
    print('summary is : ', summary)































