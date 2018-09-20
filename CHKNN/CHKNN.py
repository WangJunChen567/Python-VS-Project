#-------------------------------------------------------------------------------
# Qin, Y., Yu, Z. L., Wang, C. D., Gu, Z., & Li, Y. 
# (2018). A Novel clustering method based on hybrid K-nearest-neighbor graph. 
# Pattern Recognition, 74, 1-14.
#-------------------------------------------------------------------------------

from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

def load_data():
    yield read_csv(r"./data/DS1_4.csv", header=None)
    yield read_csv(r"./data/DS2_4.csv", header=None)
    yield read_csv(r"./data/DS3_3.csv", header=None)
    yield read_csv(r"./data/DS4_3.csv", header=None)
    yield read_csv(r"./data/DS5_7.csv", header=None)
    yield read_csv(r"./data/DS6_8.csv", header=None)

def calculate_nearest_neighbors(D):
    n = len(D)
    NN = np.zeros((n, n), dtype = int)
    DM = distance_matrix(D, D)
    for i in range(n):
        NN[i] = np.argsort(DM[i])
    return NN

def calculate_affinity_matrix(D, P_1, P_2):
    n = len(D)
    A_h = np.zeros((n, n), dtype = int)
    NN = calculate_nearest_neighbors(D)
    for i in range(n):
        for j in NN[i,1:P_2+2]:
            if j in NN[i,1:P_1+2] and i in NN[j, 1:P_1+2]:
                    A_h[i,j] = 2
                    A_h[j,i] = 2
            else:
                A_h[i,j] = 1
    return A_h, NN

def calculate_H(index, A_h):
    H = 0
    n = len(A_h)
    for i in range(n):
        if A_h[index,i] == 2:
            H += 1
    return H

def generate_bi_subcluster(index, A_h, label_S):
    bi_subcluster = [index]
    n = len(A_h)
    for j in range(n):
        if label_S[j] == 0 and A_h[index,j] == 2:
            bi_subcluster.append(j)
    return bi_subcluster

def find_subclusters(D, P_1, P_2, P_3):
    n = len(D)
    label = [0] * n
    t = 0
    S = []
    SC = []
    label_S = [0] * n
    A_h, NN = calculate_affinity_matrix(D, P_1, P_2)
    for i in range(n):
        H = calculate_H(i, A_h)
        if H <= (P_1-P_3):
            S.append(i)
            label_S[i] = 1
        else:
            if label[i] == 1:
                continue
            else:
                SC.append(generate_bi_subcluster(i, A_h, label_S))
                for x in SC[t]:
                    label[x] = 1
                t += 1
    return SC, S, A_h, NN

def calculate_NumBridgePoint(SC, A_h):
    k = len(SC)
    N_BP = np.zeros((k,k), dtype = int)
    for i in range(k):
        for j in range(k):
            if i != j:
                for u in SC[i]:
                    for v in SC[j]:
                        if A_h[u,v] == 1:
                            N_BP[i,j] += 1
    return N_BP


def mergeable(index1, index2, N_BP):
    if N_BP[index1,index2] != 0 and N_BP[index1,index2] == max(N_BP[index1]):
        return True
    else:
        return False

def merge_subclusters(SC, S, NN, A_h):
    t = len(SC)
    label = [0] * t
    k = 0
    m = 0
    C = []
    N_BP = calculate_NumBridgePoint(SC, A_h)
    for i in range(t):
        if label[i] == 0:
            k += 1
            label[i] = k
            C.append(SC[i])
        r = label[i]
        for j in range(t):
            if i != j and  mergeable(i, j, N_BP):
                C[r-1] += SC[j]
                label[j] = r
    while len(S) != 0:
        m += 1
        for l in S:
            for h in range(k):
                if NN[l, m] in C[h]:
                    C[h].append(l)
                    S.remove(l)
                    break
    return C


if __name__ == '__main__':
    P_1 = [11, 5, 6, 5, 11, 5]
    P_2 = [11, 6, 8, 5, 11, 5]
    P_3 = [2, 2, 2, 2, 2, 2]
    plt.figure(figsize=(9, 5.5))
    for i, data in enumerate(load_data()):
        SC, S, A_h, NN = find_subclusters(data, P_1[i], P_2[i], P_3[i])
        #C = merge_subclusters(SC, S, NN, A_h)
        plt.subplot(2, 3, i+1)
        #for SCs in C:
        plt.scatter(data.iloc[S][0],data.iloc[S][1], s=5,c='k', alpha=0.5)
        for SCs in SC:
            plt.scatter(data.iloc[SCs][0], data.iloc[SCs][1], s=5, alpha=0.5)
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
    plt.tight_layout()

    #plt.figure(figsize=(9, 5.5))
    #for i, data in enumerate(load_data()):
    #    plt.subplot(2, 3, i+1)
    #    plt.scatter(data[0], data[1], s=5, alpha=0.5)
    #    plt.xlabel("$x_1$")
    #    plt.ylabel("$x_2$")
    #plt.tight_layout()
    plt.show()
    plt.close()

