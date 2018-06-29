from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
#import seaborn

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

def within_k_nearest(center, tocheck, NN, k):
    flag = False
    for i in range(1,k+1):
        if NN[center,i] == tocheck:
            flag = True
            break
    if flag:
        return True
    else:
        return False

def calculate_affinity_matrix(D, P_1, P_2):
    n = len(D)
    A_h = np.zeros((n, n), dtype = int)
    NN = calculate_nearest_neighbors(D)
    for i in range(n):
        for j in range(n):
            if i == j:
                A_h[i,j] = 0
                continue
            if A_h[j,i] == 2:
                A_h[i,j] = 2
                continue
            if within_k_nearest(i,j,NN,P_2):
                A_h[i,j] = 1
                continue
            if within_k_nearest(i,j,NN,P_1) and within_k_nearest(j,i,NN,P_1):
                A_h[i,j] = 2
                continue
            else:
                A_h[i,j] = 0
    return A_h

def calculate_H(index, A_h):
    H = 0
    for i in range(n):
        if A_h[index,i] == 2:
            H += 1
    return H

def generate_bi_subcluster(index, A_h, S_bool):
    bi_subcluster = [index]
    for j in range(n):
        if S_bool[j] == False and A_h[index,j] == 2:
            bi_subcluster.append(j)
    return bi_subcluster

def find_subclusters(D, P_1, P_2, P_3):
    n = len(D)
    label = [0] * n
    t = 0
    S = []
    S_bool = [False] * n
    SC = []
    A_h = calculate_affinity_matrix(D, P_1, P_2)
    for i in range(n):
        H = calculate_H(i, A_h)
        if H <= (P1-P3):
            S.append(i)
            S_bool[i] = True
    for i in range(n):
        if S_bool[i] == False:
            if label[i] == 1:
                continue
            else:
                SC.append(generate_bi_subcluster(i, A_h, S_bool))
                for index in SC[t]:
                    label[index] = 1
                t += 1
    return SC,S

def calculate_mergeable(SC, NN):
    k = len(SC)
    N = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            if i != j:


def mergeable(index1, index2):
    

def merge_subclusters(SC, S, NN):
    t = len(SC)
    label = [0] * t
    k = 0
    m = 0
    C = []
    for i in range(t):
        if label[i] == 0:
            k += 1
            label[i] = k
            C.append(SC[i])
        r = label[i]
        for j in range(t):
            if i != j and  mergeable(i, j):
                C[r-1] += SC[j]
                label[j] = r
    while len(S) != 0:
        m += 1
        for l in S:
            for h in range(k):
                if NN[l, m] in C[h]:
                    C[h].append(l)
    return C


if __name__ == '__main__':



    ncols = 6
    plt.figure(figsize=(ncols*3, 3))
    for col, data in enumerate(load_data()):
        plt.subplot(1, ncols, col+1)
        plt.scatter(data[0], data[1], s=5, alpha=0.5)
    plt.tight_layout()
    plt.show()
    plt.close()

