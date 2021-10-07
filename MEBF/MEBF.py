import pandas as pd
import math
import numpy as np
import random

# The input should be a dataframe with only 1s and 0s.

def matrix_product(a,b):
    # follow online instruction of binary matrix production
    # https://www2.math.upenn.edu/~deturck/m170/wk8/lecture/matrix.html
    x = np.zeros([a.shape[0],b.shape[1]])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            sum = 0
            for k in range(b.shape[0]):
                sum += (a[i,k]*b[k,j])
            x[i,j] = sum % 2
    return x

def matrix_subtract(a,b):
    """
    parameter:
        a, of shape m*n
        b, of shape m*n
    return:
        res, of shape m*n
    computation:
        1 - 1 = 0
        0 - 0 = 0
        1 - 0 = 1
        0 - 1 = 1
    """
    res = np.zeros(a.shape)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            res[i][j] = int((a[i][j] and (not b[i][j])) or ((not a[i][j]) and b[i][j]))
    return res

def reconstruct(x,d,e,f,g):
    """
    parameter:
        x, the matrix we want to approximate
        d, median column of x
        e, columns of high similarity with d
        f, median row of x
        g, rows of high similarity with g
    return:
        col_pattern, matrix approximated by columns
        row_pattern, matrix approximated by rows
        col_df_1, (m,1) dataframe, equal to d, sorted by index
        col_df_2, (1,n) dataframe, equal to e, sorted by columns
        row_df_1, (m,1) dataframe, equal to g, sorted by index
        row_df_2, (1,n) dataframe, equal to f, sorted by columns
    """
    # reconstruct matrix based on cols
    col_df_1 = pd.DataFrame(d.reshape(-1,1), index =x.index) # m x 1 matrix, samples
    col_df_2 = pd.DataFrame(np.array(e).reshape(1,-1), columns = x.columns) # 1 x n matrix, features
    col_df_1.sort_index(axis = 0, inplace=True)
    col_df_2.sort_index(axis = 1, inplace=True)
    # reconstruct matrix based on rows
    row_df_1 = pd.DataFrame(np.array(g).reshape(-1,1),index = x.index)
    row_df_2 = pd.DataFrame(f.reshape(1,-1),columns = x.columns)
    row_df_1.sort_index(axis = 0, inplace=True)
    row_df_2.sort_index(axis = 1, inplace=True)
    # col pattern production
    col_pattern = matrix_product(col_df_1.values, col_df_2.values)
    # row pattern production
    row_pattern = matrix_product(row_df_1.values, row_df_2.values)
    return col_pattern, row_pattern, col_df_1, col_df_2, row_df_1, row_df_2

def bi_growth(x, t=0.9):
    """
    parameter: 
        x, the input Binary Matrix
        t, the threshold of vector similarity
    output:
        (a,b): two sorted dataframe used to approximate matrix with lower cost
    """
    # UTL operation on X
    # drop columns and rows with all zeros
    x = x.loc[(x!=0).any(axis=1)]
    x = x.loc[:, (x != 0).any(axis=0)]
    # reorder based on idea in MEBF paper
    idx = x.eq(1).sum(axis=1).sort_values(ascending=False).index
    x = x.reindex(idx)
    cols = x.eq(1).sum(axis=0).sort_values(ascending=True).index
    x = x[cols]
    # d represents the median column of x
    med_col = len(x.columns) // 2
    # d = x[med_col].values
    d = x.values[:,med_col]
    # f represents the median row of x
    med_row = len(x.index) // 2
    # f = x.loc[med_row].values
    f=x.values[med_row,:]
    # iterate over each column and compute similarity
    e = []
    for col in x.columns:
        if (np.dot(x[col].values, d) / np.dot(d,d)) >= t:
            e.append(1)
        else:
            e.append(0)
    # iterate over each column and compute similarity
    g = []
    for row in x.index:
        if (np.dot(x.loc[row].values, f) / np.dot(f,f)) >= t:
            g.append(1)
        else:
            g.append(0)
    # reconstruct the approximation matrix based on rows and cols
    col_pattern, row_pattern,col_df_1,col_df_2,row_df_1,row_df_2 = reconstruct(x,d,e,f,g)
    # sort rows and columns
    x.sort_index(axis = 0,inplace=True)
    x.sort_index(axis = 1,inplace=True)
    # compute residule
    col_residule = sum(matrix_subtract(x.values, col_pattern)).sum()
    row_residule = sum(matrix_subtract(x.values, row_pattern)).sum()
    # compare cost
    if col_residule< row_residule:
        # col pattern is better
        return col_df_1, col_df_2
    else:
        # row pattern is better
        return row_df_1,row_df_2
    
def vector_and_op(a,b):
    res = list()
    for val in zip(a,b):
        if(val[0] and val[1]):
            res.append(1)
        else:
            res.append(0)
    return np.array(res).reshape(a.shape)

def weak_signal_detection(x,t=0.9):
    # UTL operation on X
    # drop columns and rows with all zeros
    x = x.loc[(x!=0).any(axis=1)]
    x = x.loc[:, (x != 0).any(axis=0)]
    # UTL operation on X
    idx = x.eq(1).sum(axis=1).sort_values(ascending=False).index
    x = x.reindex(idx)
    cols = x.eq(1).sum(axis=0).sort_values(ascending=True).index
    x = x[cols]
    # compute d_1
    d_1 = vector_and_op(x.values[:,-2],x.values[:,-1])
    # iterate over each column and compute similarity, filter cols
    e_1 = []
    for col in x.columns:
        if (np.dot(x[col].values, d_1) / np.dot(d_1,d_1)) >= t:
            e_1.append(1)
        else:
            e_1.append(0)
    # iterate over each column and compute similarity, filter rows
    d_2 = vector_and_op(x.values[0,:],x.values[1,:])
    e_2 = []
    for row in x.index:
        if (np.dot(x.loc[row].values, d_2) / np.dot(d_2,d_2)) >= t:
            e_2.append(1)
        else:
            e_2.append(0)
    # reconstruct the approximation matrix based on rows and cols
    pattern_1, pattern_2, l1_1,l1_2,l2_1,l2_2 = reconstruct(x,d_1,e_1,d_2,e_2)
    # sort by index and columns
    x.sort_index(axis = 0,inplace=True)
    x.sort_index(axis = 1,inplace=True)
    # compute residule
    l1 = matrix_subtract(x.values, pattern_1)
    l2 = matrix_subtract(x.values, pattern_2)
     # compare cost
    if sum(l1).sum() < sum(l2).sum():
        return l1_1,l1_2
    else:
        return l2_1,l2_2
    
def MEBF(x,t=0.8):
    # re_A, re_B to record the pattern
    re_A = pd.DataFrame(index =x.index.values)
    re_B = pd.DataFrame(columns = x.columns.values)
    cost = float('inf')
    x_residule = x.copy()
    while True:
        new_1, new_2 = bi_growth(x_residule,t)
        # check whether new pattern could fit better
        A_tmp = pd.concat([re_A,new_1],axis = 1).fillna(0)
        B_tmp = pd.concat([re_B,new_2],axis = 0).fillna(0)
        tmp_residule = matrix_subtract(x.values,matrix_product(A_tmp.values,B_tmp.values))
        if sum(tmp_residule).sum() > cost:
            # weak signal detection
            l1,l2 = weak_signal_detection(x_residule,t)
            A_tmp = pd.concat([re_A,l1],axis = 1).fillna(0)
            B_tmp = pd.concat([re_B,l2],axis = 0).fillna(0)
            tmp_residule = matrix_subtract(x.values,matrix_product(A_tmp.values,B_tmp.values))
            print("weak detection")
            print("old cost", cost)
            print("new cost", sum(tmp_residule).sum())
            if sum(tmp_residule).sum() >= cost:
                print("no more pattern, break")
                break
        # update re_A, re_B, and cost
        re_A = A_tmp.copy()
        re_B = B_tmp.copy()
        cost = sum(tmp_residule).sum()
        # update x_residule
        tmp = pd.DataFrame(matrix_product(new_1.values,new_2.values), index = new_1.index, columns = new_2.columns)
        for i in tmp.index:
            for j in tmp.columns:
                if(tmp.loc[i][j] == 1):
                    x_residule.loc[i][j] = 0
        if sum(x_residule.values).sum() == 0:
            break
    return re_A, re_B