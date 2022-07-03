# bcv
scratch on bcv

# Implementation of MEBF in Python
Implement the MEBF algorithm based on 
Wan C, Chang W, Zhao T, et al. Fast And Efficient Boolean Matrix Factorization By Geometric Segmentation[J]. arXiv, 2019: arXiv: 1909.03991.
The process is random, because the order after we perform UTL operation on matrix is not unique in most cases.

# MEBF Python代码实现
根据2019年普渡大学的一篇文章, Fast and Efficient Boolean Matrix Factorization by Geometric Segmentation.
过程有些随机，因为 UTL对矩阵的操作无法保证行与列的唯一排序，在大部分情况下。

通过BCV，以及对通过记录BCV进行estimate的矩阵的variance（平均值），可以有效地帮助pca去选择number of components，但是仍然缺乏randome matrix theory相关theorem给予理论证明。 
