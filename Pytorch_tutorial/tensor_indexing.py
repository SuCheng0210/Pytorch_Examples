#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/19 0:20
# @Author  : Nicole Sue
# @File    : tensor_indexing.py
# @Project: PyTorch_Tutorials

import torch

batch_size = 10
features1 = 25
x1 = torch.rand((batch_size, features1))     # (10, 25)

print(x1[0].shape)     # 或者写成x[0, :]
# torch.Size([25]), 取第一行(样本0), 得到有25个特征
print(x1[:, 0].shape)
# torch.Size([10]), 取第一列(特征0), 得到有10个样本
print(x1[2, 0:10])     # 0:10 --> [0, 1, 2, ..., 9]
# 张量的切片（slicing） 操作
# “0:10” 表示从索引 0 开始，一直到索引 10（不包含 10）──也就是 [0,1,2,…,9]
# “2” 表示取第 2 行（注意行和列都是从 0 开始编号，第 2 行就是第 3 条样本）
# 它会把第2行的第0到第9列的元素一次取出来，共 10 个值

x1[0, 0] = 100

# Fancy indexing
x2 = torch.arange(10)
indices1 = [2, 5, 8]
print(x2[indices1])     # tensor([2, 5, 8])

x3 = torch.rand((3, 5))
rows1 = torch.tensor([1, 0])     # 行索引：[1, 0]
cols1 = torch.tensor([4, 0])     # 列索引：[4, 0]
# rows1 和 cols1 必须长度相同
print(x3[rows1, cols1].shape)     # 高级索引：会依次取出 x3[1,4] 和 x3[0,0]
# torch.Size([2])

# More advanced indexing
x4 = torch.arange(10)
print(x4[(x4 < 2) | (x4 > 8)])     # tensor([0, 1, 9])
print(x4[(x4 < 2) & (x4 > 8)])     # tensor([], dtype=torch.int64)
print(x4[x4.remainder(2) == 0])     # tensor([0, 2, 4, 6, 8])

# Useful operations
print(torch.where(x4 > 5, x4, x4*2))     # tensor([ 0,  2,  4,  6,  8, 10,  6,  7,  8,  9])
# torch.where(condition, value_if_true, value_if_false)
print(torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique())     # tensor([0, 1, 2, 3, 4])
# .unique() 会移除所有重复的元素；按照升序排序。如果你想要保留原始顺序，可以加参数.unique(sorted=False)
print(x4.ndimension())     # 1
# 返回张量 x4 的维度数（维数、维阶、阶数），也可以写成x4.dim()
print(x4.numel())     # 10
# 返回张量 x4 中的所有元素的总个数，即：张量中一共有多少个数值（无论维度是多少）
