#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/19 1:34
# @Author  : Nicole Sue
# @File    : tensor_reshaping.py
# @Project: PyTorch_Tutorials

import torch

x1 = torch.arange(9)
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])

x_3x3 = x1.view(3, 3)     # 也可以写成x_3x3 = x1.reshape(3, 3), reshape更安全更兼容
# tensor([[0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8]])

y1 = x_3x3.t()     # 转置
# tensor([[0, 3, 6],
#         [1, 4, 7],
#         [2, 5, 8]])
# print(y.view(9))会报错，要么改成reshape，要么改成print(y.contiguous().view(9))
# tensor([[0, 3, 6, 1, 4, 7, 2, 5, 8]])

x2 = torch.rand((2, 5))
x3 = torch.rand((2, 5))
print(torch.cat((x2, x3), dim=0).shape)     # torch.Size([4, 5])
# torch.cat(tensors, dim) 是 PyTorch 中的 张量拼接函数
# 将多个张量按指定维度 dim 拼接成一个新张量，要求除了拼接的那个维度以外，其他维度必须完全相同。
print(torch.cat((x2, x3), dim=1).shape)     # torch.Size([2, 10])
# dim=0: 行方向; dim=1: 列方向

z1 = x2.view(-1)
# 把张量 x2 “拉平”（flatten）成一维张量，并赋值给 z1。-1 是一个特殊值，表示 "自动推导维度"。
print(z1.shape)     # torch.Size([10])

batch = 64
x4 = torch.rand((batch, 2, 5))
z2 = x4.view(batch, -1)
print(z2.shape)     # torch.Size([64, 10])

z3 = x4.permute(0, 2, 1)     # 重排维度，64, 2, 5的维度分别是0, 1, 2
print(z3.shape)     # torch.Size([64, 5, 2])

x5 = torch.arange(10)
print(x5.unsqueeze(0).shape)     # torch.Size([1, 10])
# unsqueeze() 函数的作用是：在指定维度插入一个大小为 1 的新维度（也叫“扩展维度”）
# 在第 0 维（行方向）插入一个新维度，变成了一个 2D 行向量：1 行 10 列
print(x5.unsqueeze(1).shape)     # torch.Size([10, 1])
# 在第 1 维（列方向）插入一个新维度，变成了一个 2D 列向量：10 行 1 列

x6 = torch.arange(10).unsqueeze(0).unsqueeze(1)     # 1*1*10
z4 = x6.squeeze(1)     # 1*10

