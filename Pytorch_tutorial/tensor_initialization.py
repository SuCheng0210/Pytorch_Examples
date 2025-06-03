#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/18 11:59
# @Author  : Nicole Sue
# @File    : tensor_initialization.py
# @Project: PyTorch_Tutorials

import torch
from tensorflow.python.util.numpy_compat import np_array

device = "cuda" if torch.cuda.is_available() else "cpu"

# How to create a tensor? 怎么创建一个张量？
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                         device=device, requires_grad=True)    # 直接设置：device="cuda"

print(my_tensor)
print(my_tensor.dtype)     # torch.float32
print(my_tensor.device)     # cpu
print(my_tensor.shape)     # torch.Size([2, 3])
print(my_tensor.requires_grad)     # True

# Other common initialization methods
a = torch.empty(size=(3, 3))     # 初始torch设为空，size设为3*3
# tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
b = torch.zeros((3, 3))
# tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
c = torch.rand((3, 3))
# 从均匀分布(0,1)中随机抽样生成tensor([[0.9761, 0.1404, 0.3942], [0.0279, 0.3163, 0.7814], [0.9162, 0.2117, 0.4477]])
d = torch.ones((3, 3))
# tensor([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
e = torch.eye(5, 5)     # I, eye, 1 on the diagonal and the rest will be 0
# tensor([[1., 0., 0., 0., 0.],
#         [0., 1., 0., 0., 0.],
#         [0., 0., 1., 0., 0.],
#         [0., 0., 0., 1., 0.],
#         [0., 0., 0., 0., 1.]])
f = torch.arange(start=0, end=5, step=1)     # step是每一步的步长
# tensor([0, 1, 2, 3, 4]), 左闭右开
g = torch.linspace(start=0.1, end=1, steps=10)     # steps说明有10个数, 步长=(1-0.1)/(10-1)=0.1
# tensor([0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000, 1.0000])
h = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
# 从均值0、方差1的正态分布中随机抽样生成tensor([[ 0.4852, -0.1108, -0.2674,  0.3094,  0.7293]])
i = torch.empty(size=(1, 5)).uniform_(0, 1)
# 从均匀分布(0, 1)中随机抽样生成tensor([[0.1635, 0.0160, 0.9100, 0.6528, 0.4237]])
j = torch.diag(torch.ones(3))
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])

# How to initialize and convert tensors to other types (int, float, double)
k = torch.arange(4)     # end=4, 默认start=0, step=1
print(k.bool())     # .bool()会将每个元素转为布尔值, 0-->False, 非0-->True
# tensor([False,  True,  True,  True])
print(k.short())     # tensor([0, 1, 2, 3], dtype=torch.int16)
print(k.long())     # tensor([0, 1, 2, 3])
# int64(Important)
print(k.half())     # tensor([0., 1., 2., 3.], dtype=torch.float16)
print(k.float())     # tensor([0., 1., 2., 3.])
# float32(Important)
print(k.double())     # tensor([0., 1., 2., 3.], dtype=torch.float64)

# Array to Tensor conversion and vice versa
import numpy as np
np_array = np.zeros((5, 5))     # 创建一个形状为 5×5、所有元素都为 0.0 的浮点型 NumPy 数组
x = torch.from_numpy(np_array)     # 把一个 NumPy 数组 “视图”（view）转换成 PyTorch 张量
# 重要点：它们共享内存——也就是说，后面如果你修改了 x（张量）的值，np_array（原数组）也会跟着改变；同样地，改 np_array，x 也会变
np_array_back = x.numpy()     # 把这个张量再转回 NumPy 数组，得到一个新的 NumPy 对象 np_array_back

