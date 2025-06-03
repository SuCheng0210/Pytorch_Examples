#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/18 22:06
# @Author  : Nicole Sue
# @File    : tensor_math.py
# @Project: PyTorch_Tutorials
import torch
from dask.array import indices

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)     # z1 = tensor([10., 10., 10.])

z2 = torch.add(x, y)     # z2 = tensor([10., 10., 10.])

z3 = x + y     # z3 = tensor([10., 10., 10.])

# Subtraction
z4 = x - y

# Division
z5 = torch.true_divide(x, y)

# inplace operations
t = torch.zeros(3)
t.add_(x)
# 在 PyTorch 里，凡是方法名以“单下划线”结尾的（比如 add_、zero_、copy_ 等），都表示原地操作（in‑place）：
# 它直接在原来的张量上修改数值，不会分配新的内存。
# 等同于t += x

# Exponentiation
z6 = x.pow(2)

z7 = x ** 2

# Simple comparison
z8 = x > 0     # tensor([True, True, True])
z9 = x < 0     # tensor([False, False, False])

# Matrix Multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)     # 2*3

x4 = x1.mm(x2)     # 2*3

# Matrix Exponentiation
# 矩阵幂运算, 也就是把同一个矩阵自己连乘若干次
matrix_exp = torch.rand(5, 5)
print(matrix_exp.matrix_power(3))

# element wise multiplication逐元素乘法
z10 = x * y     # tensor([ 9, 16, 21])

# dot product数量积
z11 = torch.dot(x, y)     # tensor(46)

# Batch Matrix Multiplication批量矩阵乘法
batch = 32     # 一共有 32 对矩阵要相乘
n = 10     # 第一组矩阵的行数
m = 20     # 第一组矩阵的列数，也是第二组矩阵的行数
p = 30     # 第二组矩阵的列数
# 第一组：10*20；第二组：20*30

tensor1 = torch.rand((batch, n, m))     # 形状： (32, 10, 20)
tensor2 = torch.rand((batch, m, p))     # 形状： (32, 20, 30)
out_bmm = torch.bmm(tensor1, tensor2)   # (batch, n, p)

# Example of Broadcasting
# 广播（broadcasting） 的机制让形状不同的张量可以按“对齐规则”自动扩展，以便做逐元素运算
y1 = torch.rand((5, 5))
y2 = torch.rand((1, 5))

# 广播后 y2 会在第 0 维“重复”成 5 行，变成 5×5，然后做逐元素相减
z12 = y1 - y2
# 同理，y2 广播后再做逐元素幂运算
z13 = y1 ** y2

# Other useful tensor operations
sum_x = torch.sum(x, dim=0)     # 按第 0 维（列方向）累加所有元素
# x = tensor([[1, 2, 3],
#             [4, 5, 6]])
# tensor([5, 7, 9])
values1, indices1 = torch.max(x, dim=0)     # 在第 0 维（列）上找每一列的最大值 及其索引
# x = tensor([[1, 5, 3],
#             [4, 2, 6]])
# values1 = tensor([4, 5, 6])
# indices1 = tensor([1, 0, 1])
values2, indices2 = torch.min(x, dim=0)
# values2 = tensor([1, 2, 3])
# indices2 = tensor([0, 1, 0])
abs_x = torch.abs(x)     # 对 x 中的每个元素取绝对值
z14 = torch.argmax(x, dim=0)     # 在第 0 维上找到每一列最大值的行索引（不返回值本身）, tensor([1, 0, 1])
z15 = torch.argmin(x, dim=0)     # 在第 0 维上找到每一列最小值的行索引（不返回值本身）, tensor([0, 1, 0])
mean_x = torch.mean(x.float(), dim=0)     # 先将 x 转为浮点型，再按第 0 维计算均值
z16 = torch.eq(x, y)     # 按元素比较 x 和 y 是否相等，返回同形状的布尔张量, tensor([False, False, False])
sorted_y, indices3 = torch.sort(y, dim=0, descending=False)     # 沿第 0 维（对于向量就是对所有元素）排序, descending=False 指升序；改成 True 则降序
# y = tensor([3, 1, 2])
# sorted_y = tensor([1, 2, 3])
# indices3 = tensor([1, 2, 0])

z17 = torch.clamp(x, min=0, max=10)     # 把 x 中的值限制到区间(min,max)
# 小于 min 的全部赋值为 min, 大于 max 的全部赋值为 max, 落在区间内的保持不变

x5 = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)
z18 = torch.any(x)     # 检查张量中是否至少有一个元素为 True
# True，因为 x5 里不止一个元素是 True, tensor(True)
z19 = torch.all(x)     # 检查张量中是否所有元素都为 True
# False，因为 x5 里有一个元素是 False, tensor(False)

