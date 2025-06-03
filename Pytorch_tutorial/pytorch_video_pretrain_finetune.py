#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/3 14:44
# @Author  : Nicole Sue
# @File    : pytorch_video_pretrain_finetune.py
# @Project: PyTorch_Tutorials

# Imports
import torch     # 导入PyTorch核心库，提供张量操作、自动求导和GPU加速等功能，是构建神经网络的基础。
import torch.nn as nn     # 导入PyTorch的神经网络模块（nn），包含各种层（如全连接层Linear、卷积层Conv2d）、激活函数和损失函数等，用于定义模型结构。
import torch.optim as optim     # 导入优化算法模块（optim），包含常见的优化器如SGD、Adam等，用于在训练过程中更新模型参数。
import torchvision
from torch.utils.data import DataLoader     # 导入数据加载工具（DataLoader），用于批量加载数据、支持多线程和数据打乱，简化数据输入到模型的过程。
import torchvision.datasets as datasets     # 导入PyTorch视觉库中的数据集模块（datasets），提供常用数据集（如MNIST、CIFAR-10）的快速加载接口。
import torchvision.transforms as transforms     # 导入数据预处理模块（transforms），用于图像预处理（如调整尺寸、转换为张量、标准化），常通过Compose组合多个操作。
from torchvision.models import vgg16, VGG16_Weights

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channels=3     # 和FN不同
num_classes = 10     # num_classes：分类任务的类别数。
learning_rate = 1e-3
batch_size = 1024
num_epochs = 5

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

# Load pretrain model & modify it
weights = VGG16_Weights.DEFAULT
model = vgg16(weights=weights)

for param in model.parameters():
    param.requires_grad = False

model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Flatten(),
                                 nn.Linear(512, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 10),)
model.to(device)


# Load data
train_dataset = datasets.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
# 数据预处理：通过transform=transforms.ToTensor()将图像转换为PyTorch张量，并自动归一化像素值到[0, 1]范围。
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# 创建数据加载器DataLoader，支持批量加载和随机打乱（仅训练集需要shuffle=True）。
# train=True/False：加载训练集或测试集。
test_dataset = datasets.CIFAR10(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# batch_size：每个批次的样本数（需提前定义）。
# shuffle=True：仅对训练集打乱顺序，增强泛化性。


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# 交叉熵损失函数，适用于多分类任务（内部集成Softmax）。
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Adam优化器，用于更新模型参数，需传入学习率。

# Train Network
for epoch in range(num_epochs):
    model.train()
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible, 数据迁移：将当前批次的输入数据data和标签targets移动到指定设备（GPU或CPU）。
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward, 前向传播
        scores = model(data)     # scores（未归一化的类别分数）
        loss = criterion(scores, targets)     # 损失计算：通过交叉熵损失函数计算预测值与真实标签的误差。

        losses.append(loss.item())

        # backward, 反向传播
        optimizer.zero_grad()     # 清空上一批次的梯度，避免累积。
        loss.backward()     # 计算损失对参数的梯度。

        # gradient descent or adam stop
        optimizer.step()     # 根据梯度更新模型参数。

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:     # 判断是训练集还是测试集
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()     # 切换到评估模式（关闭Dropout和BatchNorm的随机性，确保评估结果稳定。）

    with torch.no_grad():     # 关闭梯度计算，禁用自动求导，减少内存消耗并加速计算。
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            '''相比FN需删除的部分：
            x = x.reshape(x.shape[0], -1)     # 展平数据
            '''

            scores = model(x)
            # 64*10
            _, predictions = scores.max(1)     # 在维度1（类别维度）取最大值索引，得到预测类别。
            num_correct += predictions.eq(y).sum()
            # 也可以写成num_correct += (predictions == y).sum()：比较预测结果与真实标签，统计正确数。
            num_samples += predictions.size(0)     # 统计总样本数

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}%')

    model.train()     # 恢复模型到训练模式（恢复Dropout/BatchNorm）。

# 调用函数验证训练集和测试集准确率
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

'''输出：
Checking accuracy on training data
Got 31311 / 50000 with accuracy 62.62%
Checking accuracy on test data
Got 6119 / 10000 with accuracy 61.19%
'''
