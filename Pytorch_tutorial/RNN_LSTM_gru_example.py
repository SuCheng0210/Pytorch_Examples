#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/20 22:26
# @Author  : Nicole Sue
# @File    : RNN_LSTM_gru_example.py
# @Project: PyTorch_Tutorials

# Imports
import torch     # 导入PyTorch核心库，提供张量操作、自动求导和GPU加速等功能，是构建神经网络的基础。
import torch.nn as nn     # 导入PyTorch的神经网络模块（nn），包含各种层（如全连接层Linear、卷积层Conv2d）、激活函数和损失函数等，用于定义模型结构。
import torch.optim as optim     # 导入优化算法模块（optim），包含常见的优化器如SGD、Adam等，用于在训练过程中更新模型参数。
import torch.nn.functional as F     # 导入函数式接口模块（functional），提供无状态的函数操作，如激活函数（F.relu）、池化（F.max_pool2d）和损失函数（F.cross_entropy），通常在模型前向传播中直接调用。
from torch.utils.data import DataLoader     # 导入数据加载工具（DataLoader），用于批量加载数据、支持多线程和数据打乱，简化数据输入到模型的过程。
import torchvision.datasets as datasets     # 导入PyTorch视觉库中的数据集模块（datasets），提供常用数据集（如MNIST、CIFAR-10）的快速加载接口。
import torchvision.transforms as transforms     # 导入数据预处理模块（transforms），用于图像预处理（如调整尺寸、转换为张量、标准化），常通过Compose组合多个操作。

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 28     # input_size：输入数据的特征维度。
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10     # num_classes：分类任务的类别数。
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# Create a RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Nxtime_seqxfeatures
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)
        '''When LSTM:
        self.fc = nn.Linear(hidden_size, num_classes)
        '''

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        '''When LSTM, add:
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        '''

        # Forward Prop
        out, _ = self.rnn(x, h0)
        # out, _ = self.gru(x, h0)
        # out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.size(0), -1)
        # When LSTM, delete this line
        out = self.fc(out)
        # When LSTM, out = self.fc(out[:,-1,:])
        return out

# Load data，用了MNIST 手写数字数据集
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
# 数据预处理：通过transform=transforms.ToTensor()将图像转换为PyTorch张量，并自动归一化像素值到[0, 1]范围。
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# 创建数据加载器DataLoader，支持批量加载和随机打乱（仅训练集需要shuffle=True）。
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
# train=True/False：加载训练集或测试集。
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
# batch_size：每个批次的样本数（需提前定义）。
# shuffle=True：仅对训练集打乱顺序，增强泛化性。

# Initialize network
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# 交叉熵损失函数，适用于多分类任务（内部集成Softmax）。
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Adam优化器，用于更新模型参数，需传入学习率。

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible, 数据迁移：将当前批次的输入数据data和标签targets移动到指定设备（GPU或CPU）。
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)
        '''相比FN需删去的部分：
        # Get to correct shape, 调整数据形状为 [batch_size, input_size]
        data = data.reshape(data.shape[0], -1)
        # 数据展平：原始MNIST图像形状为[batch_size, 1, 28, 28]，通过reshape展平为[batch_size, 784]以适应全连接层输入。
        '''
        # forward, 前向传播
        scores = model(data)     # scores（未归一化的类别分数）
        loss = criterion(scores, targets)     # 损失计算：通过交叉熵损失函数计算预测值与真实标签的误差。

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
            x = x.to(device=device).squeeze(1)
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
Got 58104 / 60000 with accuracy 96.84%
Checking accuracy on test data
Got 9697 / 10000 with accuracy 96.97%
'''