#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/3 17:31
# @Author  : Nicole Sue
# @File    : custom_dataset.py
# @Project: PyTorch_Tutorials

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)     # 2500

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)
            return (image, y_label)
