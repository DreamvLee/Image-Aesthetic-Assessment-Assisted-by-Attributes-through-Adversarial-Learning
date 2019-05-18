# encoding:utf-8
from torch.utils import data
from PIL import Image
import torch as t
import numpy as np
import os
import pandas as pd


class AADBDataset(data.Dataset):

    def __init__(self, root, labelroot, transforms=None):
        self.transforms = transforms
        imgs = os.listdir(root)
        self.label = pd.read_csv(labelroot)
        self.filename = self.label['filename'].as_matrix()

        # self.imgs = [os.path.join(root, img) for img in imgs]
        self.imgs = root + self.label['filename'].as_matrix()

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.label.ix[self.label.filename == self.filename[index], range(1, 13)].as_matrix()
        pil_img = Image.open(img_path)
        pil_img = pil_img.convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)

        array = np.asarray(data)

        data = t.from_numpy(array)
        # if self.transforms:
        #     data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)

class AADBDatasetTest(data.Dataset):

    def __init__(self, root, labelroot, transforms=None):
        self.transforms = transforms
        imgs = os.listdir(root)
        self.label = pd.read_csv(labelroot)
        self.filename = self.label['filename'].as_matrix()

        # self.imgs = [os.path.join(root, img) for img in imgs]
        self.imgs = root + self.label['filename'].as_matrix()

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.label.ix[self.label.filename == self.filename[index], range(1, 13)].as_matrix()
        pil_img = Image.open(img_path)
        pil_img = pil_img.convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)

        array = np.asarray(data)

        data = t.from_numpy(array)
        # if self.transforms:
        #     data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)