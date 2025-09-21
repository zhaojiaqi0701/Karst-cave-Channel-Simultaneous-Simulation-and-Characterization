
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DataGenerator(Dataset):
    'Generates data for PyTorch'
    def __init__(self, dpath, fpath, data_IDs, dim=(512, 512, 512), n_channels=1, transform=None):
        'Initialization'
        self.dim = dim
        self.dpath = dpath
        self.fpath = fpath
        self.data_IDs = data_IDs
        self.n_channels = n_channels
        self.transform = transform
        # self.subvol_dim= (64,64,64)
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        data_ID = self.data_IDs[index]
        X, Y = self.__data_generation(data_ID)

        if self.transform:
            X, Y = self.transform(X), self.transform(Y)

        return X, Y

    # def __data_generation(self, data_ID):
    #     'Generates data containing one sample'
    #     a = 4 # data augmentation
    #     n1, n2, n3 = self.dim
    #     m1, m2, m3 = 128, 128, 128  # cut a smaller volume
    #     # 7.7
    #     # m1, m2, m3 = self.subvol_dim  # Size of each sub-volume
    #     X = np.zeros((a, m1, m2, m3, self.n_channels), dtype=np.single)
    #     Y = np.zeros((a, m1, m2, m3, self.n_channels), dtype=np.int8)
    #     print(X.shape)
    #     print(Y.shape)
    #
    #     gx = np.fromfile(os.path.join(self.dpath, f"{data_ID}.dat"), dtype=np.single)
    #     kx = np.fromfile(os.path.join(self.fpath, f"{data_ID}.dat"), dtype=np.int8)
    #     # gx = np.reshape(gx, self.dim)
    #     # gx = np.transpose(gx)
    #     #
    #     # kx = np.reshape(kx, self.dim)
    #     # kx = np.transpose(kx)
    #     # Reshape and transpose to match the dimensions
    #     gx = np.reshape(gx, (n1, n2, n3))
    #     gx = np.transpose(gx, (2, 1, 0))  # Adjust the transpose according to your data format
    #
    #     kx = np.reshape(kx, (n1, n2, n3))
    #     kx = np.transpose(kx, (2, 1, 0))  # Adjust the transpose according to your data format
    #
    #     k1 = random.randint(0, n1 - m1-1 )
    #     k2 = random.randint(0, n2 - m2-1 )
    #     k3 = random.randint(0, n3 - m3-1 )
    #     gx = gx[k1:k1 + m1, k2:k2 + m2, k3:k3 + m3]  # randomly cut a smaller volume
    #     kx = kx[k1:k1 + m1, k2:k2 + m2, k3:k3 + m3]  # randomly cut a smaller volume
    #
    #     gm = np.mean(gx)
    #     gs = np.std(gx)
    #     gx = (gx - gm) / gs
    #     for i in range(a):
    #         X[i,] = np.reshape(np.rot90(gx, i, (2, 1)), (m1, m2, m3, self.n_channels))
    #         Y[i,] = np.reshape(np.rot90(kx, i, (2, 1)), (m1, m2, m3, self.n_channels))
    #         print(f"Input shape: { X[i,].shape}, Target shape: {Y[i,].shape}")  # 打印输入和目标的形状1
    #     return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.int8)
    #
    def __data_generation(self, data_ID):
        'Generates data containing one sample'
        a = 4 # data augmentation
        n1, n2, n3 = self.dim
        m1, m2, m3 = 128, 128, 128  # cut a smaller volume
        # 7.7
        # m1, m2, m3 = self.subvol_dim  # Size of each sub-volume
        X = np.zeros((a, m1, m2, m3, self.n_channels), dtype=np.single)
        Y = np.zeros((a, m1, m2, m3, self.n_channels), dtype=np.int8)

        gx = np.fromfile(os.path.join(self.dpath, f"{data_ID}.dat"), dtype=np.single)
        kx = np.fromfile(os.path.join(self.fpath, f"{data_ID}.dat"), dtype=np.int8)
        # gx = np.reshape(gx, self.dim)
        # gx = np.transpose(gx)
        #
        # kx = np.reshape(kx, self.dim)
        # kx = np.transpose(kx)
        # Reshape and transpose to match the dimensions
        gx = np.reshape(gx, (n1, n2, n3))
        gx = np.transpose(gx, (2, 1, 0))  # Adjust the transpose according to your data format

        kx = np.reshape(kx, (n1, n2, n3))
        kx = np.transpose(kx, (2, 1, 0))  # Adjust the transpose according to your data format

        k1 = random.randint(0, n1 - m1-1 )
        k2 = random.randint(0, n2 - m2-1 )
        k3 = random.randint(0, n3 - m3-1 )
        gx = gx[k1:k1 + m1, k2:k2 + m2, k3:k3 + m3]  # randomly cut a smaller volume
        kx = kx[k1:k1 + m1, k2:k2 + m2, k3:k3 + m3]  # randomly cut a smaller volume

        gm = np.mean(gx)
        gs = np.std(gx)
        gx = (gx - gm) / gs
        for i in range(a):
            X[i,] = np.reshape(np.rot90(gx, i, (2, 1)), (m1, m2, m3, self.n_channels))
            Y[i,] = np.reshape(np.rot90(kx, i, (2, 1)), (m1, m2, m3, self.n_channels))
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.int8)
