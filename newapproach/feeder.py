# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

# visualization
import time

class Feeder(torch.utils.data.Dataset):

    def __init__(self,
                 data_path,
                 data_audio_path,
                 label_path,
                 window_size=-1,
                 debug=False):
        self.debug = debug
        self.data_path = data_path
        self.data_audio_path = data_audio_path
        self.label_path = label_path
        self.window_size = window_size

        self.load_data()

    def load_data(self):
        # data: N C V T M

        # load label
        if '.pkl' in self.label_path:
            try:
                with open(self.label_path) as f:
                    self.sample_name, self.label = pickle.load(f)
            except:
                # for pickle file from python2
                with open(self.label_path, 'rb') as f:
                    self.sample_name, self.label = pickle.load(
                        f, encoding='latin1')
        # old label format
        elif '.npy' in self.label_path:
            self.label = list(np.load(self.label_path))
            self.sample_name = [str(i) for i in range(len(self.label))]
        else:
            raise ValueError()

        # load data
        self.data = np.load(self.data_path)
        self.data_audio = np.load(self.data_audio_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T = self.data.shape

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # get data
        data_numpy = self.data[index]

        # # fill data_upsampled_numpy
        # data_upsampled_numpy = np.zeros((self.C, int(self.AUDIO_LENGTH/2)))
        # for i in range(self.C):
        #     data_upsampled_numpy[i,:] = librosa.resample(data_numpy[i,:],30,11025)

        audio_numpy = self.data_audio[index]
        label = self.label[index]

        return data_numpy, audio_numpy, label

    def accuracy(self, score):
        rank = score.argsort()
        hit = [l in rank[i, -1:] for i, l in enumerate(self.label)]
        return sum(hit) * 1.0 / len(hit)





if __name__ == '__main__':
    data_path = "./data/NTU-RGB-D/xview/val_data.npy"
    label_path = "./data/NTU-RGB-D/xview/val_label.pkl"

    test(data_path, label_path, vid='S003C001P017R001A044')
