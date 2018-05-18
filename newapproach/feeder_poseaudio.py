# sys
import os
import sys
import numpy as np
import random
import pickle
import json
# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchaudio

# visualization
import time

# librosa
import librosa


class Feeder_PoseAudio(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition in kinetics-skeleton dataset
    """

    def __init__(self,
                 data_path,
                 label_path,
                 ignore_empty_sample=True,
                 window_size=-1,
                 debug=False):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.window_size = window_size
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()

    def load_data(self):
        # load file list
        self.sample_name = os.listdir(self.data_path)

        if self.debug:
            self.sample_name = self.sample_name[0:2]

        # load label
        label_path = self.label_path
        with open(label_path) as f:
            label_info = json.load(f)

        sample_id = [name.split('.')[0] for name in self.sample_name]
        self.label = np.array(
            [label_info[id]['label_index'] for id in sample_id])
        has_skeleton = np.array(
            [label_info[id]['has_skeleton'] for id in sample_id])

        # ignore the samples which does not has skeleton sequence
        if self.ignore_empty_sample:
            self.sample_name = [
                s for h, s in zip(has_skeleton, self.sample_name) if h
            ]
            self.label = self.label[has_skeleton]

        # output data shape (N, C, T, V, M)
        self.N = len(self.sample_name)  #sample
        self.C = 36  #channel
        self.T = 150  #frame


        self.AUDIO_LENGTH = 22050*5             # audio sampling rate of 22050 Hz

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # output shape (C, T, V, M)
        # get data
        sample_name = self.sample_name[index]
        sample_path = os.path.join(self.data_path, sample_name)
        with open(sample_path, 'r') as f:
            video_info = json.load(f)

        # fill data_numpy
        data_numpy = np.zeros((self.C, self.T))
        for frame_info in video_info['data']:
            frame_index = frame_info['frame_index'] - 1                     # Aniruddha - to fix array index out of bounds error
            for m, skeleton_info in enumerate(frame_info["skeleton"]):
                pose = skeleton_info['pose']
                # score = skeleton_info['score']                            # Not using the scores
                data_numpy[:,frame_index] = pose



        # # fill data_resampled_numpy
        # data_resampled_numpy = np.zeros((self.C, self.AUDIO_LENGTH))
        # for i in range(self.C):
        #     data_resampled_numpy[i,:] = librosa.resample(data_numpy[i,:],30,22050)

        # get & check label index
        label = video_info['label_index']
        assert (self.label[index] == label)

        # fill audio_numpy

        audio_numpy,_ = librosa.load(video_info['audio'], sr=22050, mono=True)
        audio_numpy = audio_numpy[:self.AUDIO_LENGTH]                            # clip because ffmpeg returns slightly longer sequences - convert to mono



        return data_numpy, audio_numpy, label

    # def top_k(self, score, top_k):
    #     assert (all(self.label >= 0))

    #     rank = score.argsort()
    #     hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
    #     return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def accuracy(self, score):
        assert (all(self.label >= 0))

        rank = score.argsort()
        hit = [l in rank[i, -1:] for i, l in enumerate(self.label)]
        return sum(hit) * 1.0 / len(hit)

if __name__ == '__main__':
    data_path = '../../data/kinetics-skeleton/kinetics_val'
    label_path = '../../data/kinetics-skeleton/kinetics_val_label.json'
    graph = 'st_gcn.graph.Kinetics'
    # test(data_path, label_path, vid='iqkx0rrCUCo', graph=graph)
    test(data_path, label_path, vid=11111, graph=graph)
    # test(data_path, label_path, vid = 11199, graph=graph)
