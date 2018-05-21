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

# tensorboard logging
from logger import Logger

from feeder import Feeder


run_name = "0007"
work_dir = "./work_dir/" + run_name
base_lr = 0.01
# tensorboard logger
logger = Logger("./work_dir/" + run_name)
COUNTER = 0
log_interval = 100
num_epochs = 90
model_device = [0, 1]
output_device = 0
batch_size = 64

loss_choice = "CrossEntropyLoss"
optimizer_choice = "SGD"

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.audio_subnetwork = nn.Sequential(

            nn.Conv1d(1, 16, 128, stride=2, padding=64),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),


            nn.Conv1d(16, 32, 64, stride=2, padding=32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),

            nn.Conv1d(32, 64, 32, stride=2, padding=16),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.Conv1d(64, 128, 16, stride=2, padding=8),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Conv1d(128, 256, 8, stride=2, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Conv1d(256, 512, 4, stride=2, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
        )

        self.vision_subnetwork = nn.Sequential(

            nn.Conv1d(36, 64, 11, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.Conv1d(64, 128, 11, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(128, 256, 5, stride=1, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Conv1d(256, 512, 5, stride=1, padding=0),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),

        )

        self.fusion_subnetwork_conv = nn.Sequential(

            nn.Conv1d(1024, 1024, 5, stride=1, padding=2),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),

            nn.Conv1d(1024, 512, 5, stride=1, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

        )

        self.fusion_subnetwork_fc = nn.Sequential(

            nn.Linear(512*28,256*28),
            nn.ReLU(True),
            # nn.Dropout(0.5),

            nn.Linear(256*28,2),

            )

    def forward(self, pose, audio):

        pose = self.vision_subnetwork(pose)
        audio = self.audio_subnetwork(audio)

        # print(pose)
        # print(audio)

        fusion = self.fusion_subnetwork_conv(torch.cat((pose,audio),1))

        fusion = fusion.view(-1,fusion.size(1)*fusion.size(2))

        out = self.fusion_subnetwork_fc(fusion)

        return out

def print_log(str, print_time=True):
    if print_time:
        localtime = time.asctime(time.localtime(time.time()))
        str = "[ " + localtime + ' ] ' + str
    print(str)
    if True:
        with open('{}/log.txt'.format(work_dir), 'a') as f:
            print(str, file=f)


# load model
model = Model().cuda(output_device)

model = nn.DataParallel(
                    model,
                    device_ids=model_device,
                    output_device=output_device)

print_log("Model loading done.")

# load loss
if loss_choice == "CrossEntropyLoss":
    criterion = nn.CrossEntropyLoss()
elif loss_choice == "BCEWithLogitsLoss":
    criterion = nn.BCEWithLogitsLoss()
else:
    pass

# load optimizer
if optimizer_choice == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
elif optimizer_choice == "SGD":
    optimizer = optim.SGD(model.parameters(), lr = base_lr, momentum=0.9, weight_decay=0.0001)
else:
    pass

# dataloader
data_path = "./data/poseaudio_randomshiftaudio/val_data.npy"
data_audio_path = "./data/poseaudio_randomshiftaudio/val_audio_data.npy"
label_path = "./data/poseaudio_randomshiftaudio/val_label.pkl"


train_dataloader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, data_audio_path, label_path, window_size=150),
        batch_size=batch_size,
        shuffle=True,
        num_workers=128)

print_log("Training data loading done.")

data_path = "./data/poseaudio_randomshiftaudio/test_data.npy"
data_audio_path = "./data/poseaudio_randomshiftaudio/test_audio_data.npy"
label_path = "./data/poseaudio_randomshiftaudio/test_label.pkl"

val_dataloader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, data_audio_path, label_path, window_size=150),
        batch_size=batch_size,
        shuffle=False,
        num_workers=128)

print_log("Validation data loading done.")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

    # _, indices = torch.sort(output)
    # hit = [torch.equal(indices[i,:],target[i,:]) for i in range(target.size(0))]
    # return sum(hit) * 1.0 / len(hit)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if optimizer_choice == "Adam":
        lr = base_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    elif optimizer_choice == "SGD":
        lr = base_lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    else:
        pass


def to_np(x):
    return x.data.cpu().numpy()

def train(epoch):
    loss = AverageMeter()
    acc = AverageMeter()

    global COUNTER, log_interval

    lr = adjust_learning_rate(optimizer, epoch)

    model.train()

    print_log("Train epoch {}".format(epoch))

    for batch_idx, (pose_data, audio_data, label) in enumerate(train_dataloader):

        # temporary fix to unsqueeze audio data
        audio_data = torch.unsqueeze(audio_data, 1)


        if loss_choice == "CrossEntropyLoss":
            label = label.long()
        elif loss_choice == "BCEWithLogitsLoss":
            label = label.float()

            label = label.view(label.size(0), 1)

            num_classes = 2
            label = (label == torch.arange(num_classes).view(1, num_classes)).float()
        else:
            pass

        score = model(Variable(pose_data.float().cuda()), Variable(audio_data.float().cuda()))
        label = label.cuda(async=True)

        loss_batch = criterion(score, Variable(label))


        # accuracy for this batch
        acc_batch = accuracy(score.data, label)
        loss.update(loss_batch.data[0], label.size(0))
        acc.update(acc_batch[0], label.size(0))



        # backward
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()


        if batch_idx % log_interval == 0:
            print_log(
                '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
                    batch_idx, len(train_dataloader), loss.avg, lr))

            #============ TensorBoard logging ============#
            # (1) Log the scalar values
            info = {
                'training loss': loss.avg,
                'training accuracy': acc.avg,
                'optimizer_LR': optimizer.param_groups[0]['lr']
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, COUNTER)

            # (2) Log values and gradients of the parameters (histogram)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, to_np(value), COUNTER)
                logger.histo_summary(tag+'/grad', to_np(value.grad), COUNTER)

            COUNTER += log_interval


def eval(epoch):
    loss = AverageMeter()
    acc = AverageMeter()

    global COUNTER, log_interval


    model.eval()

    print_log("Eval epoch {}".format(epoch))

    for batch_idx, (pose_data, audio_data, label) in enumerate(val_dataloader):

        # temporary fix to unsqueeze audio data
        audio_data = torch.unsqueeze(audio_data, 1)


        if loss_choice == "CrossEntropyLoss":
            label = label.long()
        elif loss_choice == "BCEWithLogitsLoss":
            label = label.float()

            label = label.view(label.size(0), 1)

            num_classes = 2
            label = (label == torch.arange(num_classes).view(1, num_classes)).float()
        else:
            pass

        score = model(Variable(pose_data.float().cuda(), volatile=True), Variable(audio_data.float().cuda(), volatile=True))
        label = label.cuda(async=True)

        loss_batch = criterion(score, Variable(label))


        # accuracy for this batch
        acc_batch = accuracy(score.data, label)
        loss.update(loss_batch.data[0], label.size(0))
        acc.update(acc_batch[0], label.size(0))

        if batch_idx % log_interval == 0:
            print_log(
                '\tBatch({}/{}) done. Loss: {:.4f}'.format(
                    batch_idx, len(val_dataloader), loss.avg))

    #============ TensorBoard logging ============#
    # (1) Log the scalar values
    info = {
        'validation loss': loss.avg,
        'validation accuracy': acc.avg
    }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, COUNTER)


def process():

    for epoch in range(num_epochs):
        train(epoch)
        eval(epoch)



if __name__ == "__main__":
    process()



