import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super(SoundNet, self).__init__()

        self.conv1      = nn.Conv1d(1, 16, 64, stride=2, padding=32)
        print("Conv1", self.conv1.weight.shape, self.conv1.bias.shape)
        self.batchnorm1 = nn.BatchNorm1d(16)
        print("Bn1", self.batchnorm1.weight.shape, self.batchnorm1.bias.shape)
        self.relu1      = nn.ReLU(True)
        self.maxpool1   = nn.MaxPool1d(8, stride=8)

        self.conv2      = nn.Conv1d(16, 32, 32, stride=2, padding=16)
        print("Conv2", self.conv2.weight.shape, self.conv2.bias.shape)
        self.batchnorm2 = nn.BatchNorm1d(32)
        print("Bn2", self.batchnorm2.weight.shape, self.batchnorm2.bias.shape)
        self.relu2      = nn.ReLU(True)
        self.maxpool2   = nn.MaxPool1d(8, stride=8)

        self.conv3      = nn.Conv1d(32, 64, 16, stride=2, padding=8)
        print("Conv3", self.conv3.weight.shape, self.conv3.bias.shape)
        self.batchnorm3 = nn.BatchNorm1d(64)
        print("Bn3", self.batchnorm3.weight.shape, self.batchnorm3.bias.shape)
        self.relu3      = nn.ReLU(True)

        self.conv4      = nn.Conv1d(64, 128, 8, stride=2, padding=4)
        print("Conv4", self.conv4.weight.shape, self.conv4.bias.shape)
        self.batchnorm4 = nn.BatchNorm1d(128)
        print("Bn4", self.batchnorm4.weight.shape, self.batchnorm4.bias.shape)
        self.relu4      = nn.ReLU(True)

        self.conv5      = nn.Conv1d(128, 256, 4, stride=2, padding=2)
        print("Conv5", self.conv5.weight.shape, self.conv5.bias.shape)
        self.batchnorm5 = nn.BatchNorm1d(256)
        print("Bn5", self.batchnorm5.weight.shape, self.batchnorm5.bias.shape)
        self.relu5      = nn.ReLU(True)
        self.maxpool5   = nn.MaxPool1d(4, stride=4)

        self.conv6      = nn.Conv1d(256, 512, 4, stride=2, padding=2)
        print("Conv6", self.conv6.weight.shape, self.conv6.bias.shape)
        self.batchnorm6 = nn.BatchNorm1d(512)
        print("Bn6", self.batchnorm6.weight.shape, self.batchnorm6.bias.shape)
        self.relu6      = nn.ReLU(True)

        self.conv7      = nn.Conv1d(512, 1024, 4, stride=2, padding=2)
        print("Conv7", self.conv7.weight.shape, self.conv7.bias.shape)
        self.batchnorm7 = nn.BatchNorm1d(1024)
        print("Bn7", self.batchnorm7.weight.shape, self.batchnorm7.bias.shape)
        self.relu7      = nn.ReLU(True)

        # replace the last layer
#         self.conv8_objs = nn.Conv1d(1024, 1000, 8, stride=2)
#         print("Conv81", self.conv8_objs.weight.shape, self.conv8_objs.bias.shape)
#         self.conv8_scns = nn.Conv1d(1024, 401,  8, stride=2)
#         print("Conv82", self.conv8_scns.weight.shape, self.conv8_scns.bias.shape)

#         self.conv8      = nn.Conv1d(1024, 256, 8)
#         print("Conv8", self.conv8.weight.shape, self.conv8.bias.shape)
# #         self.batchnorm8 = nn.BatchNorm1d(256)
# #         print("Bn8", self.batchnorm8.weight.shape, self.batchnorm8.bias.shape)
#         self.relu8      = nn.ReLU(True)


    def forward(self, waveform):
        """
            Args:
                waveform (Variable): Raw 10s waveform.
        """
        if torch.cuda.is_available():
            waveform.cuda()

        print("Size of input: ", waveform.size())
        out = self.conv1(waveform)
        out = self.batchnorm1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)
        print("Size after layer 1: ", out.size())

        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        print("Size after layer 2: ", out.size())

        out = self.conv3(out)
        out = self.batchnorm3(out)
        out = self.relu3(out)
        print("Size after layer 3: ", out.size())

        out = self.conv4(out)
        out = self.batchnorm4(out)
        out = self.relu4(out)
        print("Size after layer 4: ", out.size())

        out = self.conv5(out)
        out = self.batchnorm5(out)
        out = self.relu5(out)
        out = self.maxpool5(out)
        print("Size after layer 5: ", out.size())

        out = self.conv6(out)
        out = self.batchnorm6(out)
        out = self.relu6(out)
        print("Size after layer 6: ", out.size())

        out = self.conv7(out)
        out = self.batchnorm7(out)
        out = self.relu7(out)
        print("Size after layer 7: ", out.size())

        # replace
#         p_objs = self.conv8_objs(out)
#         p_scns = self.conv8_scns(out)
#         print("Size after OBJS: ", p_objs.size())
#         print("Size after SCNS: ", p_scns.size())

#         return (nn.Softmax(dim=1)(p_objs), nn.Softmax(dim=1)(p_scns))

        # out = self.conv8(out)
        # #out = self.batchnorm8(out)
        # out = self.relu8(out)
        # print("Size after layer 8: ", out.size())

        return out


    def load_weights(self):
        bn1_bs = np.load('bn1_bs.npy')
        self.batchnorm1.bias = torch.nn.Parameter(torch.from_numpy(bn1_bs))
        bn1_ws = np.load('bn1_ws.npy')
        self.batchnorm1.weight = torch.nn.Parameter(torch.from_numpy(bn1_ws))
        bn2_bs = np.load('bn2_bs.npy')
        self.batchnorm2.bias = torch.nn.Parameter(torch.from_numpy(bn2_bs))
        bn2_ws = np.load('bn2_ws.npy')
        self.batchnorm2.weight = torch.nn.Parameter(torch.from_numpy(bn2_ws))
        bn3_bs = np.load('bn3_bs.npy')
        self.batchnorm3.bias = torch.nn.Parameter(torch.from_numpy(bn3_bs))
        bn3_ws = np.load('bn3_ws.npy')
        self.batchnorm3.weight = torch.nn.Parameter(torch.from_numpy(bn3_ws))
        bn4_bs = np.load('bn4_bs.npy')
        self.batchnorm4.bias = torch.nn.Parameter(torch.from_numpy(bn4_bs))
        bn4_ws = np.load('bn4_ws.npy')
        self.batchnorm4.weight = torch.nn.Parameter(torch.from_numpy(bn4_ws))
        bn5_bs = np.load('bn5_bs.npy')
        self.batchnorm5.bias = torch.nn.Parameter(torch.from_numpy(bn5_bs))
        bn5_ws = np.load('bn5_ws.npy')
        self.batchnorm5.weight = torch.nn.Parameter(torch.from_numpy(bn5_ws))
        bn6_bs = np.load('bn6_bs.npy')
        self.batchnorm6.bias = torch.nn.Parameter(torch.from_numpy(bn6_bs))
        bn6_ws = np.load('bn6_ws.npy')
        self.batchnorm6.weight = torch.nn.Parameter(torch.from_numpy(bn6_ws))
        bn7_bs = np.load('bn7_bs.npy')
        self.batchnorm7.bias = torch.nn.Parameter(torch.from_numpy(bn7_bs))
        bn7_ws = np.load('bn7_ws.npy')
        self.batchnorm7.weight = torch.nn.Parameter(torch.from_numpy(bn7_ws))

        conv1_bs = np.load('conv1_bs.npy')
        self.conv1.bias = torch.nn.Parameter(torch.from_numpy(conv1_bs))
        conv1_ws = np.load('conv1_ws.npy')
        conv1_ws = np.reshape(conv1_ws, self.conv1.weight.shape)
        self.conv1.weight = torch.nn.Parameter(torch.from_numpy(conv1_ws))

        conv2_bs = np.load('conv2_bs.npy')
        self.conv2.bias = torch.nn.Parameter(torch.from_numpy(conv2_bs))
        conv2_ws = np.load('conv2_ws.npy')
        conv2_ws = np.reshape(conv2_ws, self.conv2.weight.shape)
        self.conv2.weight = torch.nn.Parameter(torch.from_numpy(conv2_ws))

        conv3_bs = np.load('conv3_bs.npy')
        self.conv3.bias = torch.nn.Parameter(torch.from_numpy(conv3_bs))
        conv3_ws = np.load('conv3_ws.npy')
        conv3_ws = np.reshape(conv3_ws, self.conv3.weight.shape)
        self.conv3.weight = torch.nn.Parameter(torch.from_numpy(conv3_ws))

        conv4_bs = np.load('conv4_bs.npy')
        self.conv4.bias = torch.nn.Parameter(torch.from_numpy(conv4_bs))
        conv4_ws = np.load('conv4_ws.npy')
        conv4_ws = np.reshape(conv4_ws, self.conv4.weight.shape)
        self.conv4.weight = torch.nn.Parameter(torch.from_numpy(conv4_ws))

        conv5_bs = np.load('conv5_bs.npy')
        self.conv5.bias = torch.nn.Parameter(torch.from_numpy(conv5_bs))
        conv5_ws = np.load('conv5_ws.npy')
        conv5_ws = np.reshape(conv5_ws, self.conv5.weight.shape)
        self.conv5.weight = torch.nn.Parameter(torch.from_numpy(conv5_ws))

        conv6_bs = np.load('conv6_bs.npy')
        self.conv6.bias = torch.nn.Parameter(torch.from_numpy(conv6_bs))
        conv6_ws = np.load('conv6_ws.npy')
        conv6_ws = np.reshape(conv6_ws, self.conv6.weight.shape)
        self.conv6.weight = torch.nn.Parameter(torch.from_numpy(conv6_ws))

        conv7_bs = np.load('conv7_bs.npy')
        self.conv7.bias = torch.nn.Parameter(torch.from_numpy(conv7_bs))
        conv7_ws = np.load('conv7_ws.npy')
        conv7_ws = np.reshape(conv7_ws, self.conv7.weight.shape)
        self.conv7.weight = torch.nn.Parameter(torch.from_numpy(conv7_ws))

        # replace
#         conv81_bs = np.load('conv81_bs.npy')
#         self.conv8_objs.bias = torch.nn.Parameter(torch.from_numpy(conv81_bs))
#         conv81_ws = np.load('conv81_ws.npy')
#         conv81_ws = np.reshape(conv81_ws, self.conv8_objs.weight.shape)
#         self.conv8_objs.weight = torch.nn.Parameter(torch.from_numpy(conv81_ws))

#         conv82_bs = np.load('conv82_bs.npy')
#         self.conv8_scns.bias = torch.nn.Parameter(torch.from_numpy(conv82_bs))
#         conv82_ws = np.load('conv82_ws.npy')
#         conv82_ws = np.reshape(conv82_ws, self.conv8_scns.weight.shape)
#         self.conv8_scns.weight = torch.nn.Parameter(torch.from_numpy(conv82_ws))