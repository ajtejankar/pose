import os
import re
import sys
import cv2
import math
import time
import scipy
import argparse
import matplotlib
from torch import np
import pylab as plt
from joblib import Parallel, delayed
import util
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter

import glob
import os
from stat import *
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import json
#parser = argparse.ArgumentParser()
#parser.add_argument('--t7_file', required=True)
#parser.add_argument('--pth_file', required=True)
#args = parser.parse_args()

torch.set_num_threads(torch.get_num_threads())
weight_name = './model/pose_model.pth'

blocks = {}

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
           [1,16], [16,18], [3,17], [6,18]]

# the middle joints heatmap correpondence
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
          [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
          [55,56], [37,38], [45,46]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


block0  = [{'conv1_1':[3,64,3,1,1]},{'conv1_2':[64,64,3,1,1]},{'pool1_stage1':[2,2,0]},{'conv2_1':[64,128,3,1,1]},{'conv2_2':[128,128,3,1,1]},{'pool2_stage1':[2,2,0]},{'conv3_1':[128,256,3,1,1]},{'conv3_2':[256,256,3,1,1]},{'conv3_3':[256,256,3,1,1]},{'conv3_4':[256,256,3,1,1]},{'pool3_stage1':[2,2,0]},{'conv4_1':[256,512,3,1,1]},{'conv4_2':[512,512,3,1,1]},{'conv4_3_CPM':[512,256,3,1,1]},{'conv4_4_CPM':[256,128,3,1,1]}]

blocks['block1_1']  = [{'conv5_1_CPM_L1':[128,128,3,1,1]},{'conv5_2_CPM_L1':[128,128,3,1,1]},{'conv5_3_CPM_L1':[128,128,3,1,1]},{'conv5_4_CPM_L1':[128,512,1,1,0]},{'conv5_5_CPM_L1':[512,38,1,1,0]}]

blocks['block1_2']  = [{'conv5_1_CPM_L2':[128,128,3,1,1]},{'conv5_2_CPM_L2':[128,128,3,1,1]},{'conv5_3_CPM_L2':[128,128,3,1,1]},{'conv5_4_CPM_L2':[128,512,1,1,0]},{'conv5_5_CPM_L2':[512,19,1,1,0]}]

for i in range(2,7):
    blocks['block%d_1'%i]  = [{'Mconv1_stage%d_L1'%i:[185,128,7,1,3]},{'Mconv2_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv4_stage%d_L1'%i:[128,128,7,1,3]},
{'Mconv5_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv6_stage%d_L1'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L1'%i:[128,38,1,1,0]}]
    blocks['block%d_2'%i]  = [{'Mconv1_stage%d_L2'%i:[185,128,7,1,3]},{'Mconv2_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv4_stage%d_L2'%i:[128,128,7,1,3]},
{'Mconv5_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv6_stage%d_L2'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L2'%i:[128,19,1,1,0]}]

def make_layers(cfg_dict):
    layers = []
    for i in range(len(cfg_dict)-1):
        one_ = cfg_dict[i]
        for k,v in one_.iteritems():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    one_ = cfg_dict[-1].keys()
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
    layers += [conv2d]
    return nn.Sequential(*layers)

layers = []
for i in range(len(block0)):
    one_ = block0[i]
    for k,v in one_.iteritems():
        if 'pool' in k:
            layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
            layers += [conv2d, nn.ReLU(inplace=True)]

models = {}
models['block0']=nn.Sequential(*layers)

for k,v in blocks.iteritems():
    models[k] = make_layers(v)

class pose_model(nn.Module):
    def __init__(self,model_dict,transform_input=False):
        super(pose_model, self).__init__()
        self.model0   = model_dict['block0']
        self.model1_1 = model_dict['block1_1']
        self.model2_1 = model_dict['block2_1']
        self.model3_1 = model_dict['block3_1']
        self.model4_1 = model_dict['block4_1']
        self.model5_1 = model_dict['block5_1']
        self.model6_1 = model_dict['block6_1']

        self.model1_2 = model_dict['block1_2']
        self.model2_2 = model_dict['block2_2']
        self.model3_2 = model_dict['block3_2']
        self.model4_2 = model_dict['block4_2']
        self.model5_2 = model_dict['block5_2']
        self.model6_2 = model_dict['block6_2']

    def forward(self, x):
        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2  = torch.cat([out1_1,out1_2,out1],1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3   = torch.cat([out2_1,out2_2,out1],1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4   = torch.cat([out3_1,out3_2,out1],1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5   = torch.cat([out4_1,out4_2,out1],1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6   = torch.cat([out5_1,out5_2,out1],1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        return out6_1,out6_2


model = pose_model(models)
model.load_state_dict(torch.load(weight_name))
model.cuda()
model.float()
model.eval()

param_, model_ = config_reader()


def handle_one(oriImg, bounding_box):

    #oriImg = cv2.imread(oriImgpath)
    # for visualize
    canvas = np.copy(oriImg)
    canvas_white = np.ones(oriImg.shape)
    canvas_white[:] = (255, 255, 255)
    imageToTest = Variable(T.transpose(T.transpose(T.unsqueeze(torch.from_numpy(oriImg).float(),0),2,3),1,2),volatile=True).cuda()
    #print oriImg.shape
    scale = model_['boxsize'] / float(oriImg.shape[0])
    #print scale
    h = int(oriImg.shape[0]*scale)
    w = int(oriImg.shape[1]*scale)
    pad_h = 0 if (h%model_['stride']==0) else model_['stride'] - (h % model_['stride'])
    pad_w = 0 if (w%model_['stride']==0) else model_['stride'] - (w % model_['stride'])
    new_h = h+pad_h
    new_w = w+pad_w

    imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_['stride'], model_['padValue'])
    imageToTest_padded = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5

    feed = Variable(T.from_numpy(imageToTest_padded)).cuda()

    output1,output2 = model(feed)

    heatmap = nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])).cuda()(output2)

    paf = nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])).cuda()(output1)

    #print heatmap.size()
    #print paf.size()
    #print type(heatmap)
    heatmap_avg = T.transpose(T.transpose(heatmap[0],0,1),1,2).data.cpu().numpy()
    paf_avg = T.transpose(T.transpose(paf[0],0,1),1,2).data.cpu().numpy()

    all_peaks = []
    peak_counter = 0

    #maps =
    for part in range(18):
        map_ori = heatmap_avg[:,:,part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]

        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param_['thre1']))
    #    peaks_binary = T.eq(
    #    peaks = zip(T.nonzero(peaks_binary)[0],T.nonzero(peaks_binary)[0])

        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse

        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)





    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:,:,[x-19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0]-1]
        candB = all_peaks[limbSeq[k][1]-1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1]) + 1e-8          # changed
                    vec = np.divide(vec, norm)

                    startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num))

                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                      for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                      for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)
                    criterion1 = len(np.nonzero(score_midpts > param_['thre2'])[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if(len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:,0]
            partBs = connection_all[k][:,1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])): #= 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)): #1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if(subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    #print "found = 2"
                    membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: #merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

#    canvas = cv2.imread(test_image) # B,G,R order
    # for i in range(18):
    #     for j in range(len(all_peaks[i])):
    #         cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
    #         cv2.circle(canvas_white, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    # separate joints per skeleton
    joints_per_skeleton = [[] for i in range(len(subset))]
    for i in range(18):
        #print(i)
        for n in range(len(subset)):
            #print n
            index = int(subset[n][i])
            #print index
            if -1 == index:
                joints_per_skeleton[n].append(None)
                continue
            Y = candidate[index, 0]
            X = candidate[index, 1]
            #print(Y,X)
            joints_per_skeleton[n].append((Y,X))

    flag = False
    speaker_skeleton_index = -1
    # loop through all the skeletons
    for i in range(len(joints_per_skeleton)):
        # if left eye and right eye or nose inside bounding box, say this is speaker's skeleton
        if isinsidebox(bounding_box, joints_per_skeleton[i][0]) or isinsidebox(bounding_box, joints_per_skeleton[i][15]) or isinsidebox(bounding_box, joints_per_skeleton[i][14]):
            speaker_skeleton_index = i

            # check if either left or right hip of speaker is visible
            if joints_per_skeleton[i][8] is not None or joints_per_skeleton[i][11] is not None:
                flag = True

            break
        else:
            continue

    if speaker_skeleton_index != -1
        stickwidth = 4

        for i in range(17):
            for n in range(speaker_skeleton_index,speaker_skeleton_index+1):
                index = subset[n][np.array(limbSeq[i])-1]
                if -1 in index:
                    continue
                cur_canvas = canvas.copy()
                cur_canvas_white = canvas_white.copy()
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                cv2.fillConvexPoly(cur_canvas_white, polygon, colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
                canvas_white = cv2.addWeighted(canvas_white, 0.4, cur_canvas_white, 0.6, 0)

    #cv2.imwrite(os.path.join(newdirpath,file),poseoverlayframe)
    #cv2.imwrite(os.path.join(newdirpath2,file),poseonlyframe)
    return canvas, canvas_white, flag, joints_per_skeleton[speaker_skeleton_index]

def sortPaths(paths):
    pat = re.compile(r'([0-9]+)\.jpg$')
    sorted_paths = ['' for i in range(len(paths))]

    for path in paths:
        m = pat.search(path)
        num = int(m.group(1))
        sorted_paths[num-1] = path

    return sorted_paths[400:-160]           #TED intro and exit in all the other frames

SEQUENCE_LENGTH = 30*10   # we are considering 10 second sequences

def filterfaceframes(frame_paths, speaker_dict):
    temp_paths = []
    final_paths = []

    for path in frame_paths:
        if speaker_dict.has_key(path) and len(speaker_dict[path].keys()) > 0:
            temp_paths.append(path)
        else:
            #print(path)
            if len(temp_paths) >= SEQUENCE_LENGTH:
                final_paths += temp_paths
            temp_paths = []

    if len(temp_paths) >= SEQUENCE_LENGTH:
        final_paths += temp_paths

    return final_paths

def isinsidebox(bounding_box,point):
    if point is None:
        return False
    if (left <= point[0] <= right) and (top <= point[1] <= bottom):
        return True

if __name__ == "__main__":


    for i,json_path in enumerate(sorted(glob.glob('/nfs1/code/aniruddha/sample_vids/*.json'))):
        if i in range(0,3):
            inp = open(json_path, 'r')
            print('====== Working on: ' + json_path + ' =========')
            tic = time.time()
            # load json into dictionary
            info = json.load(inp)

            inp.close()


            # get the path for frames
            vid_path = json_path.replace('.json','')
            frame_paths = sorted(glob.glob(vid_path + '/*'))[400:-160]

            # modify the path so that it matches the dictionary key format
            key_paths = ['/'.join(path.split('/')[-2:]) for path in frame_paths]


            # frames where speaker is detected
            print('====== Filtering face frames =======')
            speaker_face_detected_paths = filterfaceframes(key_paths, info)

            # add prefix
            prefix = '/'.join(path.split('/')[:5])
            #speaker_face_detected_paths = [os.path.join(prefix,path) for path in speaker_face_detected_paths]

            # dump speaker face frames into json for future use
            with open('/'.join(vid_path.split('/')[:-1]) + '/speakerfaceframesjson/' + vid_path.split('/')[-1] + '_speakerfaceframes.json', 'w') as fp:
                json.dump(speaker_face_detected_paths, fp)

            # frames where speaker upper body detected given that speaker is detected
            speaker_upperbody_detected_paths = []

            # speaker skeleton info
            speaker_skeleton_info = []

            newdirpath1 = vid_path + '_pose_only_frames'
            newdirpath2 = vid_path + '_pose_overlay_frames'

            print(newdirpath1)
            print(newdirpath2)

            if not os.path.exists(newdirpath1):
                os.makedirs(newdirpath1)
            if not os.path.exists(newdirpath2):
                os.makedirs(newdirpath2)

            for path in speaker_face_detected_paths:
                print(path)
                frame = cv2.imread(os.path.join(prefix,path))

                # access bounding box of speaker face
                loc = info[path]
                top = loc['top']
                right = loc['right']
                bottom = loc['bottom']
                left = loc['left']

                # given bounding box of face, find skeleton of speaker
                bounding_box = (left, top, bottom, right)

                poseoverlayframe, poseonlyframe, flag, joints = handle_one(frame, bounding_box)

                frameonlypath = path.split('/')[-1]
                cv2.imwrite(os.path.join(newdirpath1, frameonlypath),poseonlyframe)
                cv2.imwrite(os.path.join(newdirpath2, frameonlypath),poseoverlayframe)

                speaker_skeleton_info.append(joints)


                if flag == True:
                    speaker_upperbody_detected_paths.append(path)


                # # loop through all the skeletons
                # for i in range(len(joints)):

                #     # if left eye and right eye or nose inside bounding box, say this is speaker's skeleton
                #     if isinsidebox(bounding_box, joints[i][0]) or isinsidebox(bounding_box, joints[i][15]) or isinsidebox(bounding_box, joints[i][14]):
                #         #speaker_skeleton_index = i

                #         # check if either left or right hip of speaker is visible
                #         if joints[i][8] is not None or joints[i][11] is not None:
                #             speaker_upperbody_detected_paths.append(path)
                #             speaker_skeleton_info.append(joints[i])
                #             print('match ' + path)

                #         break
                #     else:
                #         continue


            # dump speaker face frames into json for future use
            with open('/'.join(vid_path.split('/')[:-1]) + '/speakerupperbodyframesjson/' + vid_path.split('/')[-1] + '_speakerupperbodyframes.json', 'w') as fp:
                json.dump(speaker_upperbody_detected_paths, fp)

            # dump speaker face frames into json for future use
            with open('/'.join(vid_path.split('/')[:-1]) + '/speakerskeletonjson/' + vid_path.split('/')[-1] + '_speakerskeleton.json', 'w') as fp:
                json.dump(speaker_skeleton_info, fp)

            # for all speaker_upper_body detected frames, extract int frame numbers:
            framenumber_list = []
            for i,text in enumerate(speaker_upperbody_detected_paths):
                framenumber = text.split('/')[-1]
                framenumber = framenumber.replace('.jpg','')
                framenumber = int(framenumber)
                framenumber_list.append(framenumber)

            # find out sequence start frames which last 10 seconds
            sequence_start_frames = []
            k = 0
            i = 0
            while i < len(framenumber_list):
                if (i+SEQUENCE_LENGTH) >= len(framenumber_list):
                    break
                if framenumber_list[i+SEQUENCE_LENGTH] <= (framenumber_list[i] + 1.05*SEQUENCE_LENGTH): # 5% frame drop allowed
                    sequence_start_frames.append(info[i])
                    i = i + SEQUENCE_LENGTH + 1
                else:
                    i = i + 1

            with open('/'.join(vid_path.split('/')[:-1]) + '/sequencestartframesjson/' + vid_path.split('/')[-1] +'_sequencestartframes.json','w') as fp:
                json.dump(sequence_start_frames,fp)

            toc = time.time()
            print('====== Done: ' + '%.5f'%(toc-tic) + ' =========')

    #print(os.path.join(newdirpath,file))

    # ticbegin = time.time()
    # filepath = './sample_image/000460.jpg'
    # print(filepath)
    # frame = cv2.imread(filepath) #BGR
    # poseoverlayframe,poseonlyframe = handle_one(frame)
    # tocend = time.time()
    # print 'time is %.5f'%(tocend-ticbegin)
    # cv2.imwrite("./result.jpg",poseoverlayframe)
