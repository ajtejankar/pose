from datetime import datetime
import subprocess
import random
import math
import json
import traceback
import numpy as np
import face_recognition as fr
import glob
import datetime
import os
from stat import *
import time
import sys
import re
import dlib
from multiprocessing import Pool


NUM_WORKERS = 8
FORMAT = '%-20s: %s'
FINAL_KP_ROOT = '/nfs1/shared/for_aniruddha/pose/final_kp'
AUDIOS_ROOT = '/nfs2/datasets/TED/audio'
DATASET_ROOT = '/nfs1/shared/for_aniruddha/pose/dataset'
DATASET_DATA_ROOT = os.path.join(DATASET_ROOT, 'data')
DATASET_AUDIOS_ROOT = os.path.join(DATASET_ROOT, 'audios')
SEQ_LEN = 5
FRAME_RATE = 30

os.makedirs(DATASET_DATA_ROOT, exist_ok=True)

def isDir(path):
    mode = os.stat(path)[ST_MODE]
    return S_ISDIR(mode)


# Logger to give output to console as well as a file simultaneously.
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        dt = str(datetime.datetime.now().date())
        tm = str(datetime.datetime.now().time())
        fname = 'ds_data_' + dt + '_' + tm.replace(':', '.') + '.log'
        self.log = open(fname, "a")  # specify file name here

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def handle_path(clip_path):
    if not isDir(clip_path):
        return

    print(FORMAT % ('proc_path', clip_path))
    clip_name = clip_path.split('/')[-1]
    kp_paths = sorted(glob.glob(clip_path + '/*.json'))
    clip_audio_path = os.path.join(DATASET_AUDIOS_ROOT, clip_name + '.mp3')
    clip_mpath = os.path.join(DATASET_DATA_ROOT, clip_name + '_match.json')
    clip_no_mpath = os.path.join(DATASET_DATA_ROOT, clip_name + '_no_match.json')

    if len(kp_paths) != 150:
        print(FORMAT % ('incomplete_seq', kp_paths))
        return
    
    if not os.path.exists(clip_audio_path):
        print(FORMAT % ('no_audio_path', clip_audio_path))
        return
    
    if os.path.exists(clip_mpath) and os.path.exists(clip_no_mpath):
        print(FORMAT % ('skip_clip', clip_name))
        return

    data = []

    for idx, kp_path in enumerate(kp_paths):
        with open(kp_path, 'r') as kp_file:
            frame_kp_dict = json.load(kp_file)

        fdict = {}
        skeleton = []
        fdict['skeleton'] = skeleton
        fdict['frame_index'] = idx + 1
        data.append(fdict)

        for person in frame_kp_dict['people'][:1]:
            person_kp = person['pose_keypoints_2d']
            person_dict = {}
            person_dict['pose'] = [k for i, k in enumerate(person_kp) if i%3 != 2]
            person_dict['score'] = [k for i, k in enumerate(person_kp) if i%3 == 2]
            skeleton.append(person_dict)

    audio_paths = glob.glob(DATASET_AUDIOS_ROOT + '/*.mp3')
    audio_paths.remove(clip_audio_path)
    rand_clip_audio_path = audio_paths[random.randint(0, len(audio_paths)-1)]

    print(FORMAT % ('audio_paths', 'match: %s, no_match: %s' % 
        (clip_audio_path, rand_clip_audio_path)))

    match_dict = {}
    match_dict['data'] = data
    match_dict['label'] = 'match'
    match_dict['label_index'] = 1
    match_dict['audio'] = clip_audio_path

    no_match_dict = {}
    no_match_dict['data'] = data
    no_match_dict['label'] = 'no_match'
    no_match_dict['label_index'] = 0
    no_match_dict['audio'] = rand_clip_audio_path

    with open(clip_mpath, 'w') as data_file:
        json.dump(match_dict, data_file)

    with open(clip_no_mpath, 'w') as data_file:
        json.dump(no_match_dict, data_file)

    print(sep)


sep = ''.join(['='] * 70)
sys.stdout = Logger()
sys.stderr = sys.stdout
pool = Pool(NUM_WORKERS)
paths = sorted(glob.glob(FINAL_KP_ROOT + '/*'))
pool.map(handle_path, paths)
pool.close()
pool.join()
