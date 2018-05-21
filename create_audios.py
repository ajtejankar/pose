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
DATASET_AUDIOS_ROOT = os.path.join(DATASET_ROOT, 'rand-shift-audios')
SEQ_LEN = 5
FRAME_RATE = 30

os.makedirs(DATASET_ROOT, exist_ok=True)
os.makedirs(DATASET_AUDIOS_ROOT, exist_ok=True)

def isDir(path):
    mode = os.stat(path)[ST_MODE]
    return S_ISDIR(mode)


# Logger to give output to console as well as a file simultaneously.
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        dt = str(datetime.datetime.now().date())
        tm = str(datetime.datetime.now().time())
        fname = 'ds_audio_' + dt + '_' + tm.replace(':', '.') + '.log'
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
    vid_name = '_'.join(clip_name.split('_')[:-1])
    vid_audio_path = os.path.join(AUDIOS_ROOT, vid_name + '.mp3')
    kp_paths = sorted(glob.glob(clip_path + '/*.json'))
    clip_audio_path = os.path.join(DATASET_AUDIOS_ROOT, clip_name + '.mp3')

    if len(kp_paths) != 150:
        print(FORMAT % ('incomplete_seq', kp_paths))
        return
    
    if not os.path.exists(vid_audio_path):
        print(FORMAT % ('no_audio_path', vid_audio_path))
        return

    clip_start = int(kp_paths[0].split('/')[-1].split('_')[0]) // FRAME_RATE
    
    if not os.path.exists(clip_audio_path):
        # create clip audio
        rand_start = clip_start + (random.randint(0, 2000) / 1000)
        cmd = 'ffmpeg -ss %d -t 5 -i %s %s' % (rand_start, vid_audio_path, clip_audio_path) 
        print(FORMAT % ('clip_audio', cmd))
        subprocess.check_output(cmd.split(' '))
    else:
        print(FORMAT % ('skip_audio_clip', clip_audio_path))

    print(sep)


sep = ''.join(['='] * 70)
sys.stdout = Logger()
sys.stderr = sys.stdout
pool = Pool(NUM_WORKERS)
paths = sorted(glob.glob(FINAL_KP_ROOT + '/*'))
pool.map(handle_path, paths)
pool.close()
pool.join()
