from datetime import datetime
import subprocess
import random
import math
import json
import traceback
import numpy as np
import glob
import datetime
import os
from stat import *
import time
import sys
import re
import dlib
from multiprocessing import Pool


FORMAT = '%-20s: %s'
NUM_WORKERS = 8
CLIP_PATHS_FILE = '/nfs1/shared/for_aniruddha/pose/lock_clip_paths.txt'
FINAL_KP_ROOT = '/nfs1/shared/for_aniruddha/pose/final_kp'
DATASET_ROOT = '/nfs1/shared/for_aniruddha/pose/dataset'
DATASET_DATA_ROOT = os.path.join(DATASET_ROOT, 'data')
TRAIN_ROOT = os.path.join(DATASET_ROOT, 'train')
VAL_ROOT = os.path.join(DATASET_ROOT, 'val')
TEST_ROOT = os.path.join(DATASET_ROOT, 'test')
SEQ_LEN = 5
FRAME_RATE = 30


os.makedirs(TRAIN_ROOT, exist_ok=True)
os.makedirs(VAL_ROOT, exist_ok=True)
os.makedirs(TEST_ROOT, exist_ok=True)


def isDir(path):
    mode = os.stat(path)[ST_MODE]
    return S_ISDIR(mode)


# Logger to give output to console as well as a file simultaneously.
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        dt = str(datetime.datetime.now().date())
        tm = str(datetime.datetime.now().time())
        fname = 'ds_split_' + dt + '_' + tm.replace(':', '.') + '.log'
        self.log = open(fname, "a")  # specify file name here

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def handle_path(tup):
    kind, clip_path = tup

    if not isDir(clip_path):
        return

    print(FORMAT % ('proc_path', '%-6s : %s' % (kind, clip_path) ))
    clip_name = clip_path.split('/')[-1]
    clip_mpath = os.path.join(DATASET_DATA_ROOT, clip_name + '_match.json')
    clip_no_mpath = os.path.join(DATASET_DATA_ROOT, clip_name + '_no_match.json')
    print(FORMAT % ('clip_mpath', clip_mpath))

    if not (os.path.exists(clip_mpath) and os.path.exists(clip_no_mpath)):
        print(FORMAT % ('skip_clip', clip_name))
        return
    
    root = ''

    if kind == 'test':
        root = TEST_ROOT
    elif kind == 'val':
        root = VAL_ROOT
    if kind == 'train':
        root = TRAIN_ROOT
    
    cmd = 'cp %s %s %s' % (clip_mpath, clip_no_mpath, root)
    print(FORMAT % ('run_cmd', cmd))
    subprocess.check_output(cmd.split(' '))

    print(sep)


sep = ''.join(['='] * 70)
sys.stdout = Logger()
sys.stderr = sys.stdout
pool = Pool(NUM_WORKERS)

with open(CLIP_PATHS_FILE, 'r') as cpf:
    paths = cpf.readlines()

tup_paths = []

for i, p in enumerate(paths):
    if i < 3000:
        tup_paths.append(('test', p.strip()))
    elif i >= 3000 and i < 9000:
        tup_paths.append(('val', p.strip()))
    else:
        tup_paths.append(('train', p.strip()))

pool.map(handle_path, tup_paths)
pool.close()
pool.join()
