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
DATASET_ROOT = '/nfs1/shared/for_aniruddha/pose/dataset/rand-shift'
DATASET_AUDIOS_ROOT = '/nfs1/shared/for_aniruddha/pose/dataset/shift-rand-audios'


# Logger to give output to console as well as a file simultaneously.
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        dt = str(datetime.datetime.now().date())
        tm = str(datetime.datetime.now().time())
        fname = 'ds_shift_' + dt + '_' + tm.replace(':', '.') + '.log'
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
    print(FORMAT % ('proc_path', clip_path))
    clip_name = clip_path.split('/')[-1].replace('.json', '')
    clip_name = '_'.join(clip_name.split('_')[:-2])
    print(FORMAT % ('clip_name', clip_name))

    with open(clip_path, 'r') as f:
        d = json.load(f)

    d['audio'] = os.path.join(DATASET_AUDIOS_ROOT, clip_name + '.mp3')
    print(FORMAT % ('audio_file', d['audio']))

    with open(clip_path, 'w') as f:
        json.dump(d, f)

    print(sep)


sep = ''.join(['='] * 70)
sys.stdout = Logger()
sys.stderr = sys.stdout
pool = Pool(NUM_WORKERS)
paths = sorted(glob.glob(DATASET_ROOT + '/*/*_no_match.json'))
pool.map(handle_path, paths)
pool.close()
pool.join()
