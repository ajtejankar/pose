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
DATASET_ROOT = '/nfs1/shared/for_aniruddha/pose/dataset'
TRAIN_ROOT = os.path.join(DATASET_ROOT, 'train')
VAL_ROOT = os.path.join(DATASET_ROOT, 'val')
TEST_ROOT = os.path.join(DATASET_ROOT, 'test')


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


def create_dict(kind):
    kind_dict = {}

    root = ''

    if kind == 'test':
        root = TEST_ROOT
    elif kind == 'val':
        root = VAL_ROOT
    elif kind == 'train':
        root = TRAIN_ROOT

    print(FORMAT % ('kind', kind))
    print(FORMAT % ('root', root))

    for path in glob.glob(root + '/*.json'):
        file_name = path.split('/')[-1].replace('.json', '')
        
        if '_'.join(file_name.split('_')[-2:]) == 'no_match':
            label = 'no_match'
            label_index = 0
        else:
            label = 'match'
            label_index = 1

        kind_dict[file_name] = {}
        kind_dict[file_name]['label'] = label
        kind_dict[file_name]['label_index'] = label_index
        kind_dict[file_name]['has_skeleton'] = True

    with open(os.path.join(DATASET_ROOT, kind + '.json'), 'w') as f:
        json.dump(kind_dict, f)


sep = ''.join(['='] * 70)
sys.stdout = Logger()
sys.stderr = sys.stdout

create_dict('test')
create_dict('val')
create_dict('train')

