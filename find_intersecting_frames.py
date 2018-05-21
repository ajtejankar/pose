from datetime import datetime
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
BB_DIR = '/nfs1/shared/for_aniruddha/pose/bb'
KEYPOINTS_ROOT = '/nfs1/shared/for_aniruddha/pose/keypoints'
FRAMES_ROOT = '/nfs2/datasets/TED/frames'
VID_SEQ_ROOT = '/nfs1/shared/for_aniruddha/pose/vid_seq'
SEQ_LEN = 5
FRAME_RATE = 30


def isDir(path):
    mode = os.stat(path)[ST_MODE]
    return S_ISDIR(mode)


# Logger to give output to console as well as a file simultaneously.
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        dt = str(datetime.datetime.now().date())
        tm = str(datetime.datetime.now().time())
        fname = 'intersection_' + dt + '_' + tm.replace(':', '.') + '.log'
        self.log = open(fname, "a")  # specify file name here

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def find_seq(frames):
    seq = []
    i = 0
    # print(FORMAT % ('person_frames', len(frames)))

    if len(frames) == 0:
        return []

    while i < len(frames):
        start_frame_num = int(frames[i].split('/')[-1].replace('.jpg', ''))
        start_sec = start_frame_num // FRAME_RATE

        if (i + SEQ_LEN) >= len(frames):
            break

        end_frame_num = int(frames[i + SEQ_LEN].split('/')[-1].replace('.jpg', ''))
        end_sec =  end_frame_num // FRAME_RATE
        # print(FORMAT % ('clip_seconds', '%d --> %d' % (start_sec, end_sec)))
        
        if (end_sec - start_sec) == SEQ_LEN:
            print(FORMAT % ('seq_found', '[ %d --> %d ]' % (start_sec, end_sec)))
            vid_name = frames[i].split('/')[-2]
            a = start_sec * FRAME_RATE
            b = end_sec * FRAME_RATE
            seq.append(['%s/%06d.jpg' % (vid_name, j+1) for j in range(a, b)])
            i += SEQ_LEN
        else:
            i += 1

    print(FORMAT % ('total_seq', '%-60s %d' % (frames[0].split('/')[-2], len(seq))))

    return seq


def in_box(bb, pt):
    top = bb['top']
    right = bb['right']
    bottom = bb['bottom']
    left = bb['left']

    if (left <= pt[0] <= right) and (top <= pt[1] <= bottom):
        return True

    return False


def do_intersect(kp, bb):
    # check if face bb contains the face keypoints (ears and nose)
    if in_box(bb, (kp[0*3+0], kp[0*3+1])) or \
        in_box(bb, (kp[14*3+0], kp[14*3+1])) or \
        in_box(bb, (kp[15*3+0], kp[15*3+1])):

        # check if right or left is detected
        if kp[8*3+2] != 0 or kp[11*3+2] != 0:
            return True
    
    return False


def handle_path(person_bb_fpath):
    if isDir(person_bb_fpath):
        return

    try:
        print(FORMAT % ('read_bb_json', person_bb_fpath))
        with open(person_bb_fpath, 'r') as inp:
            person_bb_dict = json.load(inp)
    except:
        print(FORMAT % ('json_load_fail', person_bb_fpath))
        return

    no_kp_file = False
    person_frames = []

    for frame_bb_key in sorted(person_bb_dict.keys()):
        frame_kp_fpath = os.path.join(KEYPOINTS_ROOT, 
                frame_bb_key).replace('.jpg', '_keypoints.json')

        person_bb = person_bb_dict[frame_bb_key]
        # print(FORMAT % ('kp_file_path', frame_kp_fpath))

        if not os.path.exists(frame_kp_fpath):
            no_kp_file = True
            break

        # print(FORMAT % ('read_keypoints', frame_kp_fpath))
        with open(frame_kp_fpath, 'r') as kpf:
            frame_kp_dict = json.load(kpf)

        intersect = False
        for person in frame_kp_dict['people']:
            person_kp = person['pose_keypoints_2d']

            if do_intersect(person_kp, person_bb):
                intersect = True
                break

        if intersect == True:
            # print(FORMAT % ('intersection', frame_bb_key))
            person_frames.append(frame_bb_key)

    if no_kp_file == True:
        print(FORMAT % ('no_kp_file', person_bb_fpath))
        return

    seq_counter = 0

    for seq in find_seq(person_frames):
        vid_name = seq[0].split('/')[-2]
        seq_name = '%s_%04d' % (vid_name, seq_counter)
        seq_dir_path = os.path.join(VID_SEQ_ROOT, seq_name)

        if os.path.exists(seq_dir_path):
            print(FORMAT % ('seq_exists', seq_dir_path))
            seq_counter += 1
            continue

        print(FORMAT % ('create_seq', seq_dir_path))
        os.makedirs(seq_dir_path)

        for frame in seq:
            frame_name = frame.split('/')[-1]
            frame_path = os.path.join(FRAMES_ROOT, frame)
            seq_frame_path = os.path.join(seq_dir_path, frame_name)
            # print(FORMAT % ('create_link', '%s --> %s' % (frame_path, seq_frame_path)))
            os.symlink(frame_path, seq_frame_path)

        seq_counter += 1

    print(sep)


sep = ''.join(['='] * 70)
sys.stdout = Logger()
sys.stderr = sys.stdout
pool = Pool(NUM_WORKERS)
pool.map(handle_path, sorted(glob.glob(BB_DIR + '/*.json')))
pool.close()
pool.join()
