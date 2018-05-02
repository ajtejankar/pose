from datetime import datetime
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

FORMAT = '%-20s: %s'
BB_DIR = '/nfs1/shared/for_aniruddha/pose/bb'
KEYPOINTS_DIR = '/nfs1/shared/for_aniruddha/pose/keypoints'


def isDir(path):
    mode = os.stat(path)[ST_MODE]
    return S_ISDIR(mode)


# Logger to give output to console as well as a file simultaneously.
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        dt = str(datetime.datetime.now().date())
        tm = str(datetime.datetime.now().time())
        fname = 'detect_speaker_' + dt + '_' + tm.replace(':', '.') + '.log'
        self.log = open(fname, "a")  # specify file name here

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def main():
    for bb_file_path in glob.glob(BB_DIR + '/*'):
        if isDir(bb_file_path):
            continue

    for path in paths:
        if not isDir(path):
            continue

        tic = time.time()
        print(FORMAT % ('start_path', path))
        frame_paths = sorted(glob.glob(path + '/*'))
        vid_name = path.split('/')[-1]
        speaker_enc_path = os.path.join(SPEAKERS_DIR, vid_name + '.npy')

        if not os.path.exists(speaker_enc_path):
            continue

        speaker_enc = np.load(speaker_enc_path)
        speaker_bb = detectSpeakerFace(frame_paths, speaker_enc)
        
        with open(vid_name + '.json', 'w') as outfile:
            json.dump(speaker_bb, outfile)

        toc = time.time()
        print(FORMAT % ('duration', '%.5f s' % (toc-tic)))
        print(FORMAT % ('done', sep))


sys.stdout = Logger()
sys.stderr = sys.stdout
sep = ''.join(['='] * 70)

if __name__ == '__main__':
    main()
