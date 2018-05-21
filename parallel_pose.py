#!/usr/bin/env python

from datetime import datetime
import traceback
import glob
import datetime
import os
from stat import *
import time
import sys
from multiprocessing import Pool
import subprocess

FORMAT = '%-20s: %s'
FRAMES_ROOT = '/nfs2/datasets/TED/frames'
SPEAKER_ROOT = '/nfs1/shared/for_aniruddha/pose/speakers'
# CAND_FRAME_LINKS_ROOT = '/nfs1/shared/for_aniruddha/pose/cand_frame_links'
VID_SEQ_ROOT = '/nfs1/shared/for_aniruddha/pose/vid_seq'
# KEYPOINTS_ROOT = '/nfs1/shared/for_aniruddha/pose/keypoints'
FINAL_KP_ROOT = '/nfs1/shared/for_aniruddha/pose/final_kp'
OPENPOSE_ROOT = '/home/ajinkya/openpose'
SEP = ''.join(['='] * 70)

def isDir(path):
    mode = os.stat(path)[ST_MODE]
    return S_ISDIR(mode)


# Logger to give output to console as well as a file simultaneously.
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        dt = str(datetime.datetime.now().date())
        tm = str(datetime.datetime.now().time())
        fname = 'keypoints_' + dt + '_' + tm.replace(':', '.') + '.log'
        self.log = open(fname, "a")  # specify file name here

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def handle_path(links_dir):
    if not isDir(links_dir):
        return

    tic = time.time()
    kp_dir = os.path.join(FINAL_KP_ROOT, links_dir.split('/')[-1])

    print(FORMAT % ('links_dir', links_dir))
    print(FORMAT % ('keypoints_dir', kp_dir))

    if os.path.exists(kp_dir):
        print(FORMAT % ('skip_path', kp_dir))
        return
    
    cmd = [OPENPOSE_ROOT + '/openpose.bin', \
            '--image_dir ' + links_dir, \
            '--write_json ' + kp_dir, \
            '--display 0']

    if '480p_' in links_dir:
        cmd.append('--write_images ' + kp_dir)
        cmd.append('--keypoint_scale 3')
        cmd.append('--number_people_max 1')
        cmd.append('--render_pose 1')
    else:
        cmd.append('--render_pose 0')

    cmd = ' '.join(cmd)
    subprocess.check_output(cmd.split(' '))
    toc = time.time()
    print(FORMAT % ('duration', '%.5f s' % (toc-tic)))
    print(SEP)


sys.stdout = Logger()
sys.stderr = sys.stdout
os.chdir(OPENPOSE_ROOT)

g_tic = time.time()
pool = Pool(1)
paths = sorted(glob.glob(VID_SEQ_ROOT + '/*'))
pool.map(handle_path, paths[15000:16000])
pool.close()
pool.join()
g_toc = time.time()
print(FORMAT % ('GLOBAL_DURATION', '%.5f s' % (g_toc-g_tic)))
