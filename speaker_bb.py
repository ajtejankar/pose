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
import json

BATCH_SIZE = 36
UPSAMPLE = 1
FORMAT = '%-20s: %s'
SPEAKERS_DIR = '/nfs1/shared/for_aniruddha/pose/speakers'

def isDir(path):
    mode = os.stat(path)[ST_MODE]
    return S_ISDIR(mode)


# Logger to give output to console as well as a file simultaneously.
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        dt = str(datetime.datetime.now().date())
        tm = str(datetime.datetime.now().time())
        fname = 'speaker_bb_' + dt + '_' + tm.replace(':', '.') + '.log'
        self.log = open(fname, "a")  # specify file name here

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def detectSpeakerFace(frame_paths, speaker_enc):
    n = len(frame_paths)
    frame_batch = []
    speaker_bb = {}

    for frame_counter in range(n):
        frame = fr.load_image_file(frame_paths[frame_counter])
        frame_batch.append(frame)

        if frame_counter != n - 1 and len(frame_batch) != BATCH_SIZE:
            continue

        loc_batch = fr.batch_face_locations(frame_batch, number_of_times_to_upsample=UPSAMPLE)

        for frame_number_in_batch, curr_locations in enumerate(loc_batch):
            curr_frame_number = frame_counter + 1 - len(frame_batch) + frame_number_in_batch
            curr_frame_path = frame_paths[curr_frame_number]
            curr_frame = frame_batch[frame_number_in_batch]

            m = ('%-20s %-6d %-3d' % (curr_frame_path, curr_frame_number, len(curr_locations)))
            print(FORMAT % ('detect_frame', m))

            if len(curr_locations) == 0:
                continue

            curr_encodings = fr.face_encodings(curr_frame, known_face_locations=curr_locations)
            res = fr.compare_faces(curr_encodings, speaker_enc)
            # TODO: find the res[k] == True such that distance to speaker is minimized

            for k in range(len(curr_encodings)):
                enc = curr_encodings[k]
                loc = curr_locations[k]
                curr_frame_name = '/'.join(curr_frame_path.split('/')[-2:])

                if res[k] == True:
                    top, right, bottom, left = curr_locations[k]
                    speaker_bb[curr_frame_name] = {}
                    speaker_bb[curr_frame_name]['top'] = top
                    speaker_bb[curr_frame_name]['right'] = right
                    speaker_bb[curr_frame_name]['bottom'] = bottom
                    speaker_bb[curr_frame_name]['left'] = left


        frame_batch = []

    return speaker_bb


def handlePaths(paths):
    for path in paths:
        if not isDir(path):
            continue

        vid_name = path.split('/')[-1]

        if os.path.exists(vid_name + '.json'):
            print(FORMAT % ('skip_path', path))
            continue

        tic = time.time()
        print(FORMAT % ('start_path', path))
        frame_paths = sorted(glob.glob(path + '/*'))

        if '480p_' not in path:
            speaker_npy = vid_name
        else:
            speaker_npy = '_'.join(vid_name.split('_')[:-1])

        speaker_enc_path = os.path.join(SPEAKERS_DIR, speaker_npy + '.npy')

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
    if len(sys.argv) < 2:
        print('Incorrect usage. Needs paths to video frames to process.')
        sys.exit(1)

    paths = sys.argv[1:]
    print(FORMAT % ('all_devices', str(dlib.cuda.get_num_devices())))
    print(FORMAT % ('gpu_device', str(dlib.cuda.get_device())))

    try:
        handlePaths(paths)
    except Exception as e:
        print(FORMAT % ('error', traceback.format_exc()))

