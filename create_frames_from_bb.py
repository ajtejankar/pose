from stat import *
import glob
import json
import os
import cv2

BB_DIR = '/nfs1/shared/for_aniruddha/pose/bb'
FRAMES_ROOT = '/nfs2/datasets/TED/frames'

def isDir(path):
    mode = os.stat(path)[ST_MODE]
    return S_ISDIR(mode)

for bb_file_path in glob.glob(BB_DIR + '/AditiShankardass*'):
    if isDir(bb_file_path):
        continue

    with open(bb_file_path, 'r') as inp:
        bb = json.load(inp)

    vid_name = bb_file_path.split('/')[-1].replace('.json', '')
    os.makedirs(vid_name, exist_ok=True)
    speaker_frame_count = 0

    for path in glob.glob(os.path.join(FRAMES_ROOT, vid_name) + '/*'):
        key = '/'.join(path.split('/')[-2:])
        frame_name = path.split('/')[-1]
        out_path = os.path.join(vid_name, frame_name)

        if key not in bb:
            print(key)
            im = cv2.imread(path)
            cv2.imwrite(out_path, im[:, :, 2])
        else:
            loc = bb[key]
            im = cv2.imread(path)
            top = loc['top']
            right = loc['right']
            bottom = loc['bottom']
            left = loc['left']
            cv2.rectangle(im, (left, top), (right, bottom), (0, 255, 0), 3)
            cv2.imwrite(out_path, im)
            speaker_frame_count += 1

    print('correct: ' + str(speaker_frame_count))
