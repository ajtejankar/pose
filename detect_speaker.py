import numpy as np
import face_recognition as fr
import glob
import os
from stat import *
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt
import time
import sys
import re
import dlib

N_CLUSTERS = 8
BATCH_SIZE = 32
UPSAMPLE = 1
FRAME_START = 400
FRAME_END = 1300

def isDir(path):
    mode = os.stat(path)[ST_MODE]
    return S_ISDIR(mode)

def extractFaces(frame_paths):
    n = len(frame_paths)
    face_encodings = []
    enc_to_loc = []
    frame_batch = []

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

            print("%-20s %-6d %-3d" % (curr_frame_path, curr_frame_number, len(curr_locations)))

            if len(curr_locations) == 0:
                continue

            curr_encodings = fr.face_encodings(curr_frame, known_face_locations=curr_locations)

            for k in range(len(curr_encodings)):
                enc = curr_encodings[k]
                loc = curr_locations[k]
                enc_to_loc.append({'frame': curr_frame_number, 'loc': loc})
                face_encodings.append(enc)

        frame_batch = []

    return (face_encodings, enc_to_loc)


def detectSpeaker(face_encodings, enc_to_loc, vid_name):
    enc_arr = np.asanyarray(face_encodings)
    k_means = KMeans(n_clusters=N_CLUSTERS).fit(enc_arr)
    preds = k_means.predict(enc_arr)
    dists = k_means.transform(enc_arr)

    largest_cluster = np.argmax(np.unique(preds, return_counts=True)[1])
    closest_to_center = np.argmin(dists[:, largest_cluster])

    face_loc = enc_to_loc[closest_to_center]
    top, right, bottom, left = face_loc['loc']
    frame_number = face_loc['frame']

    im = cv2.imread(frame_paths[frame_number])
    cv2.rectangle(im, (left, top), (right, bottom), (0, 255, 0), 3)
    cv2.imwrite(vid_name + '.jpg', im)
    np.save(vid_name + '.npy', k_means.cluster_centers_[largest_cluster, :])

    return closest_to_center

def clipPaths(paths):
    pat = re.compile(r'([0-9]+)\.jpg$')
    sorted_paths = ['' for i in range(len(paths))]

    for path in paths:
        m = pat.search(path)
        num = int(m.group(1))
        sorted_paths[num-1] = path

    return sorted_paths[FRAME_START:FRAME_END]

for path in sys.argv[1:]:
    if not isDir(path):
        continue

    tic = time.time()
    print('====== Working on: ' + path + ' =========')
    frame_paths = clipPaths(glob.glob(path + '/*'))
    face_encodings, enc_to_loc = extractFaces(frame_paths)
    vid_name = re.sub(r'^.+\/', '', path)
    face_idx = detectSpeaker(face_encodings, enc_to_loc, vid_name)
    toc = time.time()
    print('====== Done: ' + '%.5f'%(toc-tic) + ' =========')
