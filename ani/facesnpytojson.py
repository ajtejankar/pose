import numpy as np
import json
import re
import glob
import time
import face_recognition as fr


N_CLUSTERS = 8
BATCH_SIZE = 32
UPSAMPLE = 1

def getFrameInfo(frame_paths, speaker_enc):
    n = len(frame_paths)
    frame_batch = []
    frame_info = {}

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
            res = fr.compare_faces(curr_encodings, speaker_enc)

            for k in range(len(curr_encodings)):
                enc = curr_encodings[k]
                loc = curr_locations[k]
                curr_frame_name = '/'.join(curr_frame_path.split('/')[-2:])
                frame_info[curr_frame_name] = {}

                if res[k] == True:
                    top, right, bottom, left = curr_locations[k]
                    frame_info[curr_frame_name]['top'] = top
                    frame_info[curr_frame_name]['right'] = right
                    frame_info[curr_frame_name]['bottom'] = bottom
                    frame_info[curr_frame_name]['left'] = left


        frame_batch = []

    return frame_info

def sortPaths(paths):
    pat = re.compile(r'([0-9]+)\.jpg$')
    sorted_paths = ['' for i in range(len(paths))]

    for path in paths:
        m = pat.search(path)
        num = int(m.group(1))
        sorted_paths[num-1] = path

    return sorted_paths[400:-160]           #TED intro and exit in all the other frames

def npytojson():
    for path in glob.glob('../sample_vids/*.npy'):
        tic = time.time()
        print('====== Working on: ' + path + ' =========')
        vid_path = path.replace('.npy', '')
        frame_paths = sortPaths(glob.glob(vid_path + '/*'))
        vid_name = re.sub(r'^.+\/', '', vid_path)
        speaker_enc = np.load(path)
        frame_info = getFrameInfo(frame_paths, speaker_enc)

        with open(vid_path + '.json', 'w') as outfile:
            json.dump(frame_info, outfile)

        toc = time.time()
        print('====== Done: ' + '%.5f'%(toc-tic) + ' =========')

if __name__=="__main__":
    npytojson()
