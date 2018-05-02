''' Author: Aniruddha Saha
    Date: 04/30/2018 '''


import os
import sys
import glob
import subprocess
import random
from datetime import datetime


import face_recognition as fr


#Logger to give output to console as well as a file simultaneously.
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("test.log", "w")  # specify file name here



    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

sys.stdout = Logger()


videos_root = "/nfs2/videos/TED/videos"
frames_root = "/nfs2/datasets/TED/frames"
json_root = "/nfs1/data/aniruddha/openpose/json_output"
SEQUENCE_LENGTH = 10          # 10 seconds


# TODO for Ajinkya- refine this function, return a True/False list along with dictionary of face bounding boxes

def isinsidebox(bounding_box,keypoint):

    if keypoint is None:                   # keypoint information not available
        return False
    if (left <= keypoint[0] <= right) and (top <= keypoint[1] <= bottom):
        return True

def process_json(jsonpathlist, bounding_boxes):

    speaker_upperbody_found = [False] * len(jsonpathlist)

    for i,jsonpath in enumerate(jsonpathlist):
        with open(jsonpath,'r') as inp:
            openpose_info = json.load(inp)

            num_people = len(openpose_info['people'])

            # // Result for COCO (18 body parts)
            # // POSE_COCO_BODY_PARTS {
            # //     {0,  "Nose"},
            # //     {1,  "Neck"},
            # //     {2,  "RShoulder"},
            # //     {3,  "RElbow"},
            # //     {4,  "RWrist"},
            # //     {5,  "LShoulder"},
            # //     {6,  "LElbow"},
            # //     {7,  "LWrist"},
            # //     {8,  "RHip"},
            # //     {9,  "RKnee"},
            # //     {10, "RAnkle"},
            # //     {11, "LHip"},
            # //     {12, "LKnee"},
            # //     {13, "LAnkle"},
            # //     {14, "REye"},
            # //     {15, "LEye"},
            # //     {16, "REar"},
            # //     {17, "LEar"},
            # //     {18, "Background"},
            # // }

            # An array pose_keypoints_2d containing the body part locations and detection confidence formatted as x1,y1,c1,x2,y2,c2,....
            # The coordinates x and y can be normalized to the range [0,1], [-1,1], [0, source size], [0, output size], etc., depending on the flag keypoint_scale,
            # while c is the confidence score in the range [0,1].

            # loop through all the people detected
            for j in range(num_people):

                pose_keypoints_2d_info = openpose_info['people'][j]['pose_keypoints_2d']
                # if left eye or right eye or nose inside bounding box, say this is speaker's skeleton
                if isinsidebox(bounding_box, (pose_keypoints_2d_info[0*3+0],pose_keypoints_2d_info[0*3+1])) or \
                    isinsidebox(bounding_box, (pose_keypoints_2d_info[14*3+0],pose_keypoints_2d_info[14*3+1])) or \
                    isinsidebox(bounding_box, (pose_keypoints_2d_info[15*3+0],pose_keypoints_2d_info[15*3+1])):

                    # check if either left or right hip of speaker is visible - check for zero confidence
                    if pose_keypoints_2d_info[8*3+2] != 0 or pose_keypoints_2d_info[11*3+2] != 0:
                        speaker_upperbody_found[i] = True

                        # TODO - do I need to save speaker skeleton information here?
                    break
                else:
                    continue

    return speaker_upperbody_found




def preprocess_video(frames_path):

    ''' Find length of video
        Count the number of frames - 30 fps sampling
        Ignore the first 390 frames (13 sec) and last 180 frames (6 sec) to ignore the TED talk intro and exit
        Also clip some frames from the end to round to lower multiple of 30'''

    speaker_enc_fname = frames_path.split()[-1] + '.npy'
    speaker_face_encoding = np.load(os.path.join(speaker_enc_dir, speaker_enc_fname)
    frames_path_list = sorted(glob.glob(frames_path + '/*'))[390:-180]
    quo, rem = divmod(len(frames_path_list),30)

    frames_path_list = frames_path_list[:-rem]
    num_frames = len(frames_path_list)

    print(num_frames)



    ''' Consider one frame for each second in the video and run constraints on all the selected frames
        CONSTRAINT 1: speaker face detected
        CONSTRAINT 2: speaker upper body visible

        N.B. Because face recognition happens in batches and we can parallelize openpose on GPUs, I think
        decoupling both the steps seems a good idea'''

    random.seed(123)
    # choose randomly among the 30 frames in a second
    candidate_frame_indices = [random.randint(0,29)+i for i in range(0, num_frames, 30)]

    candidate_frames = [frames_path_list[i] for i in candidate_frame_indices]

    print(len(candidate_frames))

    # CONSTRAINT 1: speaker face detected

    speaker_face_bb = detect_speaker_face(
        candidate_frames, speaker_face_encoding=speaker_face_encoding)



    # CONSTRAINT 2: speaker upper body visible
    # TODO - how do I pass a specific set of images to the openpose.bin?
    # What should the json output be? normalized or not? how to put in missing joints?
    '''./build/examples/openpose/openpose.bin --image_dir examples/media/ --write_images <images-directory> --write_json <json-directory> --display 0 --render_pose 1'''

    # process the json files
    jsonpathlist = sorted(glob.glob(json_dir + '/'))
    speaker_upperbody_found = process_json(jsonpathlist, bounding_boxes)

    # find intersection of the lists speaker_face_found and speaker_upperbody_found
    candidate_frames_intersection = [x and y for x,y in zip(speaker_face_found, speaker_upperbody_found)]


    ''' Find sequences where both constraints holds
        SEQUENCE_LENGTH set to 10 seconds '''
    sequence_start_frames = []
    for i in range(len(candidate_frames_intersection)-SEQUENCE_LENGTH):
        if False not in candidate_frames_intersection[i:i+SEQUENCE_LENGTH]:
            sequence_start_frames.append(candidate_frames[i])




if __name__ == "__main__":

    print(str(datetime.now()))
    # create openpose json file output directory
    if not os.path.exists(json_root):
        os.makedirs(json_root)

    # create list of video paths
    frames_dir_path_list = sorted(glob.glob(frames_root + '/*'))

    # for every video
    for i,frame_dir_path in enumerate(frames_dir_path_list):
        print(i, frame_dir_path)
        preprocess_video(frame_dir_path)

        if i == 0:
            break

    print(str(datetime.now()))
