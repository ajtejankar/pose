import os
import glob
import random

FORMAT = '%-20s: %s'
FRAMES_ROOT = '/nfs2/datasets/TED/frames'
SPEAKER_ROOT = '/nfs1/code/ajinkya/pose/speakers'
CAND_FRAME_LINKS_ROOT = '/nfs1/code/ajinkya/pose/cand_frame_links'

os.makedirs(CAND_FRAME_LINKS_ROOT, exist_ok=True)

for speaker_enc_path in glob.glob(SPEAKER_ROOT + '/*.npy')[:1]:
    speaker_name = speaker_enc_path.split('/')[-1].replace('.npy', '')
    print(FORMAT % ('speaker_name', speaker_name))

    speaker_frames_dir_path = os.path.join(FRAMES_ROOT, speaker_name)
    print(FORMAT % ('speaker_dir', speaker_frames_dir_path)) 

    frame_paths = sorted(glob.glob(speaker_frames_dir_path +  '/*'))[390:-180]
    quo, rem = divmod(len(frame_paths), 30)

    if rem != 0:
        frame_paths = frame_paths[:-rem]

    num_frames = len(frame_paths)
    print(FORMAT % ('num_frames', num_frames)) 

    random.seed(123)
    candidate_frame_indices = [random.randint(0,29)+i for i in range(0, num_frames, 30)]
    candidate_frames = [frame_paths[i] for i in candidate_frame_indices]
    print(FORMAT % ('candidate_frames', len(candidate_frames)))

    for cand_frame in candidate_frames:
        speaker_name = cand_frame.split('/')[-2]
        cand_link_dir = os.path.join(CAND_FRAME_LINKS_ROOT, speaker_name)
        os.makedirs(cand_link_dir, exist_ok=True)
        cand_link_frame  = os.path.join(cand_link_dir, cand_frame.split('/')[-1])
        os.link(cand_frame, cand_link_frame)
        print(FORMAT % ('link', '%s --> %s' %(cand_frame, cand_link_frame)))

