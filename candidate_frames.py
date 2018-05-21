import os
import glob
import random
from multiprocessing import Pool
import datetime
import sys

NUM_WORKERS = 8
FORMAT = '%-20s: %s'
FRAMES_ROOT = '/nfs2/datasets/TED/frames'
SPEAKER_ROOT = '/nfs1/shared/for_aniruddha/pose/speakers'
CAND_FRAME_LINKS_ROOT = '/nfs1/shared/for_aniruddha/pose/cand_frame_links'

os.makedirs(CAND_FRAME_LINKS_ROOT, exist_ok=True)

def handle_path(speaker_enc_path):
    speaker_name = speaker_enc_path.split('/')[-1].replace('.npy', '')
    print(FORMAT % ('speaker_name', speaker_name))

    speaker_frames_dir_path = os.path.join(FRAMES_ROOT, speaker_name)
    print(FORMAT % ('speaker_dir', speaker_frames_dir_path)) 

    frame_paths = sorted(glob.glob(speaker_frames_dir_path +  '/*'))[390:-180]
    tail_clip = len(frame_paths) % 30 

    if tail_clip != 0:
        frame_paths = frame_paths[:-tail_clip]

    num_frames = len(frame_paths)
    print(FORMAT % ('first', frame_paths[0])) 
    print(FORMAT % ('last', frame_paths[-1])) 
    print(FORMAT % ('num_frames', num_frames)) 
    print(FORMAT % ('num_secs', num_frames//30)) 

    candidate_frames = [frame_paths[i*30] for i in range(num_frames//30)]
    candidate_frames[-1] = frame_paths[-1]

    print(FORMAT % ('cand_first', candidate_frames[0])) 
    print(FORMAT % ('cand_last', candidate_frames[-1])) 
    print(FORMAT % ('candidate_frames', len(candidate_frames)))

    if len(candidate_frames) == 0:
        return

    speaker_name = candidate_frames[0].split('/')[-2]
    cand_link_dir = os.path.join(CAND_FRAME_LINKS_ROOT, speaker_name)

    if os.path.exists(cand_link_dir):
        print(FORMAT % ('skip_cand_dir', cand_link_dir))
        return

    os.makedirs(cand_link_dir, exist_ok=True)
    x = []

    for cand_frame in candidate_frames:
        cand_link_frame  = os.path.join(cand_link_dir, cand_frame.split('/')[-1])
        os.symlink(cand_frame, cand_link_frame)
        x.append(int(cand_frame.split('/')[-1].replace('.jpg', ''))//30)
        # print(FORMAT % ('link', '%s --> %s' %(cand_frame, cand_link_frame)))

    print(FORMAT % ('check', len(x) == len(set(x))))


# Logger to give output to console as well as a file simultaneously.
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        dt = str(datetime.datetime.now().date())
        tm = str(datetime.datetime.now().time())
        fname = 'candidate_' + dt + '_' + tm.replace(':', '.') + '.log'
        self.log = open(fname, "a")  # specify file name here

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


sys.stdout = Logger()
sys.stderr = sys.stdout
pool = Pool(NUM_WORKERS)
pool.map(handle_path, glob.glob(SPEAKER_ROOT + '/*.npy'))
pool.close()
pool.join()
