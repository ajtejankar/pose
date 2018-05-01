import sys
import dlib
import os
import subprocess as sp

FORMAT = '%-20s: %s'

if len(sys.argv) < 2:
    print('Incorrect usage. Needs a file containing paths to speaker frames directories')
    sys.exit(1)

print(FORMAT % ('vid_paths', sys.argv[1]))

paths_file = sys.argv[1]
with open(paths_file) as f:
    paths = f.readlines()
paths = [path.strip() for path in paths]

device_count = dlib.cuda.get_num_devices()
batch_size = len(paths) // device_count
start = 0
script_dir = os.path.dirname(os.path.realpath(__file__))
detect_speaker_py = script_dir + '/' + 'detect_speaker.py'

print(FORMAT % ('paths', str(len(paths))))
print(FORMAT % ('devices', device_count))

for device in range(device_count):
    if device == device_count-1:
        batch_size += len(paths) % device_count

    dlib.cuda.set_device(device)
    print(FORMAT % ('device', '%d' % (dlib.cuda.get_device())))
    print(FORMAT % ('batch', '%d:%d' % (start, start+batch_size)))

    batch_paths = paths[start:start+batch_size]
    batch_paths = ' '.join(batch_paths)
    cmd = 'CUDA_VISIBLE_DEVICES=%d python %s %d %s' % \
            (device, detect_speaker_py, device, batch_paths)
    print(FORMAT % ('cmd', cmd))

    sp.Popen(cmd.split(' '), shell=True)

    start += batch_size

