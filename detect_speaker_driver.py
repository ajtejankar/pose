import sys
import dlib
import os

FORMAT = '%-20s: %s'

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
print(FORMAT % ('devices', str(len(device_count))))

for device in range(device_count):
    if device == device_count-1:
        batch_size += len(paths) % device_count

    dlib.cuda.set_device(device)
    print(FORMAT % ('device', '%d' % (dlib.cuda.get_device())))
    print(FORMAT % ('batch', '%d:%d' % (start, start+batch_size)))

    batch_paths = paths[start:start+batch_size]

    start += batch_size

