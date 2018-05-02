import sys
import dlib
import os
import subprocess as sp
import argparse as ap

FORMAT = '%-20s: %s'

parser = ap.ArgumentParser('Generate parallelized bash scripts')

parser.add_argument('script_name',
        help='the script for which to generate bash file')

parser.add_argument('vid_paths',
        help='a file containig list of paths to speaker frames directory')

parser.add_argument('devices',
        metavar='D',
        type=int,
        nargs='+',
        help='a list of available GPU device ids')

args = parser.parse_args()

print(FORMAT % ('vid_paths', args.vid_paths))

paths_file = args.vid_paths
with open(paths_file) as f:
    paths = f.readlines()
paths = [path.strip() for path in paths]

devices = args.devices
batch_size = len(paths) // len(devices)
start = 0
script_dir = os.path.dirname(os.path.realpath(__file__))
script_py = script_dir + '/' + args.script_name

print(FORMAT % ('paths', str(len(paths))))
print(FORMAT % ('devices', len(devices)))

with open('run_detector.bash', 'w') as f:
    f.write('')

for device in devices:
    if device == devices[-1]:
        batch_size += len(paths) % len(devices)

    print(FORMAT % ('device', '%d' % (device)))
    print(FORMAT % ('batch', '%d:%d' % (start, start+batch_size)))

    batch_paths = paths[start:start+batch_size]
    batch_paths = ' '.join(batch_paths)
    cmd = 'CUDA_VISIBLE_DEVICES=%d python %s %s &' % \
            (device, script_py, batch_paths)

    print(''.join(['='] * 50))
    print(cmd)

    with open('run_detector.bash', 'a') as f:
        f.write('\n' + cmd + '\n')

    print(''.join(['='] * 50))

    start += batch_size


