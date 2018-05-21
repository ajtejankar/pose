#!/usr/bin/env python

import glob

num_list = []

for path in sorted(glob.glob('*.jpg')):
    num = int(path.replace('.jpg', '')) // 30
    num_list.append(num)

if len(set(num_list)) == len(num_list):
    print('right')
else:
    print('wrong')
