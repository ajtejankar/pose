import os
import glob
import time
import random
import subprocess

video_poseonly_root = '/nfs1/code/aniruddha/sample_vids/toydataset/videos'
video_poseoverlay_root = '/nfs1/code/aniruddha/sample_vids/toydataset/videos_overlay'
audios_root = '/nfs1/code/aniruddha/sample_vids/toydataset/audios'
store_root_poseonly = '/nfs1/code/aniruddha/sample_vids/toydataset/pairs_poseonly'
store_root_poseoverlay = '/nfs1/code/aniruddha/sample_vids/toydataset/pairs_poseoverlay'


poseonlypathlist = sorted(glob.glob(video_poseonly_root + '/*'))
poseoverlaypathlist = sorted(glob.glob(video_poseoverlay_root + '/*'))
audiospathlist = sorted(glob.glob(audios_root + '/*'))

for path in poseonlypathlist:
    print path

for i,poseonlypath in enumerate(poseonlypathlist):

    if i == 12:
        # correct mapping
        audiopath = audiospathlist[i]
        poseoverlaypath = poseoverlaypathlist[i]

        store_path1 = store_root_poseonly + "/" + str(i) + "_correct.mp4"
        store_path2 = store_root_poseoverlay + "/" + str(i) + "_correct.mp4"

        bash_command1 = "ffmpeg -loglevel error -i " + poseonlypath + " -i " + audiopath + " -c:v copy -r 30 -c:a copy -strict experimental -y " + store_path1
        bash_command2 = "ffmpeg -loglevel error -i " + poseoverlaypath + " -i " + audiopath + " -c:v copy -r 30 -c:a copy -strict experimental -y " + store_path2

        # print('-'*80)
        # print(poseonlypath)
        # print(poseoverlaypath)
        # print("correct" + audiopath)

        print(bash_command1)
        print(bash_command2)

        subprocess.call(bash_command1.split(' '))
        subprocess.call(bash_command2.split(' '))


        # incorrect mapping
        randomindex = i
        while randomindex == i:
            randomindex = random.randint(0,27)

        audiopath = audiospathlist[randomindex]
        poseoverlaypath = poseoverlaypathlist[i]

        store_path1 = store_root_poseonly + "/" + str(i) + "_incorrect.mp4"
        store_path2 = store_root_poseoverlay + "/" + str(i) + "_incorrect.mp4"

        bash_command1 = "ffmpeg -loglevel error -i " + poseonlypath + " -i " + audiopath + " -c:v copy -r 30 -c:a copy -strict experimental -y " + store_path1
        bash_command2 = "ffmpeg -loglevel error -i " + poseoverlaypath + " -i " + audiopath + " -c:v copy -r 30 -c:a copy -strict experimental -y " + store_path2

        #print("incorrect" + audiopath)

        print(bash_command1)
        print(bash_command2)

        subprocess.call(bash_command1.split(' '))
        subprocess.call(bash_command2.split(' '))



