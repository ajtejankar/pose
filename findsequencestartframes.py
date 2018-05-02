import json
import glob
import subprocess
import os
import time

# test comment
sequence_start_frame = []
for path in sorted(glob.glob('/nfs1/code/aniruddha/sample_vids/speakerupperbodyframesjson/*.json')):

    with open(path,'r') as inp:

        print("Working on " + path)
        # load json into dictionary
        info = json.load(inp)
        #print(len(info))

        info = [s.encode('utf-8') for s in info]

        # for all speaker_upper_body detected frames, extract int frame numbers:
        framenumber_list = []
        for i,text in enumerate(info):
            framenumber = text.split('/')[-1]
            framenumber = framenumber.replace('.jpg','')
            framenumber = int(framenumber)
            framenumber_list.append(framenumber)


        sequence_length = 30*10   # we are considering 5 second sequences

        i = 0
        while i < len(framenumber_list):
            if (i+sequence_length) >= len(framenumber_list):
                break
            if framenumber_list[i+sequence_length] <= (framenumber_list[i] + 1.05*sequence_length): # 10% slack allowed
                sequence_start_frame.append(info[i])
                i = i + sequence_length + 1
            else:
                i = i + 1


print(len(sequence_start_frame))

vids_root = '/nfs1/code/aniruddha/sample_vids'
audios_root = '/nfs1/code/aniruddha/sample_vids_audios'
toy_dataset_path = os.path.join(vids_root,'toydataset')

for i,start_frame in enumerate(sequence_start_frame):
    print(start_frame)

    sequence_start_frame = start_frame.split('/')[-1]
    sequence_start_frame_int = int(sequence_start_frame.replace('.jpg',''))

    quo, rem = divmod(sequence_start_frame_int, 30)

    sequence_start_frame_int = sequence_start_frame_int + (30 -rem)

    print(sequence_start_frame_int)

    seconds = sequence_start_frame_int/30

    #print(seconds)

    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    #print "%d:%02d:%02d" % (h, m, s)

    # clip audio and create video (without audio) from frames
    start_time = str(h).zfill(2) + ":" + str(m).zfill(2) + ":" +str(s).zfill(2) + ".0"
    duration = "00:00:10.0"
    source_audio_path = os.path.join(audios_root,start_frame.split('/')[0]) + '.mp4'
    print(source_audio_path)
    print(start_time)

    clip_audio_path = os.path.join(toy_dataset_path,'audios',start_frame.split('/')[0]+str(i)) + '.mp4'

    bash_command_audio = "ffmpeg -i " + source_audio_path + " -ss " + start_time + " -t " + duration + " -c copy -y " + clip_audio_path
    print(bash_command_audio)

    subprocess.call(bash_command_audio.split(' '))


    pose_only_frame_root = os.path.join(vids_root,start_frame.split('/')[0] + '_pose_only_frames')
    pose_overlay_frame_root = os.path.join(vids_root,start_frame.split('/')[0] + '_pose_overlay_frames')

    clip_video_path = os.path.join(toy_dataset_path,'videos',start_frame.split('/')[0]+str(i)) + '.mp4'

    bash_command_video = "ffmpeg -framerate 30 -start_number " + str(sequence_start_frame_int) + " -i " + pose_only_frame_root + "/%6d.jpg -vframes 300 -pix_fmt yuv420p -codec:v libx264 -y " + clip_video_path
    print(bash_command_video)

    subprocess.call(bash_command_video.split(' '))

    clip_video_path2 = os.path.join(toy_dataset_path,'videos_overlay',start_frame.split('/')[0]+str(i)) + '.mp4'

    bash_command_video2 = "ffmpeg -framerate 30 -start_number " + str(sequence_start_frame_int) + " -i " + pose_overlay_frame_root + "/%6d.jpg -vframes 300 -pix_fmt yuv420p -codec:v libx264 -y " + clip_video_path2
    print(bash_command_video2)

    subprocess.call(bash_command_video2.split(' '))













