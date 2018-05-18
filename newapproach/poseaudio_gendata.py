import numpy as np
import argparse
import os
import sys
import pickle
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from feeder_poseaudio import Feeder_PoseAudio
from numpy.lib.format import open_memmap
import pickle

toolbar_width = 30


def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


def gendata(
        data_path,
        label_path,
        data_out_path,
        label_out_path,
        max_frame=150,
        audio_length=(22050*5)):    # take max 5 second sequence - sampled at 30 fps

    feeder = Feeder_PoseAudio(
        data_path=data_path,
        label_path=label_path,
        window_size=max_frame)

    sample_name = feeder.sample_name
    sample_label = []

    fp = open_memmap(
        data_out_path,
        dtype='float32',
        mode='w+',
        shape=(len(sample_name), 36, max_frame))

    for i, s in enumerate(sample_name):
        data, _, label = feeder[i]
        print_toolbar(i * 1.0 / len(sample_name),
                      '({:>5}/{:<5}) Processing posedata: '.format(
                          i + 1, len(sample_name)))
        fp[i, :, :] = data
        # fp_audio[i, :] = audio
        sample_label.append(label)


    fp_audio = open_memmap(
            data_audio_out_path,
            dtype='float32',
            mode='w+',
            shape=(len(sample_name), audio_length))

    for i, s in enumerate(sample_name):
        _, audio, label = feeder[i]
        assert(label == sample_label[i])
        print_toolbar(i * 1.0 / len(sample_name),
                      '({:>5}/{:<5}) Processing audiodata: '.format(
                          i + 1, len(sample_name)))
        fp_audio[i, :] = audio

    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PoseAudio Data Converter.')
    parser.add_argument(
        '--data_path', default='/nfs1/shared/for_aniruddha/pose/dataset')
    parser.add_argument(
        '--out_folder', default='../data/poseaudio')
    arg = parser.parse_args()

    part = ['val', 'test']
    for p in part:
        data_path = '{}/{}'.format(arg.data_path, p)
        label_path = '{}/{}.json'.format(arg.data_path, p)
        data_out_path = '{}/{}_data.npy'.format(arg.out_folder, p)
        label_out_path = '{}/{}_label.pkl'.format(arg.out_folder, p)
        data_audio_out_path = '{}/{}_audio_data.npy'.format(arg.out_folder, p)

        gendata(data_path, label_path, data_out_path, label_out_path)