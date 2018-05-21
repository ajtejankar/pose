import numpy as np
import argparse
import os
import sys
import pickle
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from st_gcn.feeder.feeder_kinetics import Feeder_kinetics
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
        num_person_in=1,  #observe the first 1 person - N.B. We just have one skeleton per frame but ST-GCN is trained on 2 skeletons
        num_person_out=1,  #then choose 1 person with the highest score
        max_frame=150,
        audio_length=(44100*5)):    # take max 5 second sequence - sampled at 30 fps

    feeder = Feeder_kinetics(
        data_path=data_path,
        label_path=label_path,
        num_person_in=num_person_in,
        num_person_out=num_person_out,
        window_size=max_frame)

    sample_name = feeder.sample_name
    sample_label = []

    fp_audio = open_memmap(
            data_out_path,
            dtype='float32',
            mode='w+',
            shape=(len(sample_name), audio_length))

    fp = open_memmap(
        data_out_path,
        dtype='float32',
        mode='w+',
        shape=(len(sample_name), 3, max_frame, 18, num_person_out))

    for i, s in enumerate(sample_name):
        data, audio, label = feeder[i]
        print_toolbar(i * 1.0 / len(sample_name),
                      '({:>5}/{:<5}) Processing data: '.format(
                          i + 1, len(sample_name)))
        fp[i, :, 0:data.shape[1], :, :] = data
        fp_audio[i, :] = audio
        sample_label.append(label)

    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Kinetics-skeleton Data Converter.')
    parser.add_argument(
        '--data_path', default='data/Kinetics/kinetics-skeleton')
    parser.add_argument(
        '--out_folder', default='data/Kinetics/kinetics-skeleton')
    arg = parser.parse_args()

    part = ['train', 'val', 'test']
    for p in part:
        data_path = '{}/kinetics_{}'.format(arg.data_path, p)
        label_path = '{}/kinetics_{}_label.json'.format(arg.data_path, p)
        data_out_path = '{}/{}_data.npy'.format(arg.out_folder, p)
        label_out_path = '{}/{}_label.pkl'.format(arg.out_folder, p)
        data_audio_out_path = '{}/{}_audio_data.npy'.format(arg.out_folder, p)

        gendata(data_path, label_path, data_out_path, label_out_path)