import argparse
import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# tensorboard
from logger import Logger


def get_parser():

    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network and SoundNet to learn pose and audio sync')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/pose-audio',
        help='the work folder for storing results')
    parser.add_argument(
        '--config',
        default='./config/pose-audio/train.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if true, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    # parser.add_argument(
    #     '--show-topk',
    #     type=int,
    #     default=[1, 5],
    #     nargs='+',
    #     help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=128,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--val-feeder-args',
        default=dict(),
        help='the arguments of data loader for validation')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--val-batch-size', type=int, default=256, help='val batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')

    return parser

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Processor():
    """
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.load_data()
        self.load_model()
        self.load_model_soundnet()
        self.load_optimizer()
        self.load_optimizer_soundnet()

        self.load_loss()

        #define TensorBoard logger
        self.logger = Logger("/nfs1/code/aniruddha/pose/tensorboard/")

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker)
        self.data_loader['val'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.val_feeder_args),
            batch_size=self.arg.val_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker)
        # self.data_loader['test'] = torch.utils.data.DataLoader(
        #     dataset=Feeder(**self.arg.test_feeder_args),
        #     batch_size=self.arg.test_batch_size,
        #     shuffle=False,
        #     num_workers=self.arg.num_worker)

    def load_model(self):
        output_device = self.arg.device[
            0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        #self.loss = nn.CosineEmbeddingLoss().cuda(output_device)

        if self.arg.weights:
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    self.print_log('Can Not Remove Weights: {}.'.format(w))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        # Freeze gt-scn weights
        for param in self.model.parameters():
            param.requires_grad = False

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def load_model_soundnet(self):
        output_device_soundnet = self.arg.device_soundnet[
            0] if type(self.arg.device_soundnet) is list else self.arg.device_soundnet
        self.output_device_soundnet = output_device_soundnet
        Model = import_class(self.arg.model_soundnet)
        self.model_soundnet = Model().cuda(output_device_soundnet)


        # Load pretrained soundnet weights
        self.model_soundnet.load_weights()


        # Freeze soundnet weights
        for param in self.model_soundnet.parameters():
            param.requires_grad = False

        if type(self.arg.device_soundnet) is list:
            if len(self.arg.device_soundnet) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device_soundnet,
                    output_device=output_device_soundnet)

    # def load_optimizer(self):
    #     if self.arg.optimizer == 'SGD':
    #         self.optimizer = optim.SGD(
    #             self.model.parameters(),
    #             lr=self.arg.base_lr,
    #             momentum=0.9,
    #             nesterov=self.arg.nesterov,
    #             weight_decay=self.arg.weight_decay)
    #         optimor = optim.SGD
    #     elif self.arg.optimizer == 'Adam':
    #         self.optimizer = optim.Adam(
    #             self.model.parameters(),
    #             lr=self.arg.base_lr,
    #             weight_decay=self.arg.weight_decay)
    #     else:
    #         raise ValueError()

    # def load_optimizer_soundnet(self):
    #     if self.arg.optimizer == 'SGD':
    #         self.optimizer = optim.SGD(
    #             self.model.parameters(),
    #             lr=self.arg.base_lr,
    #             momentum=0.9,
    #             nesterov=self.arg.nesterov,
    #             weight_decay=self.arg.weight_decay)
    #         optimor = optim.SGD
    #     elif self.arg.optimizer == 'Adam':
    #         self.optimizer = optim.Adam(
    #             self.model.parameters(),
    #             lr=self.arg.base_lr,
    #             weight_decay=self.arg.weight_decay)
    #     else:
    #         raise ValueError()

    def add_loss(self):                             # Aniruddha
        output_device = self.arg.device[
            0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        self.loss = nn.CosineEmbeddingLoss(margin=0.5).cuda(output_device)

    def add_finetuning_layers(self):                # Aniruddha

        self.finetune_model = nn.Sequential(

                nn.Conv1d(1024, 512, 1, stride=1)
                nn.ReLU(True)

                nn.Conv1d(512, 256, 8)
                nn.ReLU(True)
            )

        self.finetune_model.cuda(self.output_device)

        # self.finetune_conv1 = nn.Conv1d(1024, 512, 1, stride=1)
        # self.finetune_relu1 = nn.ReLU(True)

        # self.finetune_conv2 = nn.Conv1d(512, 256, 8)
        # self.finetune_relu2 = nn.ReLU(True)

        # self.finetune_conv1.cuda(self.output_device)
        # self.finetune_relu1.cuda(self.output_device)
        # self.finetune_conv2.cuda(self.output_device)
        # self.finetune_relu2.cuda(self.output_device)

    def add_optimizer_finetuning(self):               # Aniruddha
        self.finetune_optimizer = optim.SGD(self.finetune_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.001)

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            lr = self.arg.base_lr * (
                0.1**np.sum(epoch >= np.array(self.arg.finetune_step)))
            for param_group in self.finetune_optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):

        # SoundNet and ST-GCN used as fixed feature extractors - weights freezed, finetune model trains
        self.model.eval()
        self.model_soundnet.eval()
        self.finetune_model.train()

        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        lr = self.adjust_learning_rate(epoch)
        loss_value = []

        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        for batch_idx, (posedata, audiodata, label) in enumerate(loader):

            # if using Cosine Embedding Loss change 0 to -1 in label
            label[label == 0] = -1

            # get data
            posedata = Variable(
                posedata.float().cuda(self.output_device), requires_grad=False)
            audiodata = Variable(
                posedata.float().cuda(self.output_device), requires_grad=False)
            label = Variable(
                label.long().cuda(self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()



            # forward
            output_pose = self.model(posedata)
            output_audio = self.model_soundnet(audiodata)

            output_audio = self.finetune_model(output_audio)

            loss = self.loss(output_pose, output_audio, label)

            # backward
            self.finetune_optimizer.zero_grad()
            loss.backward()
            self.finetune_optimizer.step()
            loss_value.append(loss.data[0])
            timer['model'] += self.split_time()

            # statistics
            if batch_idx % self.arg.log_interval == 0:
                self.print_log(
                    '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
                        batch_idx, len(loader), loss.data[0], lr))

                #============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {
                    'loss': loss_value,

                    'finetune_optimizer_LR': self.finetune_optimizer.param_groups[0]['lr']
                }

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, counter)

                # (2) Log values and gradients of the parameters (histogram)
                for tag, value in self.finetune_model.named_parameters():
                    tag = tag.replace('.', '/')
                    self.logger.histo_summary(tag, to_np(value), counter)
                    self.logger.histo_summary(tag+'/grad', to_np(value.grad), counter)

            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
                **proportion))

        if save_model:
            model_path = '{}/epoch{}_model.pt'.format(self.arg.work_dir,
                                                      epoch + 1)
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1],
                                    v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, model_path)

    def eval(self, epoch, save_score=False, loader_name=['val']):
        # SoundNet and ST-GCN used as fixed feature extractors - weights freezed, finetune model evaluates
        self.model.eval()
        self.model_soundnet.eval()
        self.finetune_model.eval()

        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            for batch_idx, (posedata, audiodata, label) in enumerate(self.data_loader[ln]):

                # if using Cosine Embedding Loss change 0 to -1 in label
                label[label == 0] = -1

                posedata = Variable(
                    posedata.float().cuda(self.output_device),
                    requires_grad=False,
                    volatile=True)

                audiodata = Variable(
                    audiodata.float().cuda(self.output_device),
                    requires_grad=False,
                    volatile=True)

                label = Variable(
                    label.long().cuda(self.output_device),
                    requires_grad=False,
                    volatile=True)
                # forward
                output_pose = self.model(posedata)
                output_audio = self.model_soundnet(audiodata)

                output_audio = self.finetune_model(output_audio)

                loss = self.loss(output_pose, output_audio, label, reduce=False)        # don't reduce during evaluation



                score_frag.append(loss.data.cpu().numpy())
                loss_value.append(torch.mean(loss.data[0]))
            #score = np.concatenate(score_frag)
            # score_dict = dict(
            #     zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            # for k in self.arg.show_topk:
            #     self.print_log('\tTop{}: {:.2f}%'.format(
            #         k, 100 * self.data_loader[ln].dataset.top_k(score, k)))


            self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.accuracy(score, margin=0.5)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

    def start_finetuning(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = ((epoch + 1) % self.arg.finetune_save_interval == 0) or (
                    epoch + 1 == self.arg.finetune_num_epoch)
                eval_model = ((epoch + 1) % self.arg.finetune_eval_interval == 0) or (
                    epoch + 1 == self.arg.finetune_num_epoch)

                self.train(epoch, save_model=save_model)

                if eval_model:
                    self.eval(
                        epoch,
                        save_score=self.arg.save_score,
                        loader_name=['val'])
                else:
                    pass

        elif self.arg.phase == 'val':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(
                epoch=0, save_score=self.arg.save_score, loader_name=['val'])
            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    processor = Processor(arg)
    processor.start()