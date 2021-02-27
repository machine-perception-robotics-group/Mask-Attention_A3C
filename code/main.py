from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from environment import atari_env
from utils import read_config
from train import train
from test import test
from shared_optim import SharedRMSprop, SharedAdam
#from gym.configuration import undo_logger_setup
import time
import os
import csv

#undo_logger_setup()
parser = argparse.ArgumentParser(description='attention_a3c')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')
parser.add_argument(
    '--max_global_step',
    type=int,
    default=100000000,
    help='global step')
parser.add_argument(
    '--convlstm',
    action='store_true',
    help='Using ConvLSTM')
parser.add_argument(
    '--mask_double',
    action='store_true',
    help='Using Mask A3C double')
parser.add_argument(
    '--mask_single_p',
    action='store_true',
    help='Using Mask A3C single policy')
parser.add_argument(
    '--mask_single_v',
    action='store_true',
    help='Using Mask A3C single value')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')
parser.add_argument(
    '--seed',
    type=int,
    default=23,
    metavar='S',
    help='random seed (default: 23)')
parser.add_argument(
    '--workers',
    type=int,
    default=1,
    metavar='W',
    help='how many training processes to use (default: 1)')
parser.add_argument(
    '--num-steps',
    type=int,
    default=20,
    metavar='NS',
    help='number of forward steps in A3C (default: 20)')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=10000,
    metavar='M',
    help='maximum length of an episode (default: 10000)')
parser.add_argument(
    '--env',
    default='PongNoFrameskip-v4',
    metavar='ENV',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--env-config',
    default='config.json',
    metavar='EC',
    help='environment to crop and resize info (default: config.json)')
parser.add_argument(
    '--shared-optimizer',
    default=True,
    metavar='SO',
    help='use an optimizer without shared statistics.')
parser.add_argument(
    '--load', default=False, metavar='L', help='load a trained model')
parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run high score matched or bested')
parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='shares optimizer choice of Adam or RMSprop')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--save-model-dir',
    default='trained_models/',
    metavar='SMD',
    help='folder to save trained models')
parser.add_argument(
    '--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument(
    '--gpu-ids',
    type=int,
    default=-1,
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--amsgrad',
    default=True,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')
parser.add_argument(
    '--skip-rate',
    type=int,
    default=4,
    metavar='SR',
    help='frame skip rate (default: 4)')

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    if args.mask_double: model_name = 'Mask-A3C-double'
    elif args.mask_single_p: model_name = 'Mask-A3C-single-policy'
    elif args.mask_single_v: model_name = 'Mask-A3C-single-value'
    else: model_name = 'A3C'
    if args.convlstm: model_name += '+ConvLSTM'
    print('\033[31m' + model_name + '\033[0m')

    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]
    env = atari_env(args.env, env_conf, args)

    save_log_path = os.path.join(args.log_dir, args.env)
    if not os.path.exists(save_log_path):
        os.makedirs(save_log_path)
    test_log_path = os.path.join(save_log_path, "{}_{}_test.csv".format(args.env, model_name))
    if not os.path.exists(test_log_path):
        with open(test_log_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(["global_step", "time", "episode length", "reward_sum", "reward_mean"])
    train_log_path = os.path.join(save_log_path, "{}_{}.csv".format(args.env, model_name))
    if not os.path.exists(train_log_path):
        with open(train_log_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(["time", "global step", "score"])

    if args.mask_double:
        from models.model_mask_double import A3C
    elif args.mask_single_p:
        from models.model_mask_single_policy import A3C
    elif args.mask_single_v:
        from models.model_mask_single_value import A3C
    else:
        from models.model import A3C
    shared_model = A3C(args, env.observation_space.shape[0], env.action_space)
    
    global_step = mp.Value('i', 0)
    start_time = mp.Value('f', time.time())
    
    if args.load:
        saved_state = torch.load(
            '{0}{1}.dat'.format(args.load_model_dir, args.env),
            map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()

    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None
    

    processes = []

    #p = mp.Process(target=test, args=(args, shared_model, env_conf, global_step, start_time))
    #p.start()
    #processes.append(p)
    time.sleep(0.1)
    for rank in range(0, args.workers):
        p = mp.Process(
            target=train, args=(rank, args, shared_model, optimizer, env_conf, global_step, start_time))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    for p in processes:
        time.sleep(0.1)
        p.join()

    print('global step: {}'.format(global_step.value))
    print('finish!')
