from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
from environment import atari_env
from utils import read_config
from player_util import Agent
import time
import cv2
import numpy as np
from os import path
from statistics import mean, variance, stdev
from tqdm import tqdm
from setproctitle import setproctitle as ptitle


def min_max(x, mins, maxs, axis=None):
    result = (x - mins)/(maxs - mins)
    return result

if __name__ == '__main__':
    ptitle('Mask A3C Eval')
    parser = argparse.ArgumentParser(description='Mask-A3C_EVAL')
    parser.add_argument(
        '--convlstm',
        action='store_true',
        help='Using convLSTM')
    parser.add_argument(
        '--mask_double',
        action='store_true',
        help='Using mask a3c double')
    parser.add_argument(
        '--mask_single_p',
        action='store_true',
        help='Using mask a3c single policy')
    parser.add_argument(
        '--mask_single_v',
        action='store_true',
        help='Using mask a3c single value')
    parser.add_argument(
        '--image',
        action='store_true',
        help='Using save image')
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
        '--num-episodes',
        type=int,
        default=100,
        metavar='NE',
        help='how many episodes in evaluation (default: 100)')
    parser.add_argument(
        '--load-model-dir',
        default='trained_models/',
        metavar='LMD',
        help='folder to load trained models from')
    parser.add_argument(
        '--load-model',
        default='BreakoutNoFrameskip-v4',
        metavar='LMN',
        help='name to load trained models from')
    parser.add_argument(
        '--log-dir', default='logs/', metavar='LG', help='folder to save logs')
    parser.add_argument(
        '--render',
        action='store_true',
        help='Watch game as it being played')
    parser.add_argument(
        '--max-episode-length',
        type=int,
        default=10000,
        metavar='M',
        help='maximum length of an episode (default: 100000)')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        default=-1,
        help='GPU to use [-1 CPU only] (default: -1)')
    parser.add_argument(
        '--skip-rate',
        type=int,
        default=4,
        metavar='SR',
        help='frame skip rate (default: 4)')
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        metavar='S',
        help='random seed (default: 1)')

    args = parser.parse_args()
    print(args.load_model)
    print(args)

    if not path.exists(args.load_model):
        os.mkdir(args.load_model)

    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]

    gpu_id = args.gpu_ids

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)

    saved_state = torch.load(
        '{0}{1}.dat'.format(args.load_model_dir, args.load_model),
        map_location=lambda storage, loc: storage)

    env = atari_env("{}".format(args.env), env_conf, args)
    env.seed(0)
    num_tests = 0
    start_time = time.time()
    reward_total_sum = 0
    player = Agent(None, env, args, None)

    if args.mask_double:
        from models.model_mask_double import A3C
    elif args.mask_single_p:
        from models.model_mask_single_policy import A3C
    elif args.mask_single_v:
        from models.model_mask_single_value import A3C
    else:
        from models.model import A3C
    player.model = A3C(args, player.env.observation_space.shape[0],
                        player.env.action_space)
    player.gpu_id = gpu_id
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()

    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model.load_state_dict(saved_state)
    else:
        player.model.load_state_dict(saved_state)

    player.model.eval()
    print('Total params: %.2fM' % (sum(p.numel() for p in player.model.parameters())/1000000.0))


    scores = []
    for i_episode in tqdm(range(args.num_episodes)):
        raw_save_dir = path.join(args.load_model, 'episode{}_raw'.format(i_episode))
        if not path.exists(raw_save_dir):
            os.mkdir(raw_save_dir)
        if (args.mask_single_p or args.mask_double) and args.image:
            att_p_save_dir = path.join(args.load_model, 'episode{}_att_p'.format(i_episode))
            if not path.exists(att_p_save_dir):
                os.mkdir(att_p_save_dir)
        if (args.mask_single_v or args.mask_double) and args.image:
            att_v_save_dir = path.join(args.load_model, 'episode{}_att_v'.format(i_episode))
            if not path.exists(att_v_save_dir):
                os.mkdir(att_v_save_dir)
        rew_info = path.join(args.load_model, 'episode{}_reward.txt'.format(i_episode))
        fr = open(rew_info, 'w')
        sta_info = path.join(args.load_model, 'episode{}_state.txt'.format(i_episode))
        fs = open(sta_info, 'w')
        action_info = path.join(args.load_model, 'episode{}_action.txt'.format(i_episode))
        fa = open(action_info, 'w')
        value_info = path.join(args.load_model, 'episode{}_value.txt'.format(i_episode))
        fv = open(value_info, 'w')

        in_state, conf = player.env.reset()
        player.state = torch.from_numpy(in_state).float()
        player.visualizer = conf[0]
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.state = player.state.cuda()
        player.eps_len += 2
        reward_sum = 0

        img_idx = 0
        model_features = []
        model_atts = []
        model_atts_p = []
        model_atts_v = []
        while True:
            if args.render:
                player.env.render()

            if args.image:
                #raw img save
                crop1, crop2, dim = conf[1]["crop1"], conf[1]["crop2"], conf[1]["dimension2"]
                raw_save_path = path.join(raw_save_dir, 'raw_{0:06d}.png'.format(img_idx))
                cv2.imwrite(raw_save_path, player.visualizer[:, :, ::-1].copy())

            player.action_test()
            reward_sum += player.reward
            fa.write(str(img_idx) + ', ' + str(player.test_action[0]) + '\n')
            fr.write(str(img_idx) + ', ' + str(reward_sum) + '\n')
            fv.write(str(img_idx) + ', ' + str(player.values[0][0].cpu().detach().numpy().copy()) + '\n')

            if args.image and (args.mask_single_p or args.mask_double):
                sy, sx, sc = player.visualizer.shape
                att_p_map = np.zeros((sy, sx))
                model_att_p = player.model.att_p_sig5.cpu()
                model_att_p = model_att_p.numpy()
                model_att_p = model_att_p[0]
                model_att_p = model_att_p.transpose(1,2,0)[np.newaxis, :, :, :]
                model_atts_p = model_att_p if model_atts_p == [] else np.concatenate([model_atts_p,model_att_p])
            if args.image and (args.mask_single_v or args.mask_double):
                att_v_map = np.zeros((sy, sx))
                model_att_v = player.model.att_v_sig5.cpu()
                model_att_v = model_att_v.numpy()
                model_att_v = model_att_v[0]
                model_att_v = model_att_v.transpose(1,2,0)[np.newaxis, :, :, :]
                model_atts_v = model_att_v if model_atts_v == [] else np.concatenate([model_atts_v,model_att_v])

            np_value = player.values.cpu()
            np_value = np_value.numpy()
            fs.write(str(img_idx) + ', ' + str(float(np_value[0][0])) + '\n')

            img_idx += 1

            if player.done and not player.info:
                state = player.env.reset()
                in_state, conf = state
                player.eps_len += 2
                player.state = torch.from_numpy(in_state).float()
                player.visualizer = conf[0]
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.state = player.state.cuda()
            elif player.info:
                num_tests += 1
                scores.append(reward_sum)
                score_mean = mean(scores)
                if num_tests > 1:
                    score_variance = variance(scores)
                    score_stdev = stdev(scores)
                else:
                    score_variance = 0
                    score_stdev = 0
                if num_tests == 1:
                    reward_max = reward_sum
                else:
                    if reward_max < reward_sum:
                        reward_max = reward_sum
                player.eps_len = 0
                break

        if args.image and (args.mask_single_p or args.mask_single_v or args.mask_double):
            # normalization (mask-attention)
            if args.mask_single_p or args.mask_double:
                max_len = model_atts_p.max(axis=None, keepdims=True)
                min_len = model_atts_p.min(axis=None, keepdims=True)
                model_atts_p = min_max(model_atts_p, min_len, max_len)
            if args.mask_single_v or args.mask_double:
                max_len = model_atts_v.max(axis=None, keepdims=True)
                min_len = model_atts_v.min(axis=None, keepdims=True)
                model_atts_v = min_max(model_atts_v, min_len, max_len)
            for i in range(img_idx):
                raw_save_path = path.join(raw_save_dir, 'raw_{0:06d}.png'.format(i))
                raw_img = cv2.imread(raw_save_path)
                #mask-attention save
                if args.mask_single_p or args.mask_double:
                    att_map_p = np.zeros((sy, sx))
                    model_att_p = model_atts_p[i] * 255.
                    cv2.imwrite('./att.png', model_att_p)
                    res1_att_p = cv2.resize(model_att_p, (80, dim))
                    res2_att_p = cv2.resize(res1_att_p, (160, 160 - crop1 + crop2))
                    att_map_p[crop1 : crop2 + 160, : 160] = res2_att_p
                    att_map_p = cv2.applyColorMap(att_map_p.astype(np.uint8), cv2.COLORMAP_JET)
                    att_map_p = cv2.addWeighted(raw_img, 0.7, att_map_p, 0.3, 0)
                    #att_map_p = cv2.addWeighted(raw_img, 1.0, att_map_p, 1.0, 0)
                    att_p_save_path = path.join(att_p_save_dir, 'att_p_{0:06d}.png'.format(i))
                    cv2.imwrite(att_p_save_path, att_map_p)
                if args.mask_single_v or args.mask_double:
                    att_map_v = np.zeros((sy, sx))
                    model_att_v = model_atts_v[i] * 255.
                    cv2.imwrite('./att.png', model_att_v)
                    res1_att_v = cv2.resize(model_att_v, (80, dim))
                    res2_att_v = cv2.resize(res1_att_v, (160, 160 - crop1 + crop2))
                    att_map_v[crop1 : crop2 + 160, : 160] = res2_att_v
                    att_map_v = cv2.applyColorMap(att_map_v.astype(np.uint8), cv2.COLORMAP_JET)
                    att_map_v = cv2.addWeighted(raw_img, 0.7, att_map_v, 0.3, 0)
                    #att_map_v = cv2.addWeighted(raw_img, 1.0, att_map_v, 1.0, 0)
                    att_v_save_path = path.join(att_v_save_dir, 'att_v_{0:06d}.png'.format(i))
                    cv2.imwrite(att_v_save_path, att_map_v)

    print('# of test :', num_tests)
    print("Time {0}, length {1}, score:{2} mean:{3:.2f} variance:{4:.2f} stdev:{5:.2f} max:{6}".
            format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                    player.eps_len, reward_sum, score_mean, score_variance, score_stdev, reward_max))