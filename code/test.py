from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
from utils import setup_logger
from player_util import Agent
from torch.autograd import Variable
import time
import csv
import os
from environment import atari_env

def log_writer(log_path, global_step, time, episode_len, reward_sum, reward_mean):
    with open(log_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([global_step, time, episode_len, reward_sum, reward_mean])

def test(args, shared_model, env_conf, global_step, start_time):
    ptitle('Test Agent')
    gpu_id = args.gpu_ids[-1]

    save_interval = 1000000
    next_save = 0

    if args.mask_double: model_name = 'Mask-A3C-double'
    elif args.mask_single_p: model_name = 'Mask-A3C-single-policy'
    elif args.mask_single_v: model_name = 'Mask-A3C-single-value'
    else: model_name = 'A3C'
    if args.convlstm: model_name += '+ConvLSTM'

    test_log_path = os.path.join(args.log_dir, args.env)
    test_log_path = os.path.join(test_log_path, "{}_{}_test.csv".format(args.env, model_name))

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    env = atari_env(args.env, env_conf, args)

    reward_sum = 0
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    
    if args.mask_double:
        from models.model_mask_double import A3C
    elif args.mask_single_p:
        from models.model_mask_single_policy import A3C
    elif args.mask_single_v:
        from models.model_mask_single_value import A3C
    else:
        from models.model import A3C
    player.model = A3C(args, player.env.observation_space.shape[0], player.env.action_space)

    player.state = player.env.reset()
    player.eps_len += 2

    in_state, conf = player.env.reset()
    player.state = in_state
    player.state = torch.from_numpy(player.state).float()

    env.seed(args.seed + args.workers)
    env.action_space.seed(args.seed + args.workers)

    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.state = player.state.cuda()
    flag = True
    max_score = 0

    while True:
        if global_step.value > args.max_global_step:
            break

        if flag:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())
            player.model.eval()
            flag = False

        player.action_test()
        reward_sum += player.reward

        if player.done and not player.info:
            state = player.env.reset()
            player.eps_len += 2

            in_state, conf = player.env.reset()
            player.state = torch.from_numpy(in_state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
        elif player.info:
            flag = True
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            test_time = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time()-start_time.value))
            print('\033[31m' + 'Test: time {0} | global step {1} | episode length {2} | score {3} | score mean {4:.4f}'.format(
                test_time, global_step.value, player.eps_len, reward_sum, reward_mean) + '\033[0m')
            log_writer(test_log_path, global_step.value, test_time, player.eps_len, reward_sum, reward_mean)

            if args.save_max and reward_sum >= max_score:
                max_score = reward_sum
                print("Best Model (score:{}) save.......".format(max_score))
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        state_to_save = player.model.state_dict()
                        torch.save(state_to_save, '{0}{1}_{2}_best.dat'.format(
                            args.save_model_dir, args.env, model_name))
                else:
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, '{0}{1}_{2}_best.dat'.format(
                        args.save_model_dir, args.env, model_name))
            
            if global_step.value > next_save:
                print("{}step Model save.........".format(global_step.value))
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        state_to_save = player.model.state_dict()
                        torch.save(state_to_save, '{0}{1}_{2}.dat'.format(
                            args.save_model_dir, args.env, model_name,))
                else:
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, '{0}{1}_{2}.dat'.format(
                        args.save_model_dir, args.env, model_name))
                next_save = global_step.value + save_interval

            reward_sum = 0
            player.eps_len = 0
            state = player.env.reset()
            player.eps_len += 2
            time.sleep(10)

            in_state, conf = player.env.reset()
            player.state = torch.from_numpy(in_state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
