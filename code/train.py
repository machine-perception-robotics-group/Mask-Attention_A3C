from __future__ import division
from setproctitle import setproctitle as ptitle
import time
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads
from player_util import Agent
from torch.autograd import Variable
import csv
import os


def log_writer(log_path, time, global_step, score):
    with open(log_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([time, global_step, score])


def train(rank, args, shared_model, optimizer, env_conf, global_step, start_time):
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed_all(args.seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if args.mask_double: model_name = 'Mask-A3C-double'
    elif args.mask_single_p: model_name = 'Mask-A3C-single-policy'
    elif args.mask_single_v: model_name = 'Mask-A3C-single-value'
    else: model_name = 'A3C'
    if args.convlstm: model_name += '+ConvLSTM'

    train_log_path = os.path.join(args.log_dir, args.env)
    train_log_path = os.path.join(train_log_path, "{}_{}.csv".format(args.env, model_name))

    env = atari_env(args.env, env_conf, args)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    env.seed(args.seed + rank)
    env.action_space.seed(args.seed + rank)
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

    #player.state = player.env.reset()
    in_state, conf = player.env.reset()
    player.state = in_state
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model = player.model.cuda()
    player.model.train()
    player.eps_len += 2
    

    episode = 1
    w_score = 0

    while True:
        if global_step.value > args.max_global_step:
            break

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())
        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.cx = Variable(torch.zeros(1, 64, 10, 10).cuda())
                    player.hx = Variable(torch.zeros(1, 64, 10, 10).cuda())
                    player.cx2 = Variable(torch.zeros(1, 256, 4, 4).cuda())
                    player.hx2 = Variable(torch.zeros(1, 256, 4, 4).cuda())
            else:
                player.cx = Variable(torch.zeros(1, 64, 10, 10))
                player.hx = Variable(torch.zeros(1, 64, 10, 10))
                player.cx2 = Variable(torch.zeros(1, 256, 4, 4))
                player.hx2 = Variable(torch.zeros(1, 256, 4, 4))
        else:
            player.cx = Variable(player.cx.data)
            player.hx = Variable(player.hx.data)
            player.cx2 = Variable(player.cx2.data)
            player.hx2 = Variable(player.hx2.data)

        for step in range(args.num_steps):
            player.action_train()
            w_score += player.reward
            global_step.value += 1
            if player.done:
                break

        if player.done:
            train_time = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time()-start_time.value))
            print("{0}|{1}| worker{2}| time:{3}, global step:{4}, score:{5}".format(args.env, model_name, rank, train_time,
                                                                           global_step.value, w_score))
            log_writer(train_log_path, train_time, global_step.value, w_score)
            w_score = 0
            episode += 1
            state = player.env.reset()

            in_state, conf = player.env.reset()
            player.state = torch.from_numpy(in_state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        R = torch.zeros(1, 1)
        if not player.done:
            value, _, _, _ = player.model((Variable(player.state.unsqueeze(0)),
                                        (player.hx, player.cx), (player.hx2, player.cx2)))
            R = value.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()
        R = Variable(R)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = player.rewards[i] + args.gamma * \
                player.values[i + 1].data - player.values[i].data

            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                player.log_probs[i] * \
                Variable(gae) - 0.01 * player.entropies[i]

        player.model.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
        player.clear_actions()

