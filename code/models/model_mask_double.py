from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import norm_col_init, weights_init
from convLSTM import ConvLSTMCell

class A3C(torch.nn.Module):
    def __init__(self, args, num_inputs, action_space):
        super(A3C, self).__init__()
        if args.convlstm:
            print('\033[31m' + "Mask A3C (double) + ConvLSTM" + '\033[0m')
        else:
            print('\033[31m' + "Mask A3C (double)" + '\033[0m')
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.use_lstm = args.convlstm
        if self.use_lstm:
            self.convlstm1 = ConvLSTMCell(input_size=(10, 10), input_dim=64, hidden_dim=64, kernel_size=(3, 3), bias=True)

        # critic
        self.att_conv_c1 = nn.Conv2d(64, 1, 1, stride=1, padding=0)
        self.sigmoid_c = nn.Sigmoid()
        self.c_conv = nn.Conv2d(64, 32, 1, stride=1, padding=0)
        self.critic_linear = nn.Linear(3200, 1)
        
        # actor
        self.att_conv_a1 = nn.Conv2d(64, 1, 1, stride=1, padding=0)
        self.sigmoid_a = nn.Sigmoid()
        self.a_conv = nn.Conv2d(64, 32, 1, stride=1, padding=0)
        num_outputs = action_space.n
        self.actor_linear = nn.Linear(3200, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.a_conv.weight.data.mul_(relu_gain)
        self.c_conv.weight.data.mul_(relu_gain)
        self.att_conv_a1.weight.data.mul_(relu_gain)
        self.att_conv_c1.weight.data.mul_(relu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.train()

    def forward(self, inputs):
        # Feature extractor
        inputs, (hx, cx), (hx2, cx2) = inputs
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        if self.use_lstm:
            hx, cx = self.convlstm1(input_tensor=x, cur_state=[hx, cx])
            x = hx

        # Critic
        c_x = F.relu(self.c_conv(x))
        att_v_feature = self.att_conv_c1(x)
        self.att_v = self.sigmoid_c(att_v_feature) # mask-attention
        self.att_v_sig5 = self.sigmoid_c(att_v_feature * 5.0)
        c_mask_x = c_x * self.att_v # mask processing
        c_x = c_mask_x

        c_x = c_x.view(c_x.size(0), -1)
 

        # Actor
        a_x = F.relu(self.a_conv(x))
        att_p_feature = self.att_conv_a1(x)
        self.att_p = self.sigmoid_a(att_p_feature) # mask-attention
        self.att_p_sig5 = self.sigmoid_a(att_p_feature * 5.0)
        a_mask_x = a_x * self.att_p # mask processing
        a_x = a_mask_x

        a_x = a_x.view(a_x.size(0), -1)

        return self.critic_linear(c_x), self.actor_linear(a_x), (hx, cx), (hx2, cx2)
