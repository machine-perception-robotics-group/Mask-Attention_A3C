from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class Agent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        self.hx2 = None
        self.cx2 = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.visualizer = None
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1

    def action_train(self):
        value, logit, (self.hx, self.cx), (self.hx2, self.cx2) = self.model((Variable(
            self.state.unsqueeze(0)), (self.hx, self.cx), (self.hx2, self.cx2)))
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, Variable(action))
        state, self.reward, self.done, self.info = self.env.step(action.cpu().numpy())

        in_state, conf = state
        self.state = torch.from_numpy(in_state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        return self

    def action_test(self):
        with torch.no_grad():
            if self.done:
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        self.cx = Variable(torch.zeros(1, 64, 10, 10).cuda())
                        self.hx = Variable(torch.zeros(1, 64, 10, 10).cuda())
                        self.cx2 = Variable(torch.zeros(1, 256, 4, 4).cuda())
                        self.hx2 = Variable(torch.zeros(1, 256, 4, 4).cuda())
                else:
                    self.cx = Variable(torch.zeros(1, 64, 10, 10))
                    self.hx = Variable(torch.zeros(1, 64, 10, 10))
                    self.cx2 = Variable(torch.zeros(1, 256, 4, 4))
                    self.hx2 = Variable(torch.zeros(1, 256, 4, 4))
            else:
                self.cx = Variable(self.cx.data)
                self.hx = Variable(self.hx.data)
                self.cx2 = Variable(self.cx2.data)
                self.hx2 = Variable(self.hx2.data)

            value, logit, (self.hx, self.cx), (self.hx2, self.cx2) = self.model((Variable(
                self.state.unsqueeze(0)), (self.hx, self.cx), (self.hx2, self.cx2)))
        prob = F.softmax(logit, dim=1)
        action = prob.max(1)[1].data.cpu().numpy()
        self.test_action = action
        self.values = value
        state, self.reward, self.done, self.info = self.env.step(action[0])
        in_state, conf = state
        self.visualizer = conf[0]
        self.state = torch.from_numpy(in_state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self
