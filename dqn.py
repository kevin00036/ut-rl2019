import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer

class DQNNet(nn.Module):
    def __init__(self, obs_dim, num_act):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim*2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_act)

    def forward(self, s, g):
        sg = torch.cat([s, g-s], dim=-1)
        x = F.relu(self.fc1(sg))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# std = np.array([0.05, 0.5, 0.05, 0.5])
# astd = np.array([0.01, 0.2, 0.01, 0.2])
std = np.array([0.1, 0.01])
astd = np.array([0.01, 0.001])

def gen_goal(s):
    g = s + np.random.randn(*s.shape) * astd
    return g

def goal_reward(s, g):
    r = (np.linalg.norm((s-g)/std, axis=-1) < 0.1).astype(np.float)
    return r



class DQNAlgo:
    def __init__(self, net, replay_buffer, obs_dim, num_act, gamma):
        self.net = net
        self.replay_buffer = replay_buffer
        self.optim = optim.Adam(self.net.parameters(), lr=3e-4)
        self.obs_dim = obs_dim
        self.num_act = num_act
        self.gamma = gamma

    def get_action(self, s, g, epsilon=0.):
        with torch.no_grad():
            s = torch.from_numpy(s).float()
            g = torch.from_numpy(g).float()
            amax = self.net(s, g).argmax(dim=0)

        if np.random.random() < epsilon:
            amax = np.random.randint(self.num_act)
        return int(amax)
        

    def update_batch(self, batch_size=32):
        if self.replay_buffer.size() < batch_size:
            return

        s, a, _, sp, gg = self.replay_buffer.sample_batch(batch_size=batch_size)

        std = s.std(axis=0)
        astd = (s-sp).std(axis=0)

        her_prob = 0.5
        g = gen_goal(s)
        her_mask = np.random.binomial(size=(batch_size,), n=1, p=her_prob)[:,np.newaxis]
        g = sp * her_mask + g * (1-her_mask)

        r = goal_reward(sp, g)
        print(r.sum())
        gdone = (r > 0)

        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).long()
        g = torch.from_numpy(g).float()
        r = torch.from_numpy(r).float()
        gdone = torch.from_numpy(gdone).float()

        Qnext = (1 - gdone) * self.net(s, g).detach().max(dim=-1)[0]
        Qtarget = r + self.gamma * Qnext
        Qa = self.net(s, g).gather(1, a.unsqueeze(1))
        loss = torch.mean(0.5 * torch.sum((Qtarget - Qa) ** 2, dim=-1))

        print('Loss:', float(loss))

        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()





class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.num_act = env.action_space.n

        self.net = DQNNet(self.obs_dim, self.num_act)
        self.replay_buffer = ReplayBuffer()
        
        gamma = 0.95
        self.algo = DQNAlgo(self.net, self.replay_buffer, self.obs_dim, self.num_act, gamma)

    def run_episode(self):
        s, done = self.env.reset(), False

        g = gen_goal(s)
        # g = s

        while not done:
            a = self.algo.get_action(s, g, epsilon=0.5)
            # a = self.env.action_space.sample()
            sp, r, done, info = self.env.step(a)

            self.replay_buffer.add_sample((s, a, r, sp, g))

            s = sp

        self.algo.update_batch()

        # print(s)

    def test_episode(self):
        s, done = self.env.reset(), False
        g = gen_goal(np.array([s]))[0]

        R = 0

        min_dis = 1e9

        cnt = 0
        while not done:
            # self.env.render()
            cnt += 1
            if cnt >= 100: break
            a = self.algo.get_action(s, g, epsilon=0.05)
            sp, extr, done, info = self.env.step(a)
            r = goal_reward(sp, g)
            R += r

            min_dis = min(min_dis, np.linalg.norm((sp-g)/std))

            if r > 0:
                done = True

        return R, min_dis






