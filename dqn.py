import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer
from stats import Stats

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
    g = s + np.random.randn(*s.shape) * astd * 10
    return g

def goal_reward(s, g):
    r = (np.linalg.norm((s-g)/std, axis=-1) < 0.1).astype(np.float)
    # r = (np.linalg.norm((s-g), axis=-1) < 0.1).astype(np.float)
    return r



class DQNAlgo:
    def __init__(self, net, tnet, replay_buffer, obs_dim, num_act, gamma):
        self.net = net
        self.tnet = tnet
        self.replay_buffer = replay_buffer
        self.optim = optim.Adam(self.net.parameters(), lr=1e-3)
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
            return {}

        s, a, extr, sp, done, gg, hg = self.replay_buffer.sample_batch(batch_size=batch_size)

        # std = s.std(axis=0)
        # astd = (s-sp).std(axis=0)

        her_prob = 1.0
        g = gen_goal(s)
        her_mask = np.random.binomial(size=(batch_size,), n=1, p=her_prob)[:,np.newaxis]
        # g = sp * her_mask + gg * (1-her_mask)
        g = hg * her_mask + gg * (1-her_mask)

        r = goal_reward(sp, g)
        # print(r.sum())
        gdone = np.logical_or((r > 0), done)

        s = torch.from_numpy(s).float()
        sp = torch.from_numpy(sp).float()
        a = torch.from_numpy(a).long()
        g = torch.from_numpy(g).float()
        r = torch.from_numpy(r).float()
        extr = torch.from_numpy(extr).float()
        done = torch.from_numpy(done).float()
        gdone = torch.from_numpy(gdone).float()

        # Qnext = (1 - gdone) * self.net(sp, g).detach().max(dim=-1)[0]
        # Qtarget = r + self.gamma * Qnext
        # Qa = self.net(s, g).gather(1, a.unsqueeze(1)).squeeze(1)

        # max_act = self.net(sp, sp).detach().max(dim=-1)[1]
        # Qnext = (1 - done) * self.tnet(sp, sp).detach().gather(1, max_act.unsqueeze(1)).squeeze(1)
        # Qtarget = extr + self.gamma * Qnext
        # Qa = self.net(s, s).gather(1, a.unsqueeze(1)).squeeze(1)

        max_act = self.net(sp, g).detach().max(dim=-1)[1]
        # Qnext = (1 - gdone) * self.tnet(sp, g).detach().gather(1, max_act.unsqueeze(1)).squeeze(1)
        Qnext = (1 - gdone) * torch.min(self.tnet(sp, g), self.net(sp, g)).detach().gather(1, max_act.unsqueeze(1)).squeeze(1)
        # Qnext = torch.min(Qnext, Qnext * 0. + 1.)
        Qtarget = r + self.gamma * Qnext
        Qa = self.net(s, g).gather(1, a.unsqueeze(1)).squeeze(1)

        loss = torch.mean(0.5 * torch.sum((Qtarget - Qa) ** 2, dim=-1))

        # print(f'Loss: {float(loss):.5f}\tAvgQ: {Qa.mean():.5f}\tAvgQt: {Qtarget.mean():.5f}')
        # print(f'AvgR: {r.mean():.5f}')

        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        target_update_rate = 0.01
        
        # for p, tp in zip(self.net.parameters(), self.tnet.parameters()):
            # tp.data += target_update_rate * (p - tp)
        if np.random.random() < target_update_rate:
            for p, tp in zip(self.net.parameters(), self.tnet.parameters()):
                tp.data += p.data
                p.data = tp.data - p.data
                tp.data -= p.data

        info = {
            'Loss': float(loss),
            'AvgQ': float(Qa.mean()),
            'AvgQt': float(Qtarget.mean()),
            'AvgR': float(r.mean()),
        }

        return info





class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.num_act = env.action_space.n

        self.net = DQNNet(self.obs_dim, self.num_act)
        self.tnet = DQNNet(self.obs_dim, self.num_act)
        self.replay_buffer = ReplayBuffer()
        
        gamma = 0.99
        self.algo = DQNAlgo(self.net, self.tnet, self.replay_buffer, self.obs_dim, self.num_act, gamma)

    def run_episode(self):
        s, done = self.env.reset(), False

        g = gen_goal(s)
        # g = s

        stats = Stats()

        episode = []

        epilen = 0
        extR = 0.
        intR = 0.
        while not done:
            epilen += 1
            a = self.algo.get_action(s, g, epsilon=0.05)
            # a = self.env.action_space.sample()
            sp, r, done, info = self.env.step(a)

            episode.append((s, a, r, sp, done, g))
            # self.replay_buffer.add_sample((s, a, r, sp, done, g))

            s = sp
            extR += r

            intR = max(intR, goal_reward(s, g) * (self.algo.gamma ** epilen))


            if epilen % 4 == 0:
                info = self.algo.update_batch()
                stats.update(info)

        rpl_len = 50
        for i in range(epilen):
            s, a, r, sp, done, g = episode[i]
            hg_idx = np.random.randint(i, min(epilen, i+rpl_len))
            hg = episode[hg_idx][0]
            self.replay_buffer.add_sample((s, a, r, sp, done, g, hg))

        print(f'Epilen: {epilen}\tExtR: {extR:.2f}\tIntR: {intR:.2f}')
        print(stats)


    def test_episode(self):
        s, done = self.env.reset(), False
        g = gen_goal(np.array([s]))[0]
        g = np.array([0.5, 0.0])

        R = 0

        min_dis = 1e9

        cnt = 0
        while not done:
            # self.env.render()
            cnt += 1
            if cnt >= 500: break
            a = self.algo.get_action(s, g, epsilon=0.05)
            sp, extr, done, info = self.env.step(a)
            r = goal_reward(sp, g)
            R += r

            # min_dis = min(min_dis, np.linalg.norm((sp-g)/std))
            min_dis = min(min_dis, np.linalg.norm((sp-g)))

            if r > 0:
                done = True

        return R, min_dis






