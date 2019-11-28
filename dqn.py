import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer
from stats import Stats

class DQNNet(nn.Module):
    def __init__(self, obs_dim, num_act, hidden_dims=[400, 300]):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim*2, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], num_act)

    def forward(self, s, g):
        sg = torch.cat([s, g-s], dim=-1)
        x = F.relu(self.fc1(sg))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAlgo:
    def __init__(self, obs_dim, num_act, gamma, lr=1e-3):
        self.obs_dim = obs_dim
        self.num_act = num_act
        self.gamma = gamma

        self.target_update_rate = 0.005

        self.update_count = 0

        self.net = DQNNet(self.obs_dim, self.num_act)
        self.net_target = DQNNet(self.obs_dim, self.num_act)

        self.optim = optim.Adam(self.net.parameters(), lr=lr)

    def get_action(self, s, g, epsilon=0.):
        with torch.no_grad():
            s = torch.from_numpy(s).float()
            g = torch.from_numpy(g).float()
            amax = self.net(s, g).argmax(dim=0)

        if np.random.random() < epsilon:
            amax = np.random.randint(self.num_act)
        return int(amax)
        

    def update_batch(self, batch):
        self.update_count += 1

        s, a, r, sp, done, g = batch

        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).long()
        r = torch.from_numpy(r).float()
        sp = torch.from_numpy(sp).float()
        done = torch.from_numpy(done).float()
        g = torch.from_numpy(g).float()

        # Update Q networks

        max_act = self.net(sp, g).detach().max(dim=-1)[1]
        Qnext = (1 - done) * torch.min(self.net_target(sp, g), self.net(sp, g)).detach().gather(1, max_act.unsqueeze(1)).squeeze(1)
        Qtarget = r + self.gamma * Qnext
        Qa = self.net(s, g).gather(1, a.unsqueeze(1)).squeeze(1)

        loss = torch.mean((Qtarget - Qa) ** 2)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        for net, net_target in [
                (self.net, self.net_target)]:
            for p, tp in zip(net.parameters(), net_target.parameters()):
                tp.data += self.target_update_rate * (p.data - tp.data)

        info = {
            'Loss': float(loss),
            'AvgQ': float(Qa.mean()),
            'AvgR': float(r.mean()),
        }

        return info



class DQNAlgoOld:
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


