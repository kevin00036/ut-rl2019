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

class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_dims=[400, 300]):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim*2, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)

    def forward(self, s, g):
        sg = torch.cat([s, g-s], dim=-1)
        x = F.relu(self.fc1(sg))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNWithRewardAlgo:
    def __init__(self, obs_dim, num_act, gamma, use_td3=True, lr=1e-3, device='cpu'):
        self.obs_dim = obs_dim
        self.num_act = num_act
        self.gamma = gamma
        self.device = device
        self.use_td3 = use_td3

        self.target_update_rate = 0.005

        self.update_count = 0

        self.qnet = DQNNet(self.obs_dim, self.num_act).to(device)
        self.qnet_target = DQNNet(self.obs_dim, self.num_act).to(device)
        self.rnet = DQNNet(self.obs_dim, self.num_act).to(device)
        self.rnet_target = DQNNet(self.obs_dim, self.num_act).to(device)

        self.optim = optim.Adam([
            {'params': self.qnet.parameters()},
            {'params': self.rnet.parameters()},
        ], lr=lr)

    def get_action(self, s, g, epsilon=0.):
        with torch.no_grad():
            s = torch.from_numpy(s).float().to(self.device)
            g = torch.from_numpy(g).float().to(self.device)
            amax = self.qnet(s, g).argmax(dim=0)

        if np.random.random() < epsilon:
            amax = np.random.randint(self.num_act)
        return int(amax)

    def get_values(self, s, g, target=False):
        qnet = self.qnet_target if target else self.qnet
        rnet = self.rnet_target if target else self.rnet

        q = qnet(s, g)
        ival, amax = q.max(dim=-1)
        r = rnet(s, g)
        rval = r.gather(-1, amax.unsqueeze(-1)).squeeze(-1)

        return ival, rval, amax

    def update_batch(self, batch):
        self.update_count += 1

        s, a, r, extr, sp, done, g = batch

        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        r = torch.from_numpy(r).float().to(self.device)
        extr = torch.from_numpy(extr).float().to(self.device)
        sp = torch.from_numpy(sp).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)
        g = torch.from_numpy(g).float().to(self.device)

        # Update Q networks

        if self.use_td3:
            max_act = self.qnet(sp, g).detach().max(dim=-1)[1]
            Qnext = (1 - done) * torch.min(self.qnet_target(sp, g), self.qnet(sp, g)).detach().gather(1, max_act.unsqueeze(1)).squeeze(1)
        else:
            max_act = self.qnet_target(sp, g).detach().max(dim=-1)[1]
            Qnext = (1 - done) * self.qnet_target(sp, g).detach().gather(1, max_act.unsqueeze(1)).squeeze(1)

        Qtarget = r + self.gamma * Qnext
        Qa = self.qnet(s, g).gather(1, a.unsqueeze(1)).squeeze(1)

        loss_q = torch.mean((Qtarget - Qa) ** 2)

        # Update R networks

        Rnext = (1 - done) * self.rnet_target(sp, g).detach().gather(1, max_act.unsqueeze(1)).squeeze(1)
        Rtarget = extr + self.gamma * Rnext
        Ra = self.rnet(s, g).gather(1, a.unsqueeze(1)).squeeze(1)

        loss_r = torch.mean((Rtarget - Ra) ** 2)

        loss = loss_q + loss_r

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        for net, net_target in [
                (self.qnet, self.qnet_target), (self.rnet, self.rnet_target)]:
            for p, tp in zip(net.parameters(), net_target.parameters()):
                tp.data += self.target_update_rate * (p.data - tp.data)

        info = {
            'Loss': float(loss),
            'AvgQ': float(Qa.mean()),
            'AvgRV': float(Ra.mean()),
            'AvgR': float(r.mean()),
        }

        return info

