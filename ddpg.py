import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer
from stats import Stats

class DDPGCriticNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims=[400, 300]):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim*2 + act_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0] + act_dim, hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)

    def forward(self, s, g, a):
        sg = torch.cat([s, g-s, a], dim=-1)
        x = F.relu(self.fc1(sg))
        xa = torch.cat([x, a], dim=-1)
        x = F.relu(self.fc2(xa))
        x = self.fc3(x)
        return x

class DDPGActorNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims=[400, 300]):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim*2, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], act_dim)

    def forward(self, s, g):
        sg = torch.cat([s, g-s], dim=-1)
        x = F.relu(self.fc1(sg))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class DDPGAlgo:
    def __init__(self, obs_dim, act_dim, gamma, lr=1e-3, device='cpu'):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.device = device

        self.target_update_rate = 0.005
        self.actor_update_period = 2

        self.update_count = 0

        self.anet = DDPGActorNet(self.obs_dim, self.act_dim).to(device)
        self.anet_target = DDPGActorNet(self.obs_dim, self.act_dim).to(device)
        self.cnet1 = DDPGCriticNet(self.obs_dim, self.act_dim).to(device)
        self.cnet1_target = DDPGCriticNet(self.obs_dim, self.act_dim).to(device)
        self.cnet2 = DDPGCriticNet(self.obs_dim, self.act_dim).to(device)
        self.cnet2_target = DDPGCriticNet(self.obs_dim, self.act_dim).to(device)

        self.optim_a = optim.Adam(self.anet.parameters(), lr=lr)
        self.optim_c = optim.Adam([
            {'params': self.cnet1.parameters()},
            {'params': self.cnet2.parameters()},
        ], lr=lr)

    def get_action(self, s, g, sigma=0., target=False):
        with torch.no_grad():
            if not isinstance(s, torch.Tensor):
                s = torch.from_numpy(s).float().to(self.device)
                g = torch.from_numpy(g).float().to(self.device)
            if target:
                amax = self.anet_target(s, g)
            else:
                amax = self.anet(s, g)
        amax = amax.cpu().numpy()
        amax += np.random.normal(size=amax.shape) * sigma
        #TODO: Clip action

        return amax
        

    def update_batch(self, batch):
        self.update_count += 1

        s, a, r, sp, done, g = batch

        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).float().to(self.device)
        r = torch.from_numpy(r).float().to(self.device)
        sp = torch.from_numpy(sp).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)
        g = torch.from_numpy(g).float().to(self.device)

        # Update Critic
        
        ap = self.get_action(sp, g, target=True, sigma=0.2)
        ap = torch.from_numpy(ap).float().to(self.device)

        Qnext = (1 - done) * torch.min(self.cnet1_target(sp, g, ap), self.cnet2_target(sp, g, ap)).detach()
        Qtarget = r + self.gamma * Qnext
        Qa1 = self.cnet1(s, g, a)
        Qa2 = self.cnet2(s, g, a)

        critic_loss = torch.mean((Qtarget - Qa1) ** 2) + torch.mean((Qtarget - Qa2) ** 2)

        self.optim_c.zero_grad()
        critic_loss.backward()
        self.optim_c.step()

        info = {
            'CriticLoss': float(critic_loss),
            'AvgQ': float(Qa1.mean()),
            'AvgR': float(r.mean()),
        }

        # Update Actor
        
        if self.update_count % self.actor_update_period == 0:
            a_pi = self.anet(s, g)
            Q_pi = self.cnet1(s, g, a_pi)

            actor_loss = -torch.mean(Q_pi)

            self.optim_a.zero_grad()
            actor_loss.backward()
            self.optim_a.step()

            info.update({
                'ActorLoss': float(actor_loss)
            })

            # Update Target Networks
            
            for net, net_target in [
                    (self.anet, self.anet_target),
                    (self.cnet1, self.cnet1_target), 
                    (self.cnet2, self.cnet2_target)]:
                for p, tp in zip(net.parameters(), net_target.parameters()):
                    tp.data += self.target_update_rate * (p.data - tp.data)


        return info
