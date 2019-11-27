import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym

from replay_buffer import ReplayBuffer
from stats import Stats
from gymtool import modify_done

from dqn import DQNAlgo
from ddpg import DDPGAlgo

class StandardRLAgent:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.is_discrete_action = isinstance(env.action_space, gym.spaces.discrete.Discrete)
        self.obs_dim = env.observation_space.shape[0]

        gamma = 0.99

        if self.is_discrete_action:
            self.num_act = env.action_space.n
            self.algo = DQNAlgo(self.obs_dim, self.num_act, gamma, device=device)
        else:
            self.act_dim = env.action_space.shape[0]
            self.algo = DDPGAlgo(self.obs_dim, self.act_dim, gamma, device=device)

        self.replay_buffer = ReplayBuffer()

    def update_batch(self, batch_size=100):
        if self.replay_buffer.size() < batch_size * 10:
            return {}

        batch = self.replay_buffer.sample_batch(batch_size=batch_size)
        return self.algo.update_batch(batch)


    def run_episode(self):
        s, done = self.env.reset(), False
        zero = np.zeros_like(s)

        stats = Stats()

        epilen = 0
        R = 0.
        while not done:
            epilen += 1
            if self.is_discrete_action:
                a = self.algo.get_action(s, zero, epsilon=0.05)
            else:
                a = self.algo.get_action(s, zero, sigma=0.1)
            if len(self.replay_buffer) < 10000:
                a = self.env.action_space.sample()
            sp, r, done, _ = self.env.step(a)
            mdone = modify_done(self.env, done)

            self.replay_buffer.add_sample((s, a, r, sp, mdone, zero))

            s = sp
            R += r

            if epilen % 1 == 0:
                info = self.update_batch()
                stats.update(info)

        print(f'Epilen: {epilen}\tR: {R:.2f}')
        print(stats)


    def test_episode(self):
        s, done = self.env.reset(), False
        zero = np.zeros_like(s)

        R = 0

        cnt = 0
        while not done:
            # self.env.render()
            cnt += 1
            # if cnt >= 500: break
            if self.is_discrete_action:
                a = self.algo.get_action(s, zero, epsilon=0.)
            else:
                a = self.algo.get_action(s, zero, sigma=0.)
            sp, r, done, _ = self.env.step(a)
            R += r
            s = sp

        return R, 0.

