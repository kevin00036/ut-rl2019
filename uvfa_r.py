import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym

from replay_buffer import ReplayBuffer
from stats import Stats
from gymtool import modify_done

from dqn_r import DQNWithRewardAlgo
from ddpg import DDPGAlgo
from planning import Planner

class UVFAWithRewardAgent:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.is_discrete_action = isinstance(env.action_space, gym.spaces.discrete.Discrete)
        self.obs_dim = env.observation_space.shape[0]

        gamma = 0.99
        self.gamma = gamma

        if self.is_discrete_action:
            self.num_act = env.action_space.n
            self.algo = DQNWithRewardAlgo(self.obs_dim, self.num_act, gamma, device=device)
        else:
            self.act_dim = env.action_space.shape[0]
            self.algo = DDPGAlgo(self.obs_dim, self.act_dim, gamma, device=device)

        self.replay_buffer = ReplayBuffer()

        self.planner = Planner(trans_fn=self.planner_trans_fn, gamma=gamma, device=device)

        self.estimate_std()

    def estimate_std(self):
        tot_cnt = 0
        states = []
        action_deltas = []
        while tot_cnt < 10000:
            s, done = self.env.reset(), False
            while not done:
                tot_cnt += 1
                states.append(s)
                a = self.env.action_space.sample()
                sp, r, done, _ = self.env.step(a)
                action_deltas.append(sp - s)
                s = sp

        self.std = np.std(states, axis=0) + 1e-8
        self.astd = np.std(action_deltas, axis=0) + 1e-8

        print('Std', self.std, 'Action-Std', self.astd)

    def gen_goal(self, s):
        g = s + np.random.randn(*s.shape) * self.astd * np.random.randint(1, 10)
        return g

    def goal_reward(self, s, g):
        # r = (np.linalg.norm((s-g) / self.std, axis=-1) < 0.1).astype(np.float)
        r = (np.linalg.norm((s-g) / self.astd, axis=-1) < 1).astype(np.float)
        return r

    def update_batch(self, batch_size=32):
        if self.replay_buffer.size() < batch_size * 10:
            return {}

        batch = self.replay_buffer.sample_batch(batch_size=batch_size)
        return self.algo.update_batch(batch)


    def run_episode(self):
        s, done = self.env.reset(), False

        g = self.gen_goal(s)
        # g = s

        stats = Stats()

        episode = []

        epilen = 0
        extR = 0.
        intR = 0.
        while not done:
            epilen += 1
            if len(self.replay_buffer) < 10000:
                a = self.env.action_space.sample()
            elif self.is_discrete_action:
                a = self.algo.get_action(s, g, epsilon=0.05)
            else:
                a = self.algo.get_action(s, g, sigma=0.1)
            sp, r, done, info = self.env.step(a)
            mdone = modify_done(self.env, done)

            episode.append((s, a, r, sp, mdone, g))

            s = sp
            extR += r

            intR = max(intR, self.goal_reward(s, g) * (self.algo.gamma ** epilen))

            if epilen % 1 == 0:
                info = self.update_batch()
                stats.update(info)

        her_prob = 0.5
        rpl_len = 10
        for i in range(epilen):
            s, a, extr, sp, done, g = episode[i]
            if np.random.random() < her_prob:
                hg_idx = np.random.randint(i, min(epilen, i+rpl_len))
                g = episode[hg_idx][0]
            r = self.goal_reward(sp, g)
            done = np.logical_or((r > 0), done)
            self.replay_buffer.add_sample((s, a, r, extr, sp, done, g))

        print(f'Epilen: {epilen}\tExtR: {extR:.2f}\tIntR: {intR:.2f}')
        print(stats)


    def test_episode(self):
        s, done = self.env.reset(), False
        g = self.gen_goal(s)
        # g = np.array([0.5, 0.0])

        ss = torch.from_numpy(s).float().to(self.device).unsqueeze(0)
        gg = torch.from_numpy(g).float().to(self.device).unsqueeze(0)
        Q0, R0, _ = self.algo.get_values(ss, gg)
        Q0 = float(Q0)
        R0 = float(R0)

        ExtR = 0.
        DiscExtR = 0.
        IntR = 0.

        gamma_power = 1.0

        min_dis = 1e9

        cnt = 0
        while not done:
            # self.env.render()
            cnt += 1
            # if cnt >= 500: break
            if self.is_discrete_action:
                a = self.algo.get_action(s, g, epsilon=0.)
            else:
                a = self.algo.get_action(s, g, sigma=0.)
            sp, extr, done, info = self.env.step(a)
            mdone = modify_done(self.env, done)
            r = self.goal_reward(sp, g)

            ExtR += extr
            DiscExtR += gamma_power * extr
            IntR += gamma_power * r
            
            gamma_power *= self.gamma

            min_dis = min(min_dis, np.linalg.norm((sp-g)/self.std))

            s = sp
            if r > 0:
                done = True

        info = {
            'ExtR': ExtR,
            'DiscExtR': DiscExtR,
            'DiscExtR_Est': R0,
            'IntR': IntR,
            'IntR_Est': Q0,
        }

        return info

    def planner_trans_fn(self, s, g):
        n, m = s.shape[0], g.shape[0]
        s = s.unsqueeze(1).expand(-1, m, -1)
        g = g.unsqueeze(0).expand(n, -1, -1)

        with torch.no_grad():
            G, R, Pi = self.algo.get_values(s, g)
        return G, R, Pi

    def update_planner(self):
        n = 1000
        if len(self.replay_buffer) < n:
            return False

        waypoints = self.replay_buffer.sample_batch(n, replace=False)[0] # s
        print(waypoints.shape)
        self.planner.set_waypoint_states(waypoints)
        self.planner.update_trans()
        self.planner.pre_plan()

        return True

    def plan_episode(self, show_plan=False):
        s, done = self.env.reset(), False

        if show_plan:
            self.planner.show_plan(s, self.env)

        ExtR = 0
        gamma_power = 1.0

        step = 0
        while not done:
            step += 1
            # if step >= 10: break
            # if show_plan:
                # self.env.render()

            a = self.planner.plan(s)

            sp, extr, done, info = self.env.step(a)
            ExtR += extr

            s = sp

        return ExtR







