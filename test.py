import gym
import numpy as np
import torch

from uvfa import UVFAgent
from rl import StandardRLAgent


def main():
    # env = gym.make('Pendulum-v0')
    # env = gym.make('Acrobot-v1')
    # env = gym.make('CartPole-v1')
    env = gym.make('MountainCar-v0')
    # env = gym.make('HalfCheetah-v3')

    # agent = UVFAgent(env)
    agent = StandardRLAgent(env)

    print(env.observation_space, env.action_space)

    num_episode = 2000
    # num_episode = 0

    for ep in range(num_episode):
        print(ep)
        agent.run_episode()


        # print(s)
        # Control features linearly as subgoal

        # print(env.s)

    returns = []
    min_dis = []
    for ep in range(500):
        r, m = agent.test_episode()
        returns.append(r)
        min_dis.append(m)

    print('Avg return =', np.mean(returns))
    print('Min dis =', np.mean(min_dis))


if __name__ == '__main__':
    main()
