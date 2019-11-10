import gym
import numpy as np
import torch

from dqn import DQNAgent


def main():
    # env = gym.make('Pendulum-v0')
    # env = gym.make('Acrobot-v1')
    # env = gym.make('CartPole-v1')
    env = gym.make('MountainCar-v0')
    # env = gym.make('HalfCheetah-v3')

    agent = DQNAgent(env)

    print(env.observation_space, env.action_space)

    for ep in range(1000):
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
