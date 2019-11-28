import gym
import numpy as np
import torch

from uvfa import UVFAgent
from rl import StandardRLAgent

device = torch.device('cuda')
# device = torch.device('cpu')

def main():
    # env = gym.make('Pendulum-v0')
    # env = gym.make('Acrobot-v1')
    # env = gym.make('CartPole-v1')
    # env = gym.make('MountainCar-v0')
    # env = gym.make('MountainCarContinuous-v0')
    env = gym.make('HalfCheetah-v3')
    # env = gym.make('InvertedPendulum-v2')
    # env = gym.make('Ant-v3')

    # agent = UVFAgent(env)
    agent = StandardRLAgent(env, device=device)

    print(env.observation_space, env.action_space)

    num_episode = 20000
    # num_episode = 0

    for ep in range(num_episode):
        print(ep)
        agent.run_episode()

        if ep % 10 == 0:
            print('==Test==')
            r, m = agent.test_episode()
            print(f'Test R: {r:.2f}')


        # print(s)
        # Control features linearly as subgoal

        # print(env.s)

    returns = []
    min_dis = []
    for ep in range(20):
        # print(ep)
        r, m = agent.test_episode()
        returns.append(r)
        min_dis.append(m)

    print('Avg return =', np.mean(returns))
    print('Min dis =', np.mean(min_dis))


if __name__ == '__main__':
    main()
