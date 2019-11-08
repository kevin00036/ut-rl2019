import gym
import numpy as np
import torch


def main():
    env = gym.make('Pendulum-v0')
    # env = gym.make('Acrobot-v1')

    print(env.observation_space, env.action_space)

    state, don = env.reset(), False

    print(state)

    print(env.observation_space.sample())
    print(env.action_space.sample())

if __name__ == '__main__':
    main()
