import gym
import numpy as np
import torch

from uvfa import UVFAgent
from rl import StandardRLAgent
from uvfa_r import UVFAWithRewardAgent
from stats import Stats
from stat_logger import StatLogger

device = torch.device('cuda')
# device = torch.device('cpu')

def main():
    # env = gym.make('Pendulum-v0')
    # env = gym.make('Acrobot-v1')
    env = gym.make('CartPole-v1')
    # env = gym.make('MountainCar-v0')
    # env = gym.make('MountainCarContinuous-v0')
    # env = gym.make('HalfCheetah-v3')
    # env = gym.make('InvertedPendulum-v2')
    # env = gym.make('Ant-v3')

    agent, agent_type = StandardRLAgent(env, device=device), 'rl'
    # agent, agent_type = UVFAWithRewardAgent(env, device=device), 'uvfa_r'

    run_name = env.spec.id + '_' + agent_type


    logger = StatLogger(run_name=run_name, aggregate_steps=2000)

    print(env.observation_space, env.action_space)

    num_episode = 200000
    # num_episode = 0

    max_steps = 200000

    total_steps = 0

    for ep in range(num_episode):
        if total_steps > max_steps:
            break
        print('Episode', ep, 'Total Step', total_steps)
        steps = agent.run_episode()
        total_steps += steps

        if ep % 10 == 0:
            print('==Test==')
            tep = 10
            stats = Stats()
            for i in range(tep):
                info = agent.test_episode()
                stats.update(info)
            print(stats)
        
            logger.add_data(total_steps, stats)

        if ep % 20 == 0 and isinstance(agent, UVFAWithRewardAgent):
            if agent.update_planner():
                tep = 10
                stats = Stats()
                for i in range(tep):
                    show_plan = (i == 0)
                    show_plan = False
                    info = agent.plan_episode(show_plan=show_plan)
                    stats.update(info)
                print(stats)

                logger.add_data(total_steps, stats)


        # print(s)
        # Control features linearly as subgoal

        # print(env.s)

    # returns = []
    # min_dis = []
    # for ep in range(20):
        # # print(ep)
        # r, m = agent.test_episode()
        # returns.append(r)
        # min_dis.append(m)

    # print('Avg return =', np.mean(returns))
    # print('Min dis =', np.mean(min_dis))


if __name__ == '__main__':
    main()
