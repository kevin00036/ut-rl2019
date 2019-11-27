import gym

def modify_done(env, done):
    if not done:
        return False

    if isinstance(env, gym.wrappers.TimeLimit):
        if env._elapsed_steps >= env._max_episode_steps:
            return False

    return True
