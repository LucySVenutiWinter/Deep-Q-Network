import gym
from vizdoom import gym_wrapper

import train
import dqn

import sys

EPISODES = 50000
ENVIRONMENT = "VizdoomDefendLine-v0"

env = gym.make(ENVIRONMENT)

#print("info:")
#print(env.action_space)
state = env.reset()
#print(state.items())
#print(state['rgb'].shape)
#raise Exception


if len(sys.argv) == 2:
    if sys.argv[1] == "eval":
        dqnet = dqn.DQN(name="dqn_default")
        dqnet.load()
        dqnet.training = False
        train.test_run(10, dqnet, env, render=True)
        raise Exception

dqnet = dqn.DQN(name="dqn_default")
train.train_episodes(EPISODES, dqnet, env, log=True)
