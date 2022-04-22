import gym

import train
import dqn

import sys

EPISODES = 50000
ENVIRONMENT = "CartPole-v1"

env = gym.make(ENVIRONMENT)

if len(sys.argv) == 2:
    if sys.argv[1] == "eval":
        dqnet = dqn.DQN(name="dqn_default_best")
        dqnet.load()
        dqnet.training = False
        train.test_run(10, dqnet, env, render=True)
        raise Exception

dqnet = dqn.DQN(name="dqn_default")
train.train_episodes(EPISODES, dqnet, env, log=True)
