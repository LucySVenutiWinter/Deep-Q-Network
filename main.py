import gym
from vizdoom import gym_wrapper

import train
import dqn

import sys

EPISODES = 10000
ENVIRONMENT = "VizdoomDefendLine-v0"

env = gym.make(ENVIRONMENT)

state = env.reset()

def print_help():
    print(f"Usage: {sys.argv[0]} <mode> <name>")
    print("Mode must be either \"train\" or \"eval\"")
    print("If name is not given, \"dqn_default\" is used")

if len(sys.argv) > 1 and len(sys.argv) < 4:
    if len(sys.argv) < 3:
        name = "dqn_default"
    else:
        name = sys.argv[2]

    if sys.argv[1] == "eval":
        dqnet = dqn.DQN(name=name)
        dqnet.training = False
        train.test_run(10, dqnet, env, render=True)
    elif sys.argv[1] == "train":
        dqnet = dqn.DQN(name=name)
        train.train_episodes(EPISODES, dqnet, env, log=True)
    else:
        print_help()
else:
    print_help()
