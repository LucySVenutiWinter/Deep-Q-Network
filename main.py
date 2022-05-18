import gym
from vizdoom import gym_wrapper

import train
import dqn

import sys

ENVIRONMENT = "VizdoomDefendLine-v0"

env = gym.make(ENVIRONMENT)

state = env.reset()

def print_help():
    print(f"Usage: {sys.argv[0]} <mode> <name> <value>")
    print("Mode must be either \"train\" or \"eval\"")
    print("Name can be any name which can be saved to disk")
    print("If in train mode, train for value episodes")
    print("If in eval mode, show value episodes")
    print("If value is invalid or not given, it's set to 10 for eval and 10000 for train")

if len(sys.argv) > 2:
    name = sys.argv[2]

    num_in = 0
    try:
        num_in = int(sys.argv[3])
    except:
        pass

    if num_in < 1:
        if sys.argv[1] == "eval":
            num_in = 10
        if sys.argv[1] == "train":
            num_in = 10000
        if num_in >= 1:
            print(f"Setting number of episodes to default ({num_in} for {sys.argv[1]})")

    if sys.argv[1] == "eval":
        dqnet = dqn.DQN(name=name)
        dqnet.training = False
        train.test_run(num_in, dqnet, env, render=True)
    elif sys.argv[1] == "train":
        dqnet = dqn.DQN(name=name)
        train.train_episodes(num_in, dqnet, env, log=True)
    else:
        print_help()
else:
    print_help()
