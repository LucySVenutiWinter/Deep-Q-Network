import gym
from vizdoom import gym_wrapper

import train
import dqn

import sys

ENVIRONMENT = "VizdoomDefendLine-v0"

env = gym.make(ENVIRONMENT)

state = env.reset()

def print_help():
    print(f"Usage: {sys.argv[0]} <mode> <value> <name>")
    print("Mode must be either \"train\" or \"eval\"")
    print("If in train mode, train for value episodes")
    print("If in eval mode, show value episodes")
    print("If name is not given, \"dqn_default\" is used")

if len(sys.argv) > 1 and len(sys.argv) < 5:
    if len(sys.argv) < 4:
        name = "dqn_default"
    else:
        name = sys.argv[3]

    try:
        num_in = int(sys.argv[2])
    except:
        if sys.argv[1] == "eval":
            num_in = 10
            print(f"Given no or bad number, assuming {num_in}")
        elif sys.argv[1] == "train":
            num_in = 10000
            print(f"Given no or bad number, assuming {num_in}")

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
