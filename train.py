import math
import random
from time import time

import torch as torch

DEVICE = "cpu"

try:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
except:
    DEVICE = "cpu"

DEVICE = "cpu"
print("CPU DEV HARDCODED")

#Default reward shaping/feature translation function
null_f = lambda x: x

#Necessary for easy code alteration - sometimes states are tensors, but sometimes not.
def state_to_device(state):
    for key in state.keys():
        state[key] = torch.tensor(state[key], dtype=torch.float32).to(DEVICE)

def step_aggregator(env, action, steps=4):
    states = []
    rewards = []
    dones = []
    infos = []
    for _ in range(steps):
        if not done:
            state, reward, done, info = env.step(action)
            print("STATE IS")
            print(state)
            raise Exception
            state = state_to_device(state)
        else:
            reward = 0

        states.append(state)
        rewards.append(reward)
        dones.append(dones)
        infos.append(infos)

    print(f"Aggregate {aggregate}")
    return aggregate, reward, done, all_info

def test_run(n, approximator, env, render=False):
    for i in range(n):
        state = state_to_device(env.reset())
        done = False
        rewards = []
        while not done:
            state, reward, done, _ = step_aggregator(env, approximator.get_action(state))
            rewards.append(reward)
            state = state_to_device(state)
            if render:
                env.render()
        print(f"For {i}, {sum(rewards):3.3f}")

def train_episode(approximator, env, shape_f=null_f, feature_f=null_f, seed=None, render=False):
    """Trains approximator on env for one episode.
    approximator can be any function approximator, but must have two functions:
        get_action(state), and train(prev_state, action, state, reward).
    env is the openai gym environment to learn in.
    shape_f is a function that takes in a state, reward tuple and returns a possibly shaped reward.
    feature_f is a function that takes in a state and ouputs a feature vector.
    seed is the seed to seed the environment with. None -> no seed set.
    render is whether or not to render the episode.
    """
    if seed:
        env.seed(seed)
    done = False
    state = env.reset()
    state = state_to_device(state)

    reward_history = []

    while not done:
        action = approximator.get_action(feature_f(state).to(DEVICE))
        new_state, reward, done, _ = step_aggregator(env, action)
        if render:
            env.render()
        new_state = state_to_device(new_state)
        action = torch.tensor(action).to(DEVICE)
        reward = torch.tensor(reward, dtype=torch.float32).to(DEVICE)
        approximator.train(feature_f(state), action, feature_f(new_state), shape_f(reward), done)
        state=new_state
        reward_history.append(float(reward))

    return reward_history

def train_episodes(num_episodes, approximator, env, shape_f=null_f, feature_f=null_f, seed=None, log=False, render=False):
    """Trains approximator for num_episodes. See train_episode for details on args.
    Seed is only set before the first episode, if passed."""
    state = env.reset()
    done = False
    q_episode = []
    while not done:
        q_episode.append(state_to_device(state))
        state, _, done, _ = step_aggregator(env, env.action_space.sample())

    if seed:
        env.seed(seed)
    frames = 0
    tally = 0
    highest_tally = -float('inf')
    last_tally = 0
    times = []
    then = time()
    for i in range(1, num_episodes + 1):
        reward_history = train_episode(approximator, env, render=render)
        total_reward = sum(reward_history)
        frames += len(reward_history)
        tally += total_reward
        print(f"Reward for {i} ({frames} frames): {total_reward:3.1f} (peak: {highest_tally:3.1f} prev: {last_tally:3.1f}) ({torch.cuda.memory_allocated()})", end='\r')
        if i % 100 == 0:
            times.append(int(time() - then))
            then = time()
            print(f"Reward for last hundred averages to {tally/100:3.3f}")# times is {times}")
            last_tally = tally / 100
            if last_tally > highest_tally:
                highest_tally = last_tally
                approximator.save(f"{approximator.name}_best")
            tally = 0
        if log:
            if i % 100 == 1:
                approximator.log_episode(total_reward, q_episode)
                approximator.save()
            else:
                approximator.log_episode(total_reward)
            if i % 1000 == 0:
                approximator.save(f"{approximator.name}_{i}")
    approximator.save()
