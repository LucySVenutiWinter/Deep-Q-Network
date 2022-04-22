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

#Default reward shaping/feature translation function
null_f = lambda x: x

def test_run(n, approximator, env, render=False):
    for i in range(n):
        state = torch.tensor(env.reset(), dtype=torch.float32)
        done = False
        rewards = []
        while not done:
            state, reward, done, _ = env.step(approximator.get_action(state))
            rewards.append(reward)
            state = torch.tensor(state, dtype=torch.float32)
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
    state = torch.tensor(state, dtype=torch.float32)

    reward_history = []

    while not done:
        action = approximator.get_action(feature_f(state).to(DEVICE))
        new_state, reward, done, _ = env.step(action)
        if render:
            env.render()
        new_state = torch.tensor(new_state, dtype=torch.float32).to(DEVICE)
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
        q_episode.append(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE))
        state, _, done, _ = env.step(env.action_space.sample())

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