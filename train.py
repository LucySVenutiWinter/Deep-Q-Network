import math
import random
from time import time

import torch as torch
import numpy as np

import config

STEPS = config.STEPS
DEVICE = config.DEVICE

#Converts the elements of input 'rgb' in the state dictionary to [-1, 1] singles
def state_to_device(state):
    state = torch.tensor(state, dtype=torch.float32).to(DEVICE)
    state = torch.sum(state, 0) / (255*3)#Normalize to [0, 1]
    state = state * 2 - 1#Convert to [-1, 1]
    return state

#Game cannot be reset directly, as it returns a single frame for state
def resetter(game):
    game.new_episode()
    state = game.get_state().screen_buffer
    state = state_to_device(state)
    states = []
    for _ in range(STEPS):
        states.append(state)

    states = torch.concat(states, 0)
    state = convert_to_input(states)

    return state

#Converts an input array to one dimensional form
def convert_to_input(states):
    states_1d = torch.reshape(states, (-1, 1*STEPS*240*320))
    return torch.squeeze(states_1d)

#Given an action, repeatedly steps the game with it
def step_aggregator(game, action):
    states = []
    rewards = []
    infos = []
    done = False
    oh_action = np.zeros(config.ACT_SPACE)
    oh_action[action] = 1
    for _ in range(STEPS):
        if not done:
            #We need the terminal screen, which becomes inaccessible once the game is done
            state = state_to_device(game.get_state().screen_buffer)
            game.make_action(oh_action)
            done = game.is_episode_finished()
            if done:
                reward = 0
            else:
                state = game.get_state().screen_buffer
                reward = game.get_last_reward()
                state = state_to_device(state)
        else:
            reward = 0

        states.append(state)
        rewards.append(reward)
        infos.append(infos)

    states = torch.concat(states, 0)

    states = convert_to_input(states)
    rewards = sum(rewards)
    return states, rewards, done, infos

#Runs the approximator n times in evaluation mode.
def test_run(n, approximator, game):
    for i in range(n):
        state = resetter(game)
        done = False
        rewards = []
        while not done:
            now = time()
            state, reward, done, _ = step_aggregator(game, approximator.get_action(state))
            rewards.append(reward)
            while (time() - now) < 1/10:
                pass
        print(f"For {i+1}, {sum(rewards):3.3f}")

def train_episode(approximator, game):
    """Trains approximator on game for one episode.
    approximator can be any function approximator, but must have two functions:
        get_action(state), and train(prev_state, action, state, reward).
    game is the initialized ViZDOOM game to learn in.
    """
    done = False
    state = resetter(game)

    reward_history = []

    while not done:
        action = approximator.get_action(state)
        new_state, reward, done, _ = step_aggregator(game, action)
        action = torch.tensor(action).to(DEVICE)
        reward = torch.tensor(reward, dtype=torch.float32).to(DEVICE)
        approximator.train(state, action, new_state, reward, done)
        state=new_state
        reward_history.append(reward)

    return reward_history

def train_episodes(num_episodes, approximator, game, log=False):
    """Trains approximator for num_episodes. See train_episode for details on args."""
    state = resetter(game)
    done = False
    q_episode = []
    while not done:
        q_episode.append(state)
        state, _, done, _ = step_aggregator(game, random.randint(0, config.ACT_SPACE - 1))
    q_episode.append(state)

    frames = 0
    tally = 0
    highest_tally = -float('inf')
    last_tally = 0
    times = []
    then = time()
    for i in range(1, num_episodes + 1):
        reward_history = train_episode(approximator, game)
        total_reward = sum(reward_history)
        frames += len(reward_history)
        tally += total_reward
        print(f"Reward for episode {i} ({frames} frames): {int(total_reward)} (peak: {highest_tally:3.1f} prev: {last_tally:3.1f}) (network trained for {approximator.episode} total episodes)", end='\r')
        if i % 100 == 0:
            times.append(int(time() - then))
            then = time()
            print(f"\nReward for last hundred averages to {tally/100:3.3f}")
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
