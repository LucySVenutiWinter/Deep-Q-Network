import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import os

from functools import partial

import random

BATCH_SIZE = 32

DEVICE = "cpu"

try:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
except:
    DEVICE = "cpu"

CHECKPOINT_DIRECTORY = "checkpoints"
LOG_DIRECTORY = "logs"

ACT_SPACE_SIZE = 2
OBS_SPACE_SIZE = 4

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        #Input is four 240x320 arrays stacked together
        #Pytorch wants these in CHW form (channels first)
        #In: 4*240*320
        self.conv1 = nn.Conv2D(in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=2)
        #32*60*80
        self.mp1 = nn.MaxPool2d(2)
        #32*30*40
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv2D(32, 64, 4, 2, 1)
        #64*15*20
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv2D(64, 64, 3, 1)
        #64*18*13
        self.bn3 = nn.BatchNorm1d(64)
        #64*18*13 is 14976, plus the two non-screen inputs
        self.fc1 = nn.Linear(64*18*13+2, 512)
        self.fc2 = nn.Linear(512, 3)
        self.lrelu = nn.LeakyRelu()

    def forward(self, x):
        out = self.conv1(x)
        out = self.mp1(out)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.lrelu(out)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = self.fc2(out)

        return out

class DQN(nn.Module):
    def __init__(self, name=None, criterion=nn.MSELoss, discount=0.99, buffer_size=1000000, epsilon_end=1000000):
        super(DQN, self).__init__()

        self.name = name
        self.log = None

        self.network = ConvNet()
            
        self.optimizer = optim.Adam(self.parameters())

        if name==None or not os.path.exists(os.path.join(CHECKPOINT_DIRECTORY, name)):
            self.episode = 0
            self.epsilon = DecayingValue(1, 1/epsilon_end, "arithmetic")
        else:
            self.load()

        #We need to log if given a name, regardless of whether or not the log exists already
        if name:
            self.set_logfile(name + ".csv")

        self.to(DEVICE)

        self.criterion = criterion()
        self.discount = discount
        self.chooser = torch.utils.data.WeightedRandomSampler
        self.training = True

        self.buffer = ExperienceBuffer(buffer_size, OBS_SPACE_SIZE)

        self.target = type(self.network)()
        self.target.to(DEVICE)
        print("thing")
        print(self.target.requires_grad)
        self.target.requires_grad_(False)
        print("thing")
        print(self.target.requires_grad)
        self.target.load_state_dict(self.network.state_dict())
        print("thing")
        print(self.target.requires_grad)

        self.optimizer = optim.Adam(self.network.parameters())
        self.train_counter = 0

    def save(self, name=None):
        if not name:
            name = self.name
        if not name:
            raise Exception("Tried to save an unnamed model")
        os.makedirs(CHECKPOINT_DIRECTORY, exist_ok=True)
        torch.save({
            'episode': self.episode,
            'optimizer': self.optimizer,
            'network': self.network,
            'epsilon': self.epsilon,
            }, os.path.join(CHECKPOINT_DIRECTORY, name))

    def load(self):
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIRECTORY, self.name))
        self.network = checkpoint['network']
        self.optimizer = checkpoint['optimizer']
        self.episode = checkpoint['episode']
        self.epsilon = checkpoint['epsilon']

    def __del__(self):
        if self.log:
            self.log.close()
            
    #Creates a log file, or if one exists, opens it and sets the episode appropriately
    def set_logfile(self, filename):
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        filename = os.path.join(LOG_DIRECTORY, filename)
        if os.path.isfile(filename):
            self.log = open(filename, 'a')
        else:
            self.log = open(filename, 'w')
            self.log.write("episode,reward,Q\n")

    #Log the results of an episode. Note that this increments the episode number.
    def log_episode(self, total_rewards, q_episode=None):
        if not self.log:
            raise Exception("Tried to log an episode with no open log")
        self.episode = self.episode + 1
        if not q_episode:
            self.log.write(f"{self.episode},{total_rewards}\n")
        else:
            self.log.write(f"{self.episode},{total_rewards},{q_score(q_episode, self)}\n")

        self.log.flush()

    def predict(self, X):
        return self.network(X.to(DEVICE))

    def get_action(self, X):
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
            self.network.eval()
        if self.training == True:
            if self.epsilon:
                if random.random() < self.epsilon():
                    return self.get_epsilon_action(X)
                else:
                    return self.get_best_action(X)
            else:
                return self.get_rand_action(X)
        else:
            return self.get_best_action(X)

    #Return the most valuable action
    def get_rand_action(self, X):
        soft_pred = self.predict(X)
        soft_pred = nn.functional.softmax(soft_pred, dim=1)
        return list(self.chooser(soft_pred, 1))[0]

    def get_epsilon_action(self, X):
        return random.randint(0, ACT_SPACE_SIZE-1)

    def get_best_action(self, X):
        soft_pred = self.predict(X)
        return int(torch.argmax(soft_pred, dim=1))

    #Return the highest value of the input, expects the input to be the output of the network
    def get_best_value(self, X):
        val, _ = torch.max(X, dim=1)
        return val

    def train(self, prev_state, action, state, reward, done):
        self.network.train()

        self.buffer.add_state(prev_state, reward, action, done)
        if not self.buffer.ok_to_sample(BATCH_SIZE):
            return
    
        self.train_counter += 1
        if self.train_counter % 15000 == 0:
            print("Don't forget this one")
            self.target.load_state_dict(self.network.state_dict())
            self.target.requires_grad_(False)
            self.train_counter = 0

        self.optimizer.zero_grad()

        #Neeed.... two q values to make this work... the one we're updating for and the future one's max value..
        states, successors, actions, rewards, done_b = self.buffer.get_random(BATCH_SIZE)

        #Dims should be batch_size, actions
        old_q = self.predict(states)
        old_q = torch.gather(old_q, 1, actions)
        with torch.no_grad():
            new_q = self.target(successors)
            new_q, _ = torch.max(new_q, dim=1, keepdims=True)
            targ = rewards + new_q * done_b * self.discount

        loss = self.criterion(old_q, targ)

        loss.backward()

        self.optimizer.step()

        self.network.eval()

class CircleBuffer():
    """A CircleBuffer maintains a torch tensor of some size. Old members are overwritten."""

    def __init__(self, shape, dtype=torch.float32):
        """Init parameters:
            shape (tuple of ints): The shape of the buffer. Should be (size, <shape of a single element>).
            dtype: The torch type that the buffer is filled with."""
        self.buffer = torch.empty(shape, dtype=dtype, requires_grad=False).to(DEVICE)
        self.type = dtype
        self.size = shape[0]
        self.index = 0
        self.full = False

    def find_break(self):
        """Returns the current index, where access is discontinuous.
        The index holds the oldest entry, so its predecessor is temporally unrelated."""
        return self.index

    def add_entry(self, example):
        """Add a single entry to the buffer. Does not check for shape/compatibility first!"""
        self.buffer[self.index] = example
        self.index += 1
        if self.index == self.size:
            self.index = 0
            self.full = True

    def get_entries(self, entry_indexes):
        """Creates a tensor where each element is one example.
        
        Parameters:
            entry_indexes (list of ints): The list of examples to fill the result tensor with.

        Raises:
            RuntimeError if any index is negative, greater than the buffer size, or has not yet been added."""
        max_ent = max(entry_indexes)
        min_ent = min(entry_indexes)
        if not self.full and max_ent >= self.index:
            raise RuntimeError(f"Tried to get entry {max_ent} in a CircleBuffer with {self.index} entry_indexes")
        if max_ent > self.size:
            raise RuntimeError(f"Tried to get entry {max_ent} in a CircleBuffer of size {self.size}")
        if min_ent < 0:
            raise RuntimeError(f"Tried to get entry {min_entry}")

        return_shape = [len(entry_indexes)]
        return_shape += self.buffer.shape[1:]

        entries = torch.empty(return_shape, dtype=self.type, requires_grad=False).to(DEVICE)

        entries = self.buffer[entry_indexes]

        return entries

class ExperienceBuffer():
    """The ExperienceBuffer holds data to replay during training, and processes that data into batches.

    The buffer holds state, action, and reward data, as well as information on if a state was terminal.
    There are a few things worth noting, implementationally:
        When adding states, it expects that it is being given a state, the action taken in that state,
        and the reward achieved for that action. This may be different from what is expected, as 
        frequently, the rewards are associated with the following/successor state.

        Additionally, it saves the "done" buffer in an unusual format. If a state is terminal (the
        Markov chain is finished), the done buffer stores a zero, and if it is not terminal, it stores
        a one. This is the reverse of the C convention. The reason for this is due to how Q learning works.
        At termination, value is updated solely with reward information, not with any value estimate
        of the terminal state. Thus, returning a vector where terminal states are marked with zeros
        and nonterminal ones with ones allows a simple broadcast multiplication to set the value
        of any terminal state's successor to zero while preserving values of nonterminal states.
        The terminal states' successors are actually the first state of the next episode, anyway."""
    def __init__(self, size, state_size):
        """Initialization parameters:
            size (int): The number of examples to save.
            state_size: The size of a single example."""

        #We can save a /lot/ of space if, instead of saving states and their successors, we just maintain one list.
        #However, that requires increasing the size, since one element will lack a predecessor and be unusable.
        size += 1
        shape = (size, state_size)
        scalar_shape = (size, 1)
        self._state_buffer = CircleBuffer(shape, torch.float32)
        self._action_buffer = CircleBuffer(scalar_shape, torch.int64)
        self._reward_buffer = CircleBuffer(scalar_shape, torch.float32)
        self._done_buffer = CircleBuffer(scalar_shape, torch.int)#Not bool, because of how it's used

    def add_state(self, state, reward, action, done):
        """Adds the information required to create the replay buffer.

        Parameters:
            state: The state vector in question.
            reward: The reward received on the following time step, for the action taken in the state.
            action: The action taken in the state.
            done: Whether or not the state is terminal.

        Note that the reward is not the action received upon transition into a state, but instead out."""
        self._state_buffer.add_entry(state)
        #We will multiply q values by the done buffer, which will zero them in the case of a final frame
        if done:
            self._done_buffer.add_entry(0)
        else:
            self._done_buffer.add_entry(1)
        self._reward_buffer.add_entry(reward)
        self._action_buffer.add_entry(action)
    
    def ok_to_sample(self, num_episodes):
        """Checks if it's legal to request num_episodes samples."""
        return self._state_buffer.full or num_episodes <= self._state_buffer.index

    def get_random(self, num_episodes):
        """Get a random batch of examples. There must be at least that many examples in the buffer.

        Parameters:
            num_episodes (int): The size of the batch

        Returns:
            A five-member tuple.
            First member: batch of random state vectors.
            Second member: batch of those vectors' successors.
            Third member: vector of actions taken in those states.
            Fourth member: vector of rewards achieved on transition into the successors.
            Fifth member: a vector containing data on if the state was terminal (0 if it was, 1 otherwise)."""
        assert self.ok_to_sample(num_episodes)

        #Set up random index function
        if self._state_buffer.full:
            rand = partial(random.randint, 0, self._state_buffer.size - 1)
        else:
            rand = partial(random.randint, 0, self._state_buffer.index - 1)

        #The point in the circle buffer that is disconnected with its predecessor.
        disjoint = self._state_buffer.find_break()

        samples = []
        sample_successors = []

        for i in range(num_episodes):
            idx = rand()

            #Do not pick a state where we don't have the successor anymore (or yet, if not full)
            while idx == disjoint - 1:
                idx = rand()
            next_idx = idx + 1
            #Deal with circular nature of buffer
            if next_idx == self._state_buffer.size:
                next_idx = 0

            samples.append(idx)
            sample_successors.append(next_idx)

        states = self._state_buffer.get_entries(samples)
        next_states = self._state_buffer.get_entries(sample_successors)
        action_values = self._action_buffer.get_entries(samples)
        reward_values = self._reward_buffer.get_entries(samples)
        done_values = self._done_buffer.get_entries(samples)

        return states, next_states, action_values, reward_values, done_values

#Defined in the Mnih et al Atari paper
#Use this with a consistent trajectory to estimate learning
def q_score(trajectory, network):
    network.network.eval()
    with torch.no_grad():
        score = [torch.max(network.predict(state)) for state in trajectory]
    network.network.train()
    return sum(score)/len(score)

class DecayingValue():
    """A self-decaying epsilon class."""
    def __init__(self, value, decay_factor, decay_mode, decay_min=0.1):
        """Sets up a decaying epsilon.

        Parameters:
            value (float): The starting value for epsilon. There are no checks,
            so you can set it above 1 if you want to pick entirely randomly for a set of time.
            decay_factor (float): The factor by which it decays. Meaning varies based on mode.
            decay_mode (string): 'geometric' or 'arithmetic'. If geometric, epsilon's value
            is multiplied by the decay factor. If arithmetic, the decay factor is subtracted.
            decay_min (float): the minimum epsilon value."""
        assert decay_mode == "geometric" or decay_mode == "arithmetic"
        self.value = value
        self.decay_factor = decay_factor
        self.decay_mode = decay_mode
        self.decay_min = decay_min

    def get(self):
        """Returns the value WITHOUT decaying it. For actual use, call it."""
        return self.value

    def __call__(self):
        """Returns the value, decaying it appropriately."""
        if self.value == self.decay_min:
            return self.value

        val = self.value

        if self.decay_mode == "geometric":
            self.value *= self.decay_factor
        elif self.decay_mode == "arithmetic":
            self.value -= self.decay_factor
        self.value = max(self.value, self.decay_min)

        return val
