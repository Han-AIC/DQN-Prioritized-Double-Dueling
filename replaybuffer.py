import numpy as np
import random
import torch
from collections import namedtuple, deque, OrderedDict, Counter
import copy


class ReplayBuffer():
  def __init__(self,
               memory_size,
               batch_size,
               greedy_coeff,
               default_priority,
               shed_amount):

    self.current_size = 0
    self.curr_idx = 0
    self.greedy_coeff = greedy_coeff
    self.default_priority = default_priority
    self.shed_amount = shed_amount
    self.max_size = memory_size
    self.batch_size = batch_size
    self.memory_buffer = {}
    self.memory_template = namedtuple('memory', field_names=['index',
                                                            'priority',
                                                            'state',
                                                            'action',
                                                            'reward',
                                                            'next_state',
                                                            'done_flags'])

  def add(self,
          state,
          action,
          reward,
          next_state,
          done):
    if self.current_size >= self.max_size:
      self.shed()
      self.reindex_buffer()
      self.curr_idx = len(self.memory_buffer.keys())
    self.memory_buffer.update({self.curr_idx : self.memory_template(self.curr_idx,
                                                                    self.default_priority,
                                                                    state,
                                                                    action,
                                                                    reward,
                                                                    next_state,
                                                                    done)})
    self.curr_idx += 1

  def sample(self):
    sample = []
    if np.random.random() > self.greedy_coeff:
      sample, probs = self.choose_greedy_sample()
    else:
      sample, probs = self.choose_random_sample()

    indices = [x.index for x in sample]
    priorities = np.vstack([x.priority for x in sample])
    states = np.vstack([x.state for x in sample])
    actions = np.vstack([x.action for x in sample])
    rewards = np.vstack([x.reward for x in sample])
    next_states = np.vstack([x.next_state for x in sample])
    done_flags = np.vstack([x.done_flags for x in sample])

    return [indices,
            torch.from_numpy(priorities),
            torch.from_numpy(states),
            torch.from_numpy(actions),
            torch.from_numpy(rewards),
            torch.from_numpy(next_states),
            torch.from_numpy(done_flags)], probs

  def choose_greedy_sample(self):
    # Highest priority has a probability to be taken but it is not certain.
    indices = [x[0] for x in self.memory_buffer.items()]
    summed_priorities = sum([x[1].priority for x in self.memory_buffer.items()])
    probs = [(x[1].priority / summed_priorities) for x in self.memory_buffer.items()]
    chosen = list(np.random.choice(indices,
                                  self.batch_size,
                                  p=probs,
                                  replace=False))
    return [self.memory_buffer[i] for i in chosen], torch.tensor([probs[i] for i in chosen])

    # Take highest priority deterministically.
    # k = Counter({x[0]:x[1].priority for x in self.memory_buffer.items()})
    # return [self.memory_buffer[y[0]] for y in k.most_common(self.batch_size)]

  def choose_random_sample(self):
    # print("random")
    indices = random.sample(self.memory_buffer.keys(), k=self.batch_size)
    probs = [1/len(self.memory_buffer.keys()) for i in range(self.batch_size)]
    return [self.memory_buffer[k] for k in indices], torch.tensor(probs)

  def update_priority(self,
                      indices,
                      new_priorities):
    for i in range(len(indices)):
      entry = self.memory_buffer[indices[i]]
      self.memory_buffer.update({indices[i] : self.memory_template(entry.index,
                                                            new_priorities[i][0],
                                                            entry.state,
                                                            entry.action,
                                                            entry.reward,
                                                            entry.next_state,
                                                            entry.done_flags) })

  def shed(self):
    sample = random.sample(memory_buffer.keys(), self.shed_amount)
    for key in sample:
      del self.memory_buffer[key]

  def reindex_buffer(self):
    self.memory_buffer = {idx: x[1] for idx, x in enumerate(self.memory_buffer.items())}

  def print_buffer(self, start_idx):
    print(self.memory_buffer[-start_idx:])
