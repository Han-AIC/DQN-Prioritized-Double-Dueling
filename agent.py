import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque, OrderedDict, Counter
import copy

from replaybuffer import ReplayBuffer
from model import Model, DuelingQ



class Agent():
  def __init__(self,
               action_shape,
               model_structure,
               agent_hyperparams,
               dueling,
               double):
    self.device = torch.device(agent_hyperparams["device"])
    self.action_shape = action_shape
    self.dueling = dueling
    self.double = double
    if self.dueling:
      prime = model_structure[0]
      value = model_structure[1]
      advantage = model_structure[2]
      self.local_model = DuelingQ(prime, value, advantage).to(self.device)
      self.target_model = DuelingQ(prime, value, advantage).to(self.device)
      self.target_model.load_state_dict(self.local_model.state_dict())
    else:
      self.local_model = Model(model_structure).to(self.device)
      self.target_model = Model(model_structure).to(self.device)
      self.target_model.load_state_dict(self.local_model.state_dict())

    self.optimizer = optim.RMSprop(self.local_model.parameters(),
                                lr=agent_hyperparams['lr'])

    self.replay_buffer = ReplayBuffer(agent_hyperparams['memory_size'],
                                      agent_hyperparams['batch_size'],
                                      agent_hyperparams['greedy_coeff'],
                                      agent_hyperparams['default_priority'],
                                      agent_hyperparams['shed_amount'])

    self.eps = agent_hyperparams['eps']
    self.alpha = agent_hyperparams['alpha']
    self.gamma = agent_hyperparams['gamma']
    self.beta = agent_hyperparams['beta']

    self.eps_decay = agent_hyperparams['eps_decay']
    self.alpha_decay = agent_hyperparams['alpha_decay']
    self.gamma_decay = agent_hyperparams['gamma_decay']
    self.beta_decay = agent_hyperparams['beta_decay']

    self.min_eps = agent_hyperparams['min_eps']
    self.min_alpha = agent_hyperparams['min_alpha']
    self.min_gamma = agent_hyperparams['min_gamma']
    self.min_beta = agent_hyperparams['min_beta']

  def load_saved_model(self, state_dict):
    self.local_model.load_state_dict(state_dict)
    self.target_model.load_state_dict(state_dict)

  def get_Q_values(self, state):
    state = torch.from_numpy(state)
    state = state.float().unsqueeze(0)
    state = state.to(self.device)
    self.local_model.eval()
    with torch.no_grad():
      Q_values = self.local_model(state)
    self.local_model.train()
    return Q_values

  def choose_action(self, state):
    if np.random.random() < self.eps:
      return np.random.choice(self.action_shape)
    else:
      action_values = self.get_Q_values(state).cpu().data.numpy()
      return np.argmax(action_values)

  def update_buffer(self,
                    state,
                    action,
                    reward,
                    next_state,
                    terminal):
    self.replay_buffer.add(state,
                          action,
                          reward,
                          next_state,
                          terminal)

  def decay_epsilon(self):
    self.eps = max(self.eps * self.eps_decay, self.min_eps)
    self.gamma = min(self.gamma * self.gamma_decay, self.min_gamma)

  def train_network(self):

    # ==========================
    # SAR tuples for a minibatch of experiences,
    # along with the probability values used to sample
    # each minibatch.
    sample, probs = self.replay_buffer.sample()
    indices = sample[0]
    priorities = sample[1]
    states = sample[2].float().to(self.device)
    actions = sample[3].long().to(self.device)
    rewards = sample[4].float().to(self.device)
    next_states = sample[5].float().to(self.device)
    done_flags = sample[6].float().to(self.device)
    probs = probs.to(self.device)

    # ==========================
    # Calculate Deep Q reward.
    if self.double:
      expected_Q = self.local_model(states)
      # print(expected_Q)
      argmax_indices = torch.argmax(expected_Q, dim = 1).view(-1, 1)
      expected_Q = expected_Q.gather(1, actions)
      # print(actions)
      # print(indices)
      # print('+================')
      # max_Q = self.target_model(next_states).max(1)[0].unsqueeze(1)
      # print(max_Q)
      max_Q = self.target_model(next_states).gather(1, argmax_indices)
      # print(max_Q)
      # print(max_Q)
      max_Q = rewards + (self.gamma * max_Q * (1 - done_flags))
    else:
      expected_Q = self.local_model(states)
      expected_Q = expected_Q.gather(1, actions)
      max_Q = self.target_model(next_states).max(1)[0].unsqueeze(1)
      max_Q = rewards + (self.gamma * max_Q * (1 - done_flags))

    # ==========================
    # Prioritized Experience Replay - Update priorities of samples
    # using TD error as a measure of informativity.
    TD_error = torch.abs((max_Q - expected_Q)).tolist()
    self.replay_buffer.update_priority(indices, TD_error)

    # ==========================
    # Prioritized Experience Replay - Importance Sampling
    probs = probs.view(-1, 1)
    expected_Q *= (probs.shape[0] * probs)**(-1 * self.beta)
    max_Q *= (probs.shape[0] * probs)**(-1 * self.beta)

    # ==========================
    # Network optimization
    loss = F.smooth_l1_loss(expected_Q, max_Q)
    self.optimizer.zero_grad()
    loss.backward()

    # ==========================
    # Unsure what effect this has
    for param in self.local_model.parameters():
        param.grad.data.clamp_(-1, 1)

    self.optimizer.step()

  def update_target_network(self):
    aggregate_params = self.target_model.state_dict()
    for params in aggregate_params:
      if 'num_batches_tracked' in params:
        aggregate_params[params] = self.local_model.state_dict()[params]
      else:
        aggregate_params[params] *= (1.0 - self.alpha)
        aggregate_params[params] += (self.alpha * self.local_model.state_dict()[params])
