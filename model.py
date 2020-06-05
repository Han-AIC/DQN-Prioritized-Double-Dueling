import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque, OrderedDict, Counter
import copy


class Model(nn.Module):

  def __init__(self, structural_definition):
    super(Model, self).__init__()
    self.seed = torch.manual_seed(0)
    self.layers = []

    for layer in structural_definition:
        layer_params = structural_definition[layer]
        layer_type = self.parse_layer_type(layer_params['layer_type'])
        layer_size_mapping = layer_params['layer_size_mapping']
        activation = self.parse_activation(layer_params['activation'])

        setattr(self,
                layer,
                layer_type(**layer_size_mapping))

        self.layers.append((layer, activation))

  def parse_layer_type(self, layer_type):
      """ Detects layer type of a specified layer from configuration

      Args:
          layer_type (str): Layer type to initialise
      Returns:
          Layer definition (Function)
      """
      if layer_type == "linear":
          return nn.Linear
      elif layer_type == "batchnorm1d":
          return nn.BatchNorm1d
      elif layer_type == 'conv2d':
          return nn.Conv2d
      else:
          raise ValueError("Specified layer type is currently not supported!")

  def parse_activation(self, activation):
      """ Detects activation function specified from configuration

      Args:
          activation(str): Activation function to use
      Returns:
          Activation definition (Function)
      """
      if activation == "sigmoid":
          return torch.sigmoid
      elif activation == "relu":
          return torch.relu
      elif activation == "nil":
          return None
      else:
          raise ValueError("Specified activation is currently not supported!")

  def forward(self, x):
      for layer_activation_tuple in self.layers:
          current_layer =  getattr(self, layer_activation_tuple[0])
          if layer_activation_tuple[1] is None:
              x = current_layer(x)
          else:
              x = layer_activation_tuple[1](current_layer(x))
      return x

class DuelingQ(nn.Module):
  def __init__(self,
               prime_structure,
               value_structure,
               advantage_structure):
    super(DuelingQ, self).__init__()
    self.prime_model = Model(prime_structure)
    self.value_model = Model(value_structure)
    self.advantage_model = Model(advantage_structure)

  def forward(self, x):
    x = self.prime_model(x)
    value = self.value_model(x)
    advantage = self.advantage_model(x)
    Q = value + (advantage - advantage.mean())
    return Q
