import sys
import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque, OrderedDict, Counter
import matplotlib.pyplot as plt
import copy
import time
import gc
import json
from deepQlearning import deep_Q_learning

env = gym.make('LunarLander-v2')
env.seed(random.randint(0, 9999))
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

EXPERIMENT_PATH = "./ExperimentSettings/Experiment" + str(sys.argv[1]) + ".json"

with open(EXPERIMENT_PATH) as json_file:
		settings = json.load(json_file)

agent_hyperparams = settings["agent_hyperparams"]

if agent_hyperparams["dueling"] == "True":
    model_structure = [settings["prime"],
                      settings["value"],
                      settings["advantage"]]
    dueling = True
else:
    model_structure = settings["model_structure"]
    dueling = False

if agent_hyperparams["double"] == "True":
    double = True
else:
    double = False

experiment_args = {"env": env,
                "num_episodes": 20,
                "max_steps": 1000,
                "policy_update_threshold": 4,
                "target_update_threshold": 16,
                "action_shape": env.action_space.n,
                "model_structure": model_structure,
                "agent_hyperparams": agent_hyperparams,
                "dueling": dueling,
                "double": double,
				"experiment_idx": str(sys.argv[1])}

gc.disable()

start = time.time()

scores, agent = deep_Q_learning(**experiment_args)

end = time.time()
print(end - start)
