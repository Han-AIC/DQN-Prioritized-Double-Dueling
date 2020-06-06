import numpy as np
import random
from collections import namedtuple, deque, OrderedDict, Counter
import copy
import csv

from agent import Agent

def deep_Q_learning(env,
                    num_episodes,
                    max_steps,
                    policy_update_threshold,
                    target_update_threshold,
                    action_shape,
                    model_structure,
                    agent_hyperparams,
                    dueling,
                    double,
                    experiment_idx):

  agent = Agent(action_shape,
              model_structure,
              agent_hyperparams,
              dueling,
              double)

  step_counter = 0

  scores = []
  scores_window = deque(maxlen=100)

  with open('./Reports/report' + experiment_idx + '.csv', 'w+', newline='') as csvfile:
      writer = csv.writer(csvfile, delimiter=' ',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
      for i in range(num_episodes):
        state = env.reset()
        score = 0
        policy_update_counter = 0
        target_update_counter = 0
        for j in range(max_steps):
          action = agent.choose_action(state)
          next_state, reward, done, _ = env.step(action)
          score += reward
          agent.update_buffer(state, action, reward, next_state, done)
          state = next_state

          step_counter += 1

          if step_counter % policy_update_threshold == 0:
            if step_counter > agent_hyperparams['batch_size']:
              policy_update_counter += 1
              agent.train_network()

          if step_counter % target_update_threshold == 0:
            if step_counter > agent_hyperparams['batch_size']:
              target_update_counter += 1
              agent.update_target_network()

          if done:
            break

        agent.decay_epsilon()

        scores.append(score)
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        # eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}, Network Updates: {:.2f}, Target Updates: {:.2f}'.format(i, np.mean(scores_window), policy_update_counter, target_update_counter), end="")
        if i % 20 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}, Network Updates: {:.2f}, Target Updates: {:.2f}'.format(i, np.mean(scores_window), policy_update_counter, target_update_counter))
            writer.writerow(['\rEpisode {}\tAverage Score: {:.2f}, Network Updates: {:.2f}, Target Updates: {:.2f}'.format(i, np.mean(scores_window), policy_update_counter, target_update_counter)])
        if np.mean(scores_window)>=220.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i-100, np.mean(scores_window)))
            writer.writerow(['\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i-100, np.mean(scores_window))])
            torch.save(agent.local_model(), './Solutions/experiment_' + experiment_idx + '.pth')

  return scores, agent
