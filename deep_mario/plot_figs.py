import numpy as np
import matplotlib.pyplot as plt

path = '/home/local/VANDERBILT/bayrakrg/mario_deep_q_learning/deep_mario/Results/reward/total_reward.txt'

rewards = np.zeros([0])
steps = np.zeros([0])
with open(path, 'r') as f:
    lines = f.readlines()
    for i in range(1,len(lines)):
        reward, step = str.strip(lines[i]).split('\t')
        rewards = np.append(rewards, float(reward))
        steps = np.append(steps, int(step))


fig, ax = plt.subplots(2, 1)
ax[0].plot(rewards, label='Color DDDQN')
ax[0].set_ylabel(' Total Reinforcement')

ax[1].plot(steps, label='Color DDDQN')
ax[1].set_ylabel('# of Steps')
ax[1].set_xlabel('# of Epoch')
