import numpy as np
import matplotlib.pyplot as plt

path = 'DDDQN_PER/total_reward.txt'

rewards = np.zeros([0])
steps = np.zeros([0])
with open(path, 'r') as f:
    lines = f.readlines()
    for i in range(1,len(lines)):
        reward, step = str.strip(lines[i]).split('\t')
        rewards = np.append(rewards, float(reward))
        steps = np.append(steps, int(step))


plt.plot(steps)
plt.show()

plt.plot(rewards)
plt.show()