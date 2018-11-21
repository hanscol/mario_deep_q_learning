import torch
import random


class transition():
    def __init__(self, state, action, reward, next_q, terminal, gamma):
        self.state = state
        self.action = action

        if terminal:
            self.target = reward
        else:
            self.target = reward + gamma*next_q


class dataset():
    def __init__(self, capacity, batch_size):
        self.replay = []
        self.capacity = capacity
        self.batch_size = batch_size
        self.max_alert = True

    def add(self, trans):
        self.replay.append(trans)
        if len(self.replay) > self.capacity:
            if self.max_alert:
                print('Max Capacity Reached')
                self.max_alert = False
            self.replay.pop(0)

    def get_batch(self):
        img_size = self.replay[0].state.shape
        states = torch.zeros([self.batch_size, img_size[0], img_size[1], img_size[2]], dtype=torch.float32)
        actions = torch.zeros([self.batch_size, 1], dtype = torch.long)
        targets = torch.zeros([self.batch_size, 1])

        for i in range(self.batch_size):
            idx = random.randint(0,len(self.replay)-1)
            trans = self.replay[idx]
            states[i,:,:,:] = trans.state
            actions[i,:] = trans.action
            targets[i,:] = trans.target

        batch = {'states': states, 'actions': actions, 'targets': targets}

        return batch


