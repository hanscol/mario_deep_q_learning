import torch
import random





class transition():
    def __init__(self, state, action, reward, next_state, terminal, gamma, next_q):
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

    def add(self, trans):
        self.replay.append(trans)
        if len(self.replay) > self.capacity:
            self.replay.pop(0)

    def get_batch(self):
        img_size = self.replay[0].state.shape
        batch = torch.zeros([self.batch_size, img_size[0], img_size[1], img_size[2]], dtype=torch.int32)
        actions = torch.zeros([self.batch_size, 1])
        targets = torch.zeros([self.batch_size, 1])

        for i in range(self.batch_size):
            idx = random.randint(0,len(self.replay)-1)
            trans = self.replay[idx]
            batch[i,:,:,:] = trans.state
            actions[i,:] = trans.action
            targets[i,:] = trans.target

        return batch, actions, targets


