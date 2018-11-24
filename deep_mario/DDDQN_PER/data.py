import torch
import random
import pickle
import os
import numpy as np
from DDDQN_PER.per import Memory


def maxQ(state, model, device):
    model.eval()
    with torch.no_grad():
        state = state.expand(1, -1, -1, -1)
        state = state.to(device)
        q_vals = model(state)
        q_vals = torch.Tensor.cpu(q_vals)
        q_vals = q_vals.squeeze()
        max_q = torch.max(q_vals)
        max_a = torch.argmax(q_vals)
        max_a = max_a.type(torch.long)

        return max_q, max_a, q_vals

class transition():
    def __init__(self, state, action, reward, next_state, terminal):
        self.state = state
        self.action = action
        self.reward= reward
        self.next_state = next_state
        self.terminal = terminal

class dataset():
    def __init__(self, capacity, batch_size, replay_dir, size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.replay_dir = replay_dir
        self.cur_id = 0
        self.size = size
        self.max_alert = True

        self.per = Memory(capacity)


    def add(self, trans):
        file_name = '{}trans_{}'.format(self.replay_dir, self.cur_id)
        with open(file_name, 'wb') as f:
            pickle.dump(trans, f)

        self.cur_id += 1

        if self.cur_id == self.capacity:
            if self.max_alert:
                print('Max Capacity Reached')
                self.max_alert = False
            self.cur_id = 0

        self.per.store(file_name)

    def get_batch(self, model, target_model, device, gamma):
        states = torch.zeros([self.batch_size, self.size[0], self.size[1], self.size[2]], dtype=torch.float32)
        actions = torch.zeros([self.batch_size, 1], dtype = torch.long)
        targets = torch.zeros([self.batch_size, 1])

        idx, files, weights = self.per.sample(self.batch_size)
        weights = torch.from_numpy(weights)

        for i in range(self.batch_size):

            with open(files[i][0], 'rb') as f:
                trans = pickle.load(f)

            n_q, n_a, n_vals = maxQ(trans.next_state, model, device)
            t_q, t_a, t_vals = maxQ(trans.next_state, target_model, device)
            next_q = t_vals[n_a]

            if trans.terminal:
                target = trans.reward
            else:
                target = trans.reward + gamma*next_q

            states[i,:,:,:] = trans.state
            actions[i,:] = trans.action
            targets[i,:] = target

        batch = {'states': states, 'actions': actions, 'targets': targets, 'idx': idx, 'weights': weights}

        return batch

    def update_batch(self, idx, abs_errors):
        self.per.batch_update(idx, abs_errors)


