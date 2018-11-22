import torch
import random
import pickle
import os

class transition():
    def __init__(self, state, action, reward, next_q, terminal, gamma):
        self.state = state
        self.action = action

        if terminal:
            self.target = reward
        else:
            self.target = reward + gamma*next_q


class dataset():
    def __init__(self, capacity, batch_size, replay_dir, start_id, size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.replay_dir = replay_dir
        self.last_id = start_id
        self.first_id = start_id
        self.size = size
        self.max_alert = True

    def add(self, trans):
        with open('{}trans_{}'.format(self.replay_dir, self.last_id), 'wb') as f:
            pickle.dump(trans, f)

        if self.last_id - self.first_id > self.capacity:
            if self.max_alert:
                print('Max Capacity Reached')
                self.max_alert = False
            os.remove('{}trans_{}'.format(self.replay_dir, self.first_id))
            self.first_id += 1

        self.last_id +=1

    def get_batch(self):
        states = torch.zeros([self.batch_size, self.size[0], self.size[1], self.size[2]], dtype=torch.float32)
        actions = torch.zeros([self.batch_size, 1], dtype = torch.long)
        targets = torch.zeros([self.batch_size, 1])

        for i in range(self.batch_size):
            idx = random.randint(self.first_id, self.last_id - 1)

            with open('{}trans_{}'.format(self.replay_dir, idx), 'rb') as f:
                trans = pickle.load(f)

            trans.state = trans.state.expand(1, -1, -1)
            for j in range(idx-1, idx-4, -1):
                prev_idx = j
                if j < self.first_id:
                    prev_idx = self.first_id

                with open('{}trans_{}'.format(self.replay_dir, prev_idx), 'rb') as f:
                    prev_trans = pickle.load(f)

                prev_trans.state = prev_trans.state.expand(1, -1, -1)
                trans.state = torch.cat((prev_trans.state, trans.state))

            states[i,:,:,:] = trans.state
            actions[i,:] = trans.action
            targets[i,:] = trans.target

        batch = {'states': states, 'actions': actions, 'targets': targets}

        return batch


