import torch
import random
import pickle
import os


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

    def get_batch(self, model, device, gamma):
        states = torch.zeros([self.batch_size, self.size[0], self.size[1], self.size[2]], dtype=torch.float32)
        actions = torch.zeros([self.batch_size, 1], dtype = torch.long)
        targets = torch.zeros([self.batch_size, 1])

        for i in range(self.batch_size):
            idx = random.randint(self.first_id, self.last_id - 1)

            with open('{}trans_{}'.format(self.replay_dir, idx), 'rb') as f:
                trans = pickle.load(f)

            next_q, n_a, n_vals = maxQ(trans.next_state, model, device)

            if trans.terminal:
                target = trans.reward
            else:
                target = trans.reward + gamma*next_q

            states[i,:,:,:] = trans.state
            actions[i,:] = trans.action
            targets[i,:] = target

        batch = {'states': states, 'actions': actions, 'targets': targets}

        return batch


