from __future__ import print_function, division
import torch
from torchvision import transforms
from models import *
import numpy as np
from skimage import transform, color
import matplotlib.pyplot as plt
import random

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import time
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from data import *


def preprocess(x, size, final_height):
    x = color.rgb2gray(x)
    x = transform.resize(x, size, mode='constant', anti_aliasing=True)
    x = x[size[0]-final_height:, :]

    x = np.expand_dims(x, axis=2)
    x = transforms.ToTensor()(x)
    x = x.type(torch.float32)
    return x


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

        return max_q, max_a


def QLoss(output, targets):
    return torch.mean((targets - output)**2)


def train(model, device, optimizer, batch):
    model.train()
    optimizer.zero_grad()

    batch['states'] = batch['states'].to(device)
    output = model(batch['states'])
    output_at_action = torch.zeros([output.shape[0], 1]).to(device)

    for i in range(output.shape[0]):
        output_at_action[i,0] = output[i,batch['actions'][i,:]]

    batch['targets'] = batch['targets'].to(device)


    loss = QLoss(output_at_action, batch['targets']).to(device)

    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    movement = SIMPLE_MOVEMENT
    movement.append(['left', 'A'])
    movement.append(['left', 'B'])
    movement.append(['left', 'A', 'B'])
    movement.append(['B'])
    movement.append(['down'])
    movement.append(['up'])

    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, movement)

    #channels is acting as the number of frames in history
    #if resize_height and height are different, assert final_height < resize_height and image will be cropped
    channels = 4
    width = 84
    resize_height = 110
    final_height = 84

    batch_size = 32
    replay_capacity = 6000
    epsilon = 1.0
    gamma = 0.90

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")

    model = simple_net(channels, len(movement), device).to(device)
    model_file = 'mario_agent'
    model.load_state_dict(torch.load(model_file))

    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)


    max_steps = 5000
    num_eps = 4000

    data = dataset(replay_capacity, batch_size)

    for episode in range(num_eps):
        print('Episode {}'.format(episode+1))
        state = env.reset()
        state = preprocess(state, [resize_height, width], final_height)
        state = torch.cat((state, state, state, state))
        action = 0

        for step in range(max_steps):
            if step % 3 == 0:
                if random.random() < epsilon:
                    action = random.randint(0,len(movement)-1)
                else:
                    q_val, action = maxQ(state, model, device)

            next_state, reward, done, info = env.step(int(action))
            if reward != 0:
                reward = reward/abs(reward)

            next_state = preprocess(next_state, [resize_height, width], final_height)
            next_state = torch.cat((state[1:,:,:], next_state))
            next_q, next_a = maxQ(next_state, model, device)

            trans = transition(state, action, reward, next_q, done, gamma)
            data.add(trans)

            #learn something
            train(model, device, optimizer, data.get_batch())
            #print(reward)

            state = next_state

            env.render()
            #time.sleep(0.03)

            if done:
                break

        epsilon -= (1 / num_eps)
        if episode % 10 == 0:
            with open(model_file, 'wb') as f:
                torch.save(model.state_dict(), f)

    env.close()



if __name__ == '__main__':
    main()