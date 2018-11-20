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
    replay_capacity = 50
    epsilon = 0.10
    gamma = 0.10

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")

    model = simple_net(channels, len(movement), device).to(device)

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    max_steps = 500
    num_eps = 10

    data = dataset(replay_capacity, batch_size)

    for episode in range(num_eps):

        state = env.reset()
        state = preprocess(state, [resize_height, width], final_height)
        state = torch.cat((state, state, state, state))

        for step in range(max_steps):
            if random.random() < epsilon:
                action = random.randint(0,len(movement)-1)
            else:
                q_val, action = maxQ(state, model, device)

            next_state, reward, done, info = env.step(int(action))
            next_state = preprocess(next_state, [resize_height, width], final_height)
            next_state = torch.cat((state[1:,:,:], next_state))
            next_q, next_a = maxQ(next_state, model, device)

            trans = transition(state, action, reward, next_state, done, gamma, next_q)
            data.add(trans)


            state = next_state

            env.render()
            time.sleep(0.03)

            if done:
                break

    env.close()



if __name__ == '__main__':
    main()