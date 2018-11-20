from __future__ import print_function, division
import torch
from torchvision import transforms
from models import *
import numpy as np
from skimage import transform, color
import matplotlib.pyplot as plt

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import time
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

def process_numpy(x, size, height):
    x = color.rgb2gray(x)
    x = transform.resize(x, size, mode='constant', anti_aliasing=True)
    x = x[size[0]-height:, :]

    plt.imshow(x, cmap='gray')
    plt.show()

    x = np.expand_dims(x, axis=2)
    x = transforms.ToTensor()(x)
    return x

def main():
    custom_movement = SIMPLE_MOVEMENT
    custom_movement.append(['left', 'A'])
    custom_movement.append(['left', 'B'])
    custom_movement.append(['left', 'A', 'B'])
    custom_movement.append(['B'])
    custom_movement.append(['down'])
    custom_movement.append(['up'])

    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, custom_movement)

    #channels is acting as the number of frames in history
    channels = 4
    width = 84
    resize_height = 110
    height = 84
    batch_size = 32
    exp_size = 50
    epsilon = 0.10

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")

    model = simple_net(channels, len(custom_movement)).to(device)

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    max_steps = 500
    num_eps = 1
    for episode in range(num_eps):

        state = env.reset()
        for step in range(max_steps):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            state = process_numpy(state, [resize_height, width], height)
            env.render()
            time.sleep(0.03)

            if done:
                step = max_steps
    env.close()



if __name__ == '__main__':
    main()