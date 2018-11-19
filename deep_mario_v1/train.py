from __future__ import print_function, division
import torch
from models import *

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import time
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


def get_action(model, state, train=False):


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

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")

    model = resnet(13, 18, True).to(device)

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    max_steps = 500
    num_eps = 1
    for episode in range(num_eps):
        done = True
        init = False
        state = env.reset()
        for step in range(max_steps):
            if done:
                state = env.reset()

            action = get_action(model, state, train=True)



if __name__ == '__main__':
    main()