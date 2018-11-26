from __future__ import print_function, division
from torchvision import transforms
from DDDQN_Color.models import *
import numpy as np
from skimage import transform, color
import matplotlib.pyplot as plt

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from DDDQN_Color.data import *


def preprocess(x, size, final_height, bottom_chop):
    #x = color.rgb2gray(x)
    x = transform.resize(x, size, mode='constant', anti_aliasing=True)
    x = x[(size[0]-final_height-bottom_chop):size[0]-bottom_chop, :, :]

    # plt.imshow(x)
    # plt.show()

    #x = np.expand_dims(x, axis=2)
    x = transforms.ToTensor()(x)
    x = x.type(torch.float32)
    return x


# def QLoss(output, targets):
#     return torch.mean((targets - output)**2)


def train(model, device, optimizer, batch):
    model.train()
    optimizer.zero_grad()

    batch['states'] = batch['states'].to(device)
    output = model(batch['states'])
    output_at_action = torch.zeros([output.shape[0], 1]).to(device)

    for i in range(output.shape[0]):
        output_at_action[i,0] = output[i,batch['actions'][i,:]]

    batch['targets'] = batch['targets'].to(device)


    loss = torch.nn.SmoothL1Loss()(output_at_action, batch['targets']).to(device)

    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    movement = SIMPLE_MOVEMENT
    movement.append(['left', 'A'])
    movement.append(['left', 'B'])
    movement.append(['left', 'A', 'B'])
    #movement.append(['B'])
    #movement.append(['down'])
    #movement.append(['up'])

    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, movement)

    #channels is acting as the number of frames in history
    #if resize_height and height are different, assert final_height < resize_height and image will be cropped
    channels = 3
    frames = 4
    width = 128
    resize_height = 180
    final_height = 128
    bottom_chop= 15
    size = [channels*frames, final_height, width]

    batch_size = 16
    replay_capacity = 100000
    replay_dir = '/home-local/bayrakrg/mario_replay/'
    start_epsilon = 1.0
    stop_epsilon = 0.01
    epsilon_decay = 0.00005
    gamma = 0.75

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")

    model = simple_net(channels, len(movement), device).to(device)
    target_model = simple_net(channels, len(movement), device).to(device)

    model_file = 'mario_agent'
    model.load_state_dict(torch.load(model_file))
    target_model.load_state_dict(torch.load(model_file))

    lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_reward_file ='total_reward.txt'
    with open(total_reward_file, 'w') as f:
        f.write('Reward\tSteps\n')



    max_steps = 500
    num_eps = 10000

    data = dataset(replay_capacity, batch_size, replay_dir, 1, size)

    tau = 0
    max_tau = 10000
    decay_step = 0

    for episode in range(num_eps):
        print('Episode {}'.format(episode+1))
        state = env.reset()
        state = preprocess(state, [resize_height, width, 3], final_height, bottom_chop)
        state = torch.cat((state, state, state, state))
        action=0
        episode_reward = 0

        for step in range(max_steps):
            tau += 1
            decay_step += 1

            epsilon = stop_epsilon + (start_epsilon - stop_epsilon)*np.exp(-epsilon_decay*decay_step)

            if random.random() < epsilon:
                action = random.randint(0,len(movement)-1)
            else:
                q_val, action, q_vals = maxQ(state, model, device)

            next_state, reward, done, info = env.step(int(action))

            if step == max_steps-1:
                reward -= 10

            if reward > 0:
                reward = 1
            else:
                reward = -1

            episode_reward += reward

            next_state = preprocess(next_state, [resize_height, width, 3], final_height, bottom_chop)
            next_state = torch.cat((state[3:, :, :], next_state))

            trans = transition(state, action, reward, next_state, done)
            data.add(trans)
            train(model, device, optimizer, data.get_batch(model, target_model, device, gamma))

            state = next_state

            env.render()

            if tau > max_tau:
                target_model.load_state_dict(model.state_dict())
                tau = 0

            if done:
                break

        with open(total_reward_file, 'a') as f:
            f.write('{}\t{}\n'.format(episode_reward, step))

        if episode % 5 == 0:
            with open(model_file, 'wb') as f:
                torch.save(model.state_dict(), f)


    env.close()




if __name__ == '__main__':
    main()