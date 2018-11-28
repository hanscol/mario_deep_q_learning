from __future__ import print_function, division
from torchvision import transforms
from DDDQN_PER.models import *
import numpy as np
from skimage import transform, color
import matplotlib.pyplot as plt

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from DDDQN_PER.data import *


def preprocess(x, size, final_height):
    x = color.rgb2gray(x)
    x = transform.resize(x, size, mode='constant', anti_aliasing=True)
    x = x[size[0]-final_height:, :]

    # plt.imshow(x, cmap='gray')
    # plt.show()

    x = np.expand_dims(x, axis=2)
    x = transforms.ToTensor()(x)
    x = x.type(torch.float32)
    return x


def QLoss(output, targets, weights):
    diff = (targets - output)
    abs_err = abs(diff)

    loss = torch.mean(weights * diff**2)

    return loss, abs_err



def train(model, device, optimizer, batch):
    model.train()
    optimizer.zero_grad()

    batch['states'] = batch['states'].to(device)
    output = model(batch['states'])
    output_at_action = torch.zeros([output.shape[0], 1]).to(device)

    for i in range(output.shape[0]):
        output_at_action[i,0] = output[i,batch['actions'][i,:]]

    batch['targets'] = batch['targets'].to(device)
    batch['weights'] = batch['weights'].to(device)


    loss, abs_err = QLoss(output_at_action, batch['targets'], batch['weights'])

    loss = loss.to(device)

    loss.backward()
    optimizer.step()

    return loss.item(), abs_err.detach().cpu()


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
    channels = 4
    # width = 84
    # resize_height = 110
    # final_height = 84
    width=128
    resize_height = 168
    final_height = 128
    size = [channels, final_height, width]

    batch_size = 16
    replay_capacity = 100000
    replay_dir = '/home/hansencb/mario_replay/'

    gamma = 0.95

    start_epsilon = 0.3
    stop_epsilon = 0.01
    epsilon_decay = 0.00025

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")

    model = simple_net(channels, len(movement), device).to(device)
    target_model = simple_net(channels, len(movement), device).to(device)

    data_file = 'data_loader'
    model_file = 'mario_agent'
    continue_train = True
    model.load_state_dict(torch.load(model_file))

    if continue_train:
        target_model.load_state_dict(torch.load(model_file))

    lr = 0.00005
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_reward_file ='total_reward.txt'


    if not continue_train:
        with open(total_reward_file, 'w') as f:
            f.write('Reward\tSteps\n')


    max_steps = 5000
    num_eps = 5000

    if continue_train:
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            data.batch_size = batch_size
    else:
        data = dataset(replay_capacity, batch_size, replay_dir, size)

        #initialize memory with 100 experiences
        done = True
        for i in range(100):
            if done:
                state = env.reset()
                state = preprocess(state, [resize_height, width], final_height)
                state = torch.cat((state, state, state, state))

            action = random.randint(0,len(movement)-1)
            next_state, reward, done, info = env.step(int(action))

            # if reward>0:
            #     reward = 1
            # else:
            #     reward = -1
            reward /= 15
            if reward == 0:
                reward = -0.1

            next_state = preprocess(next_state, [resize_height, width], final_height)
            next_state = torch.cat((state[1:, :, :], next_state))

            trans = transition(state, action, reward, next_state, done)
            data.add(trans)

            state = next_state


    tau = 0
    max_tau = 2000
    decay_step = 0
    farthest = 3000
    cur_x = 1

    #training loop
    for episode in range(num_eps):
        print('Episode {}'.format(episode+1))
        state = env.reset()
        state = preprocess(state, [resize_height, width], final_height)
        state = torch.cat((state, state, state, state))
        action = 0

        episode_reward = 0

        for step in range(max_steps):
            tau += 1


            #epsilon = stop_epsilon+(start_epsilon - stop_epsilon)*np.exp(-epsilon_decay*decay_step)
            epsilon = start_epsilon * np.exp(1-(1/(cur_x/farthest)))
            if epsilon < stop_epsilon:
                epsilon = stop_epsilon

            if random.random() < epsilon:
                action = random.randint(0,len(movement)-1)
            else:
                q_val, action, q_vals = maxQ(state, model, device)

            next_state, reward, done, info = env.step(int(action))

            cur_x = info['x_pos']

            if cur_x > farthest:
                farthest = cur_x

            # if reward > 0:
            #     reward = 1
            # else:
            #     reward = -1

            reward /= 15
            if reward == 0:
                reward = -0.1

            episode_reward += reward

            next_state = preprocess(next_state, [resize_height, width], final_height)
            next_state = torch.cat((state[1:,:,:], next_state))

            trans = transition(state, action, reward, next_state, done)
            data.add(trans)
            batch = data.get_batch(model, target_model, device, gamma)
            loss, abs_err = train(model, device, optimizer, batch)

            data.update_batch(batch['idx'], np.squeeze(torch.Tensor.numpy(abs_err)))

            state = next_state

            env.render()
            #time.sleep(0.03)

            if tau > max_tau:
                target_model.load_state_dict(model.state_dict())
                tau = 0

            if done:
                break

        decay_step += step
        with open(total_reward_file, 'a') as f:
            f.write('{}\t{}\n'.format(episode_reward, step))

        if episode % 5 == 0:
            with open(model_file, 'wb') as f:
                torch.save(model.state_dict(), f)
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)


    env.close()




if __name__ == '__main__':
    main()