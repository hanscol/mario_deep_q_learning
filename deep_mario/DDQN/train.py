from __future__ import print_function, division
from torchvision import transforms
from DQN.models import *
import numpy as np
from skimage import transform, color

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from DQN.data import *


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
        max_a = max_a.type(torch.long)

        return max_q, max_a, q_vals


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
    size = [channels, final_height, width]

    batch_size = 32
    replay_capacity = 1000000
    replay_dir = '/home/hansencb/mario_replay/'
    epsilon = 1
    gamma = 0.9

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")

    model = simple_net(channels, len(movement), device).to(device)
    target_model = simple_net(channels, len(movement), device).to(device)

    model_file = 'mario_agent'
    model.load_state_dict(torch.load(model_file))
    target_model.load_state_dict(torch.load(model_file))

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_reward_file ='total_reward.txt'
    with open(total_reward_file, 'w') as f:
        f.write('Reward\tSteps\n')



    max_steps = 5000
    num_eps = 1000

    data = dataset(replay_capacity, batch_size, replay_dir, 1, size)


    for episode in range(num_eps):
        print('Episode {}'.format(episode+1))
        state = env.reset()
        state = preprocess(state, [resize_height, width], final_height)
        state = torch.cat((state, state, state, state))
        action = 0

        episode_reward = 0

        for step in range(max_steps):
            if step % 3 == 0:
                if random.random() < epsilon:
                    action = random.randint(0,len(movement)-1)
                else:
                    q_val, action, q_vals = maxQ(state, model, device)

            next_state, reward, done, info = env.step(int(action))

            if reward > 0:
                reward = 1
            else:
                reward = -1

            episode_reward += reward

            next_state = preprocess(next_state, [resize_height, width], final_height)
            next_state = torch.cat((state[1:,:,:], next_state))
            n_q, n_a, n_vals = maxQ(next_state, model, device)
            t_q, t_a, t_vals = maxQ(next_state, target_model, device)
            next_q = t_vals[n_a]

            trans = transition(state[3,:,:], action, reward, next_q, done, gamma)
            data.add(trans)
            train(model, device, optimizer, data.get_batch())

            state = next_state

            env.render()
            #time.sleep(0.03)

            if done:
                with open(total_reward_file, 'a') as f:
                    f.write('{}\t{}\n'.format(episode_reward, step))

                break

        epsilon -= (1 / num_eps)
        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

            with open(model_file, 'wb') as f:
                torch.save(model.state_dict(), f)


    env.close()




if __name__ == '__main__':
    main()