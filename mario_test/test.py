from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import time
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

#env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

custom_movement = SIMPLE_MOVEMENT
custom_movement.append(['left', 'A'])
custom_movement.append(['left', 'B'])
custom_movement.append(['left', 'A', 'B'])
custom_movement.append(['B'])
custom_movement.append(['down'])
custom_movement.append(['up'])

env = BinarySpaceToDiscreteSpaceEnv(env, custom_movement)


#actions defined by integers 0-6

done = True
for step in range(5000):
    if done:
        state = env.reset()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    print(reward)
    env.render()
    time.sleep(0.03)

env.close()
