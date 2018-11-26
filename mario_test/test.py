from pynput.keyboard import Key, Listener
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

movement= SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, movement)
key_map = {"'d'":1, "'w'":5, "'a'":6 }
state = env.reset()

def on_press(key):
    key = str(key)
    print(key in key_map)
    if key in key_map:
        next_state, reward, done, info = env.step(int(key_map[key]))
    else:
        next_state, reward, done, info = env.step(int(0))
    env.render()

def on_release(key):
    print('{0} release'.format(
        key))
    if key == Key.esc:
        # Stop listener
        return False


# Collect events until released
with Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    print('hello')
    listener.join()
