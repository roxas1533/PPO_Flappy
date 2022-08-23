from gym.envs.registration import register

from . import envs

register(id="Flappy-v0", entry_point="flappy_gym_env.envs.Flappy:FlappyClass")
