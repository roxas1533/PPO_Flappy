import gym

import flappy_gym_env

flappy_gym_env.envs.config.render_display = True

env = gym.make("Flappy-v0")

obs = env.reset()
while True:
    _, _, done, info = env.step(0)
    if done:
        env.reset()
    if info["finish"]:
        break
