import argparse

import gym
import numpy as np
import onnxruntime
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import flappy_gym_env


def load_model(load_path, train_type, load_type):
    flappy_gym_env.envs.config.train_type = train_type
    flappy_gym_env.envs.config.render_display = True
    if load_type == "onnx":
        model = onnxruntime.InferenceSession(load_path)
    elif load_type == "zip":
        model = PPO.load(load_path)
    else:
        raise Exception("load_type not supported")
    env = DummyVecEnv([lambda: gym.make("Flappy-v0")])
    env = VecFrameStack(env, n_stack=5, channels_order="last")

    obs = env.reset()

    # oon = onnxruntime.InferenceSession("flappy.onnx")

    def predict(obs):
        global oon
        obs = np.transpose(obs, (0, 3, 1, 2))
        obs = torch.tensor(obs).cuda()
        pro_obs = obs.float() / 255.0
        pro_obs = pro_obs.cpu().numpy()
        return oon.run(None, {"input": pro_obs})[0]

    while True:
        # action = predict(obs)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        # action = predict(obs)
        if dones:
            obs = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", help="load model", default="myenv_ppo2")
    parser.add_argument("--load_type", help="zip or onnx", default="zip")
    parser.add_argument("--train_type", help="trained type cnn or mlp", default="mlp")
    args = parser.parse_args()
    load_model(args.load, args.train_type, args.load_type)
