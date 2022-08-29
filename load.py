import argparse

import gym
import numpy as np
import onnxruntime
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import flappy_gym_env


def predict(obs, onn, train_type):
    if train_type == "mlp":
        obs = torch.tensor(obs).cuda()
    else:
        obs = np.transpose(obs, (0, 3, 1, 2))
        obs = torch.tensor(obs).cuda()
        obs = obs.float() / 255.0
    pro_obs = obs.cpu().numpy()
    return onn.run(None, {"input": pro_obs})[0]


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
    # print(obs.shape)
    time = 0
    while True:
        # print(obs)

        if load_type == "onnx":
            action = predict(obs, model, train_type)
            if time > 110:
                print(obs, time)
            # action = [0] if np.random.rand() < action[0, 0] else [1]
        else:
            action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step([np.argmax(action)])
        time = time + 1
        if dones:
            obs = env.reset()
        if info[0]["finish"]:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", help="load model", default="myenv_ppo2")
    parser.add_argument("--load_type", help="zip or onnx", default="zip")
    parser.add_argument("--train_type", help="trained type cnn or mlp", default="mlp")
    args = parser.parse_args()
    load_model(args.load, args.train_type, args.load_type)
