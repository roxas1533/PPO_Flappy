import argparse
import os
from multiprocessing import allow_connection_pickling

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

import flappy_gym_env


def train(load_path, save_path, train_type, lr):
    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    env = gym.make("Flappy-v0")
    env = Monitor(env, "./logs/log", allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=5, channels_order="last")
    flappy_gym_env.envs.config.train_type = train_type
    if load_path is None:
        model = PPO(
            "MlpPolicy" if train_type == "mlp" else "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log="flappy_tensorboard_mlp",
            learning_rate=lr,
        )
    else:
        model = PPO.load(
            load_path,
            env=env,
            verbose=1,
            tensorboard_log="flappy_tensorboard_mlp",
            learning_rate=lr,
        )
    model.learn(total_timesteps=100000 * 60)
    model.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", help="load model", default=None)
    parser.add_argument("--save", help="filename", default="myenv_ppo2")
    parser.add_argument("--train_type", help="train type cnn or mlp", default="mlp")
    parser.add_argument("--lr", type=float, default=2e-6)
    args = parser.parse_args()
    train(args.load, args.save, args.train_type, args.lr)
