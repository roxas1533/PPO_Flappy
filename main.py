from multiprocessing import allow_connection_pickling

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

import flappy_gym_env

n_cpu = 8
env = gym.make("Flappy-v0")
env = Monitor(env, "./logs/log", allow_early_resets=True)
# env = SubprocVecEnv([(lambda: env) for i in range(n_cpu)])
env = DummyVecEnv([lambda: env])
# env = MaxAndSkipEnv(env)
env = VecFrameStack(env, n_stack=5, channels_order="last")


# env = Flappy.FlappyClass()
# eval_env = DummyVecEnv([lambda: Flappy.FlappyClass()])
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="flappy_tensorboard")
model = PPO.load(
    "myenv_ppo2_back",
    env=env,
    verbose=1,
    tensorboard_log="flappy_tensorboard",
    learning_rate=2e-6,
)
model.learn(total_timesteps=100000 * 60)
model.save("myenv_ppo2")

for i in range(100):
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            break
