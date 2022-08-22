import gym
import numpy as np
import onnxruntime
import torch.onnx
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from torch import nn
from torch.distributions import Categorical
from torchinfo import summary

import flappy_gym_env

model = PPO.load("myenv_ppo2")
env = DummyVecEnv([lambda: gym.make("Flappy-v0")])
# env = MaxAndSkipEnv(env)
env = VecFrameStack(env, n_stack=5, channels_order="last")

obs = env.reset()
obs = np.transpose(obs, (0, 3, 1, 2))
obs = torch.tensor(obs).cuda()
model.policy.set_training_mode(False)
pro_obs = obs.float() / 255.0
# feature = model.policy.features_extractor(pro_obs)
# mean_actions = model.policy.action_net(feature)


class GetAction(nn.Module):
    def __init__(self, observation_space):
        super(GetAction, self).__init__()
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, 512), nn.ReLU())
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.linear(self.cnn(x))
        x = self.fc(x)
        dis = Categorical(logits=x)
        probs_2d = dis.probs.reshape(-1, dis._num_events)
        samples_2d = torch.multinomial(probs_2d, torch.Size().numel(), True)
        return samples_2d


# print(model.policy._predict(obs))
# model.predict(obs)
mm = GetAction(model.policy.observation_space).cuda()
model_state = model.policy.state_dict()
mm_state = mm.state_dict()
for i in range(0, 6, 2):
    mm_state[f"cnn.{i}.weight"] = model_state[f"features_extractor.cnn.{i}.weight"]
    mm_state[f"cnn.{i}.bias"] = model_state[f"features_extractor.cnn.{i}.bias"]
mm_state["linear.0.weight"] = model_state["features_extractor.linear.0.weight"]
mm_state["linear.0.bias"] = model_state["features_extractor.linear.0.bias"]
mm_state["fc.weight"] = model_state["action_net.weight"]
mm_state["fc.bias"] = model_state["action_net.bias"]
mm.load_state_dict(mm_state)
# print(mm.state_dict().keys())

torch.onnx.export(
    mm,
    pro_obs.detach(),
    "flappy.onnx",
    export_params=True,
    input_names=["input"],  # モデルへの入力変数名
    output_names=["output"],
)
# features_extractor = onnxruntime.InferenceSession("flappy_features.onnx")
# shared_net = onnxruntime.InferenceSession("flappy_shared_net.onnx")
# latent_pi = onnxruntime.InferenceSession("flappy_policy_net.onnx")
# action_net = onnxruntime.InferenceSession("flappy_latent_net.onnx")
# feature = features_extractor.run(None, {"input": pro_obs})
# shared = shared_net.run(None, {"input": feature[0]})
# latent_pi = latent_pi.run(None, {"input": shared[0]})
# mean_actions = action_net.run(None, {"input": latent_pi[0]})
# print(mean_actions[0])
