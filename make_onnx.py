import argparse

import gym
import numpy as np
import torch.onnx
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from torch import multinomial, nn
from torch.distributions import Categorical
from torch.distributions.utils import logits_to_probs
from torchinfo import summary

import flappy_gym_env


class GetAction(nn.Module):
    def __init__(self, observation_space, train_type="mlp"):
        super(GetAction, self).__init__()
        n_input_channels = observation_space.shape[0]
        if train_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(n_input_channels, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
            )
            self.feature_dim = 64
        elif train_type == "cnn":
            self.head = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.feature_dim = 512
        else:
            raise ValueError("Wrong train type")
        with torch.no_grad():
            n_flatten = self.head(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = (
            nn.Sequential(nn.Linear(n_flatten, self.feature_dim), nn.ReLU())
            if train_type == "cnn"
            else nn.Sequential()
        )
        self.fc = nn.Linear(self.feature_dim, 2)

    def forward(self, x):
        x = self.linear(self.head(x))
        x = self.fc(x)
        probs = self.get_prob(x)
        probs_2d = probs.reshape(-1, 2)
        return probs_2d

    def get_prob(self, x):
        vmax = torch.max(x, -1, keepdim=True).values
        out = torch.log(torch.sum(torch.exp(x - vmax), -1, keepdim=True)) + vmax
        logits = x - out
        return logits_to_probs(logits)


def make_onnx(model, train_type):
    model = PPO.load(model)
    flappy_gym_env.envs.config.train_type = train_type

    onnx_model = GetAction(model.observation_space, train_type).cuda()
    env = DummyVecEnv([lambda: gym.make("Flappy-v0")])
    env = VecFrameStack(env, n_stack=5, channels_order="last")
    obs = env.reset()
    # feature = model.policy.mlp_extractor.policy_net(torch.tensor(obs).cuda())
    # print(model.policy.state_dict().keys())
    model.policy.set_training_mode(False)

    if train_type == "cnn":
        obs = np.transpose(obs, (0, 3, 1, 2))
        obs = torch.tensor(obs).cuda()
        pro_obs = obs.float() / 255.0
    else:
        pro_obs = torch.tensor(obs).cuda()
    # feature = model.policy.features_extractor(pro_obs)
    # mean_actions = model.policy.action_net(feature)

    print(model.policy.state_dict().keys())
    # model.predict(obs)
    model_state = model.policy.state_dict()
    onnx_model_state = onnx_model.state_dict()
    if train_type == "cnn":
        for i in range(0, 6, 2):
            onnx_model_state[f"head.{i}.weight"] = model_state[
                f"features_extractor.cnn.{i}.weight"
            ]
            onnx_model_state[f"head.{i}.bias"] = model_state[
                f"features_extractor.cnn.{i}.bias"
            ]
        onnx_model_state["linear.0.weight"] = model_state[
            "features_extractor.linear.0.weight"
        ]
        onnx_model_state["linear.0.bias"] = model_state[
            "features_extractor.linear.0.bias"
        ]
    else:
        onnx_model_state["head.0.weight"] = model_state[
            "mlp_extractor.policy_net.0.weight"
        ]
        onnx_model_state["head.0.bias"] = model_state["mlp_extractor.policy_net.0.bias"]
        onnx_model_state["head.2.weight"] = model_state[
            "mlp_extractor.policy_net.2.weight"
        ]
        onnx_model_state["head.2.bias"] = model_state["mlp_extractor.policy_net.2.bias"]
    onnx_model_state["fc.weight"] = model_state["action_net.weight"]
    onnx_model_state["fc.bias"] = model_state["action_net.bias"]
    onnx_model.load_state_dict(onnx_model_state)
    torch.onnx.export(
        onnx_model.cuda(),
        pro_obs.detach(),
        f"flappy_{train_type}.onnx",
        opset_version=11,
        export_params=True,
        input_names=["input"],  # モデルへの入力変数名
        output_names=["output"],
    )
    # features_extractor = onnxruntime.InferenceSession("flappy_features.onnx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="load model", default=None)
    parser.add_argument("--train_type", help="train type cnn or mlp", default="mlp")
    args = parser.parse_args()
    make_onnx(args.model, args.train_type)
