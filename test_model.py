import gym
import torch
import numpy as np
import torch.nn as nn
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v1")


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super().__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


def make_env(gym_id, seed, idx):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # if args.capture_video:
        #    if idx == 0:
        #        env = Monitor(env, f'videos/{experiment_name}')
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class StudentAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.layer1 = nn.Linear(np.array(envs.observation_space.shape).prod(), 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, envs.action_space.n)

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = F.softmax(self.layer3(out))
        return out


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.action_space.n), std=0.01),
        )

    def get_action(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_value(self, x):
        return self.critic(x)


envs = VecPyTorch(DummyVecEnv([make_env("CartPole-v1", 1 + i, i) for i in range(1)]), device)

student_model = StudentAgent(envs)
student_model.load_state_dict(torch.load("runs/student_model.pth"))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)
teacher_model = Agent(envs).to(device)
teacher_model.load_state_dict(torch.load("runs/teacher_model.pth"))


def choose_action(given_state):
    global steps_done
    with torch.no_grad():
        given_state = torch.Tensor(given_state)
        act_val = student_model(given_state)
        action = int(torch.argmax(act_val))
        return action


ep = 10
loss_list = []
for i in range(ep):
    step = 0
    st = env.reset()
    done = False
    reward = 0
    episode_loss = 0

    while not done:
        env.render()
        step += 1
        action = choose_action(st)
        st, rew, done, info = env.step(action)
        reward += rew
        # print(reward)
        if done:
            loss_list.append(episode_loss)
            print("ep: ", i)
            print("steps: ", step)
