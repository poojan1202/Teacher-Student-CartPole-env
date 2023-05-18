import gym
import torch
import numpy as np
import torch.nn as nn
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("Acrobot-v1")

# vectorised env for teacher network
# Same as teacher_cleanRL.py
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


# Student Architecture
class StudentAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.inp = 4 + 1
        self.hidden_space = 32
        # self.layer1 = nn.Linear(np.array(inp).prod(),32)
        self.layer1 = nn.Linear(self.inp, self.hidden_space)
        self.layer2 = nn.LSTM(input_size=self.hidden_space, hidden_size=self.hidden_space, batch_first=True)
        self.layer3 = nn.Linear(self.hidden_space, envs.action_space.n)

    def forward(self, x, h, c):
        out = F.relu(self.layer1(x))
        out, (new_h, new_c) = self.layer2(out, (h, c))
        out = out.reshape(-1, 32)
        out = F.softmax(self.layer3(out))
        return out, new_h, new_c

    def sample_action(self, state, prev_action, h, c):
        given_state = torch.Tensor(state)
        # print(given_state,prev_action)
        rnn_state = torch.cat((given_state, torch.tensor([prev_action])))
        # print(given_state.size(),torch.tensor([prev_action]).size())
        rnn_state = rnn_state.resize(1, 1, 5)
        act_val, h_new, c_new = self.forward(rnn_state, h, c)
        probs = Categorical(probs=act_val)
        action = probs.sample().item()
        # action = int(torch.argmax(act_val))
        return action, h_new, c_new

    def init_hidden_state(self, batch_size, training=None):

        # assert training is not None, "training step parameter should be dtermined"

        if training is True:
            return torch.zeros([1, batch_size, self.hidden_space]), torch.zeros([1, batch_size, self.hidden_space])
        else:
            return torch.zeros([1, 1, self.hidden_space]), torch.zeros([1, 1, self.hidden_space])


envs = VecPyTorch(DummyVecEnv([make_env("Acrobot-v1", 1 + i, i) for i in range(1)]), device)


student_model = StudentAgent(envs)
student_model.load_state_dict(torch.load("runs/student_model_rnn_acrobot.pth"))
teacher_model = Agent(envs).to(device)
teacher_model.load_state_dict(torch.load("runs/acrobot_teacher_model.pth"))


ep = 10
loss_list = []
for i in range(ep):
    step = 0
    st = env.reset()
    st_h = st[:4]
    done = False
    reward = 0
    episode_loss = 0
    action = envs.action_space.sample()
    h, c = student_model.init_hidden_state(batch_size=128, training=False)

    while not done:
        env.render()
        step += 1
        action_new, h, c = student_model.sample_action(st_h, action, h, c)
        # action_new, hidden_1 = choose_action(st_h, action,hidden_1)
        # action_new,_,_ = teacher_model.get_action(torch.tensor(st))
        # action_new = action_new.item()
        st_new, rew, done, info = env.step(action_new)
        time.sleep(0.125)
        st = st_new
        st_h = st_new[:4]
        action = action_new
        # print(reward)
        if done:
            loss_list.append(episode_loss)
            print("ep: ", i)
            print("steps: ", step)
