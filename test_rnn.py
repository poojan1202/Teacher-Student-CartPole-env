import gym
import torch
import numpy as np
import torch.nn as nn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = gym.make('CartPole-v1')

#vectorised env for teacher network
#Same as teacher_cleanRL.py
class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
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
        #if args.capture_video:
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
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,  envs.action_space.n), std=0.01),
        )

    def get_action(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_value(self, x):
        return self.critic(x)

#Student Architecture
class StudentAgent(nn.Module):
    def __init__(self, envs):
        super(StudentAgent, self).__init__()
        inp = 3
        #self.layer1 = nn.Linear(np.array(inp).prod(),32)
        self.layer1 = nn.LSTM(input_size=inp,hidden_size=32,batch_first=True)
        self.layer2 = nn.Linear(32,  envs.action_space.n)
        self.hidden = None

    def forward(self,x,hidden):

        #out = F.relu(self.layer1(x))
        out, hidden = self.layer1(x, hidden)
        out = out.reshape(-1,32)
        #self.hidden = hidden
        out = F.softmax(self.layer2(out))
        return out, hidden


envs = VecPyTorch(DummyVecEnv([make_env('CartPole-v1', 1+i, i) for i in range(1)]), device)


#Choosing action from student_model
def choose_action(given_state,prev_action):
    with torch.no_grad():
        #if prev_action is None:
        #    action = envs.action_space.sample()
        #else:
        #    action = prev_action
        given_state = torch.Tensor(given_state)
        rnn_state = torch.cat((given_state,torch.tensor([prev_action])))
        #print(given_state.size(),torch.tensor([prev_action]).size())
        rnn_state  =rnn_state.resize(1,1,3)
        act_val, hidden = student_model(rnn_state,None)
        action = int(torch.argmax(act_val))
        return action

student_model = StudentAgent(envs)
student_model.load_state_dict(torch.load('runs/student_model_rnn_5000_1804.pth'))
teacher_model = Agent(envs).to(device)
teacher_model.load_state_dict(torch.load('runs/teacher_model.pth'))





ep = 10
loss_list = []
for i in range(ep):
    step=0
    st = env.reset()
    st_h = st[::2]
    done=False
    reward = 0
    episode_loss = 0
    action = envs.action_space.sample()

    while not done:
        env.render()
        step += 1
        action_new = choose_action(st_h, action)

        st_new, rew, done, info = env.step(action_new)
        st = st_new
        st_h = st_new[::2]
        action = action_new
        #print(reward)
        if done:
            loss_list.append(episode_loss)
            print('ep: ', i)
            print('steps: ',step)
