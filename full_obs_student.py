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
        self.layer1 = nn.Linear(np.array(envs.observation_space.shape).prod(),64)
        self.layer2 = nn.Linear(64,32)
        self.layer3 = nn.Linear(32,  envs.action_space.n)

    def forward(self,x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = F.softmax(self.layer3(out))
        return out


envs = VecPyTorch(DummyVecEnv([make_env('CartPole-v1', 1+i, i) for i in range(1)]), device)


model = Agent(envs).to(device)
model.load_state_dict(torch.load('runs/models/teacher_model.pth'))


student_model = StudentAgent(envs)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)

#Choosing action from student_model
def choose_action(given_state):
    with torch.no_grad():
        given_state = torch.Tensor(given_state)
        act_val = student_model(given_state)
        action = int(torch.argmax(act_val))
        return action

# Replay Buffer
class ReplayMemory(): # Stores [[state]]

    def __init__(self, size):
        self.size = size
        self.memory = [[]]

    def store(self, data):
        """Saves a transition."""
        for idx, part in enumerate(data):
            #print("Col {} appended {}".format(idx, point))
            self.memory[idx].append(part)

    def pop(self):
        for idx in range(1):
            self.memory[idx].pop(0)

    def remove(self):
        for idx in range(1):
            self.memory[idx].clear()

    def sample(self, batch_size):
        rows = random.sample(range(0, len(self.memory[0])), batch_size)
        experiences = [[]]
        for row in rows:
            for col in range(1):
                experiences[col].append(self.memory[col][row])
        return experiences

    def __len__(self):
        return len(self.memory[0])

memory = ReplayMemory(50000)
BATCH_SIZE = 128


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return 0
    experiences = memory.sample(BATCH_SIZE)
    states = torch.tensor(experiences[0])

    student_action_batch = student_model(states)
    student_action_batch = torch.Tensor(student_action_batch)#.unsqueeze(1)
    student_action_batch = student_action_batch.resize(2,128)

    with torch.no_grad():
        teacher_action_batch,x,y = model.get_action(states)

    teacher_act_vec = torch.zeros((2, BATCH_SIZE))

    for i in range(BATCH_SIZE):
        teacher_act_vec[teacher_action_batch[i].item()][i] = 1

    loss = criterion(student_action_batch,teacher_act_vec)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

ep = 2500
loss_list = []
step_loss = []
reward_list = []
for i in range(ep):
    step=0
    st = env.reset()
    done=False
    reward = 0
    episode_loss = 0

    while not done:
        step+=1
        action = choose_action(st)
        memory.store([st])
        if (i+1)%500==0:
            memory.remove()
        st,rew,done,info = env.step(action)

        loss = float(optimize_model())
        if loss>100:
            step_loss.append(loss)
        episode_loss+=loss
        if len(memory)>50000:
            memory.pop()
        #env.render()
        reward +=rew
        #print(reward)
        if done:
            loss_list.append(episode_loss)
            reward_list.append(reward)
            print('ep: ', i)
            print('steps: ',step)
            print('episode_loss: ',episode_loss)


PATH = 'runs/student_model.pth'
torch.save(student_model.state_dict(), PATH)
x1 = np.arange(len(loss_list))
fig1,ax1 = plt.subplots()
ax1.plot(x1, loss_list)
ax1.set_xlabel('episodes')
ax1.set_ylabel('episodic loss')
ax1.set_title('Loss per Episode')
x2 = np.arange(len(reward_list))
fig2,ax2 = plt.subplots()
ax2.plot(x2, reward_list)
ax2.set_xlabel('episodes')
ax2.set_ylabel('rewards')
ax2.set_title('Reward per Episode')
x3 = np.arange(len(step_loss))
fig3,ax3 = plt.subplots()
ax3.plot(x3, step_loss)
ax3.set_xlabel('steps')
ax3.set_ylabel('loss')
ax3.set_title('Loss per step')
plt.show()
