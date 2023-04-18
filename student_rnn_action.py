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


model = Agent(envs).to(device)
model.load_state_dict(torch.load('runs/teacher_model.pth'))


student_model = StudentAgent(envs)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)

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

# Replay Buffer
class ReplayMemory(): # Stores [[state],[hidden_state],[prev_action]]

    def __init__(self, size):
        self.size = size
        self.memory = [[],[],[]]

    def store(self, data):
        """Saves a transition."""
        for idx, part in enumerate(data):
            #print("Col {} appended {}".format(idx, point))
            self.memory[idx].append(part)

    def pop(self):
        for idx in range(3):
            self.memory[idx].pop(0)

    def remove(self):
        for idx in range(3):
            self.memory[idx].clear()

    def sample(self, batch_size):
        rows = random.sample(range(10, len(self.memory[0])), batch_size)
        experiences = [[],[],[]]
        for row in rows:
            hold = [[],[],[]]
            start = row-10
            for vals in range(start,row):
                for col in range(3):
                    hold[col].append(self.memory[col][vals])

            for col in range(3):
                if col==0:
                    for i in (range(10)):
                        experiences[col].append(self.memory[col][row+i-10])
                    continue
                experiences[col].append(hold[col])
        return experiences

    def __len__(self):
        return len(self.memory[1])

memory = ReplayMemory(50000)
BATCH_SIZE = 128


def optimize_model(hidden):
    if str(type(hidden)) == "<class 'int'>":
        hidden = None
    else:
        hidden = hidden
    if len(memory) < BATCH_SIZE+10:
        return 0,0
    experiences = memory.sample(BATCH_SIZE)
    states = torch.tensor(experiences[0])
    hidden_states = torch.tensor(experiences[1])
    prev_action = torch.tensor([experiences[2]])
    prev_action = prev_action.resize(128,10,1)
    #prev_action = prev_action.resize(1,10,128)
    #print(hidden_states.size(),prev_action.size())
    rnn_state = torch.cat((hidden_states,prev_action),2)

    student_action_batch, hidden = student_model(rnn_state,hidden)
    hidden1 = hidden[0].detach()
    hidden2 = hidden[1].detach()
    hidden = (hidden1,hidden2)
    #print(hidden)
    student_action_batch = torch.Tensor(student_action_batch)#.unsqueeze(1)

    with torch.no_grad():
        teacher_action_batch,x,y = model.get_action(states)

    #teacher_act_vec = torch.zeros((2, BATCH_SIZE*10))
    teacher_act_vec = torch.zeros(( BATCH_SIZE * 10, 2))

    for i in range(BATCH_SIZE*10):
        #teacher_act_vec[teacher_action_batch[i].item()][i] = 1
        teacher_act_vec[i][teacher_action_batch[i].item()] = 1

    loss = criterion(student_action_batch,teacher_act_vec)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss), hidden

ep = 5000
loss_list = []
step_loss = []
reward_list = []
for i in range(ep):
    step=0
    st = env.reset()
    st_h = st[::2]
    done=False
    reward = 0
    episode_loss = 0
    action = envs.action_space.sample()
    hidden = None

    while not done:
        step+=1
        action_new = choose_action(st_h,action)

        if (i+1)%1000==0:
            memory.remove()
        st_new,rew,done,info = env.step(action_new)

        memory.store([st,st_h, action])
        st = st_new
        st_h = st_new[::2]
        action = action_new
        loss, hidden = optimize_model(hidden)
        #loss = float(loss)
        if loss>0:
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


PATH = 'runs/student_model_rnn_5000_1804.pth'
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
