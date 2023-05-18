import gym
import torch
import numpy as np
import torch.nn as nn
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

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

        if training is True:
            return torch.zeros([1, batch_size, self.hidden_space]), torch.zeros([1, batch_size, self.hidden_space])
        else:
            return torch.zeros([1, 1, self.hidden_space]), torch.zeros([1, 1, self.hidden_space])


# Replay Buffer
class ReplayMemory:  # Stores [[state],[hidden_state],[prev_action],[h],[c]]
    def __init__(self, size):
        self.size = size
        self.memory = [[], [], [], [], []]

    def store(self, data):
        """Saves a transition."""
        for idx, part in enumerate(data):
            # print("Col {} appended {}".format(idx, point))
            self.memory[idx].append(part)

    def pop(self):
        """Deletes the least recent transition stored in the buffer"""
        for idx in range(5):
            self.memory[idx].pop(0)

    def remove(self):
        for idx in range(5):
            self.memory[idx].clear()

    def sample(self, batch_size):
        """Create a random batch of BATCH_SIZE with sequence of 10 time-steps"""
        #
        rows = random.sample(range(10, len(self.memory[0])), batch_size)
        experiences = [[], [], [], [], []]
        for row in rows:
            hold = [[], [], [], [], []]
            start = row - 10
            for vals in range(start, row):
                for col in range(5):
                    if col > 2:
                        continue
                    hold[col].append(self.memory[col][vals])

            for col in range(5):
                if col == 0:
                    for i in range(10):
                        experiences[col].append(self.memory[col][row + i - 10])
                    continue
                if col > 2:
                    experiences[col].append(self.memory[col][row - 10])
                    continue
                experiences[col].append(hold[col])
        return experiences

    def __len__(self):
        return len(self.memory[1])


def optimize_model(memory, BATCH_SIZE, student_model, teacher_model, criterion, optimizer):
    # h_net, c_net = student_model.init_hidden_state(batch_size=BATCH_SIZE,training=True)

    if len(memory) < BATCH_SIZE + 10:
        return 0
    # Getting samples
    experiences = memory.sample(BATCH_SIZE)
    # Fully Observable states for Teacher Network
    states = torch.tensor(experiences[0])
    # Partial Observable states for Student Network
    hidden_states = torch.tensor(experiences[1])
    # Batch of previous actions from the samples
    prev_action = torch.tensor([experiences[2]])
    prev_action = prev_action.resize(128, 10, 1)
    # Concatenating actions and partial states to feed the student network
    rnn_state = torch.cat((hidden_states, prev_action), 2)

    # hidden_memory for the batch of sequence
    h_net = torch.stack(experiences[3])
    h_net = h_net.resize(1, 128, 32)
    # hidden_memory for the batch of sequence
    c_net = torch.stack(experiences[4])
    c_net = c_net.resize(1, 128, 32)
    # print(h_net)
    # return torch.zeros([1, batch_size, self.hidden_space]), torch.zeros([1, batch_size, self.hidden_space])

    student_action_batch, _, _ = student_model(rnn_state, h_net, c_net)
    student_action_batch = torch.Tensor(student_action_batch)  # .unsqueeze(1)

    with torch.no_grad():
        teacher_action_batch, x, y = teacher_model.get_action(states)

    # teacher_act_vec = torch.zeros((2, BATCH_SIZE*10))
    teacher_act_vec = torch.zeros((BATCH_SIZE * 10, 3))

    for i in range(BATCH_SIZE * 10):
        # teacher_act_vec[teacher_action_batch[i].item()][i] = 1
        teacher_act_vec[i][teacher_action_batch[i].item()] = 1

    loss = criterion(student_action_batch, teacher_act_vec)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    return float(loss)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("Acrobot-v1")

    envs = VecPyTorch(DummyVecEnv([make_env("Acrobot-v1", 1 + i, i) for i in range(1)]), device)

    teacher_model = Agent(envs).to(device)
    teacher_model.load_state_dict(torch.load("runs/acrobot_teacher_model.pth"))

    student_model = StudentAgent(envs)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)

    memory = ReplayMemory(50000)
    BATCH_SIZE = 128

    # Initialize buffer with teacher 50 ep
    b_ep = 50

    # Training loop
    ep = 1000
    loss_list = []
    step_loss = []
    reward_list = []

    for i in range(b_ep):
        st = env.reset()
        st_h = st[:4]
        done = False
        reward = 0
        episode_loss = 0
        action = envs.action_space.sample()
        # hidden = None
        # hidden_1 = None
        step = 0
        h, c = student_model.init_hidden_state(batch_size=BATCH_SIZE, training=False)

        while not done:
            step += 1
            action_new, h_n, c_n = student_model.sample_action(st_h, action, h, c)
            teacher_action = teacher_model.get_action(torch.tensor(st))
            st_new, rew, done, info = env.step(teacher_action[0].item())

            memory.store([st, st_h, action, h_n.detach(), c_n.detach()])
            st = st_new
            st_h = st_new[:4]
            action = action_new
            h = h_n
            c = c_n

            if done:
                print("ep: ", i)
                print("steps: ", step)
            #    print('episode_loss: ',episode_loss)

    for i in range(ep):
        step = 0
        st = env.reset()
        st_h = st[:4]
        done = False
        reward = 0
        episode_loss = 0
        action = envs.action_space.sample()
        # hidden = None
        h, c = student_model.init_hidden_state(batch_size=BATCH_SIZE, training=False)
        if (i + 1) % 50 == 0:
            PATH = f"runs/acrobot/student_model_rnn_acrobot_{i + 1}.pth"
            # print(PATH)
            torch.save(student_model.state_dict(), PATH)

        if i > 200 and (i + 1) % 200 == 0:
            memory.remove()

        while not done:
            step += 1
            action_new, h, c = student_model.sample_action(st_h, action, h, c)

            st_new, rew, done, info = env.step(action_new)

            memory.store([st, st_h, action, h.detach(), c.detach()])
            st = st_new
            st_h = st_new[:4]
            action = action_new
            loss = optimize_model(
                memory=memory,
                BATCH_SIZE=BATCH_SIZE,
                student_model=student_model,
                teacher_model=teacher_model,
                criterion=criterion,
                optimizer=optimizer,
            )
            if loss > 0:
                step_loss.append(loss)

            episode_loss += loss
            if len(memory) > 50000:
                memory.pop()
            # env.render()
            reward += rew
            # print(reward)
            if done:
                loss_list.append(episode_loss)
                reward_list.append(reward)
                print("ep: ", i)
                print("steps: ", step)
                print("episode_loss: ", episode_loss)

    PATH = "runs/student_model_rnn_acrobot.pth"
    torch.save(student_model.state_dict(), PATH)
    x1 = np.arange(len(loss_list))
    fig1, ax1 = plt.subplots()
    ax1.plot(x1, loss_list)
    ax1.set_xlabel("episodes")
    ax1.set_ylabel("episodic loss")
    ax1.set_title("Loss per Episode")
    x2 = np.arange(len(reward_list))
    fig2, ax2 = plt.subplots()
    ax2.plot(x2, reward_list)
    ax2.set_xlabel("episodes")
    ax2.set_ylabel("rewards")
    ax2.set_title("Reward per Episode")
    x3 = np.arange(len(step_loss))
    fig3, ax3 = plt.subplots()
    ax3.plot(x3, step_loss)
    ax3.set_xlabel("steps")
    ax3.set_ylabel("loss")
    ax3.set_title("Loss per step")
    plt.show()
