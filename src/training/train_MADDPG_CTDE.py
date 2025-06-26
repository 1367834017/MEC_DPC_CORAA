#Training of MADDPG approach under the CTDE scheme
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as Fun
import torch.optim as optim
import collections
import time
import random
from partial_commu import DecPOSGPartialCommunication

class Env():
    def __init__(self, alpha, beta, B, N0, hi, pi, K, ser, Di, Ci, fi_m, fi_l):
        """
        B,N0：Bandwidth and the variance of Gaussian white noise（10MHz=10e6Hz， pow(10, -174 / 10) * 0.001）
        hi, pi: Channel gain, transmission power 0.001 * pow(np.random.uniform(50, 200, num), -3)、500mW=0.5W、100
        K, ser: Number of UDs, number of ESs
        Di, Ci: Task data size, the required number of CPU cycles (300~500kb) 1024kb=1Mb, (900, 1100)兆周期数 1Mhz = 1000khz = 1000*1000hz
        fi_m: Maximum computational capacity of the server 3-7 GHz/s 10e9Hz/s
        fi_l: Local computational capacity 800-1500 MHz
        state System observation
        """
        self.alpha, self.beta = alpha, beta
        self.B, self.N0, self.hi, self.pi, self.K, self.ser = B, N0, hi, pi, K, ser
        self.Di, self.Ci = Di, Ci
        self.fi_m, self.fi_l = fi_m, fi_l
        self.reward = np.zeros(self.K)
        self.done = []
        self.state = np.zeros((self.K, 2 * self.ser + 1))

    def step(self, action):

        self.Di = np.random.uniform(300, 500, self.K)
        self.Ci = np.random.uniform(900, 1100, self.K)

        self.done = [False] * self.K

        np.clip(action, 0, 1, out=action)
        action[np.isnan(action)] = 1
        # 拆分 `action_1`
        stra, f = np.split(action, [self.ser + 1], axis=1)
        stra_sum = np.sum(stra, axis=1, keepdims=True)
        f_sum = np.sum(f, axis=0, keepdims=True)
        stra /= np.maximum(stra_sum, 1e-6)
        f /= np.maximum(f_sum, 1e-6)
        a = self.pi * 0.001 * self.hi
        r_1 = f * self.B * 1e6 * np.log2(1 + (a / self.N0))


        T1_ij = stra[:, :self.ser] * self.Di[:, None] * 102400 / (1 + r_1)
        E1_ij = T1_ij * self.pi * 1e-5

        T2_ij = stra[:, :self.ser] * self.Ci[:, None] * 100 / (self.fi_m * 1000)
        E2_ij = stra[:, :self.ser] * self.Ci[:, None] * (self.fi_m ** 2) * 1e-5

        T3 = stra[:, self.ser] * self.Ci * 100 / (self.fi_l * 1000)
        E3 = stra[:, self.ser] * self.Ci * (self.fi_l ** 2) * 1e-5


        T1 = np.max(T1_ij, axis=1)
        T2 = np.max(T2_ij, axis=1)

        T = np.maximum(T1 + T2, T3)
        E = np.sum(E1_ij + E2_ij, axis=1) + E3


        self.reward_i = -(self.alpha * T + self.beta * E)[:, None]
        comm = DecPOSGPartialCommunication(N=self.K, M=self.ser, action_dim=2 * self.ser + 1,
                                           observation_space= (2 * self.ser + 1), lambda_step=0.1, k_max=500)

        self.state = comm.update_step(t=0, current_actions=action, current_obs=self.state)

        return self.state, self.reward_i, self.done, {}

    def reset(self):
        state, reward, done, _ = self.step(np.random.uniform(0, 1, (self.K, self.ser * 2 + 1)))
        return state



parser = argparse.ArgumentParser()
#The following parameters can be adjusted for different training scenarios.
parser.add_argument("--env_name", default="task offloading")
parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
parser.add_argument('--max_step', default=1000, type=int)
parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
parser.add_argument('--capacity', default=50000, type=int)  # replay buffer size
parser.add_argument('--batch_size', default=32, type=int)  # mini batch size
parser.add_argument('--hidden_dim', default=64, type=int, help='hidden dim')
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=1000, type=int)  # num of games

args = parser.parse_args()
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = Fun.relu(self.l1(x))
        x = Fun.relu(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return x

class Critic(nn.Module):
    def __init__(self, critic_dim, hidden_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(critic_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, u):
        x = torch.cat((x, u), 1)
        x = Fun.relu(self.l1(x))
        x = Fun.relu(self.l2(x))
        return self.l3(x)

class DDPG(object):
    def __init__(self, state_dim, action_dim, critic_dim, hidden_dim, device):
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(critic_dim, hidden_dim).to(device)
        self.critic_target = Critic(critic_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=2e-4)

        self.replay_buffer = ReplayBuffer(50000)
        self.minimal_size = args.batch_size
        self.batch_size = args.batch_size
        self.transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}

    def select_action(self, state):
        return self.actor(state).detach().cpu().numpy().flatten()

    def soft_update(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

class MADDPG:
    def __init__(self, env, state_dims, action_dims, critic_dim, hidden_dim, device):
        self.agents = []
        for i in range(env.K):
            self.agents.append(DDPG(state_dims[i], action_dims[i], critic_dim, hidden_dim, device))
        self.num = env.K
        self.device = device
        self.batchsize = args.batch_size

    def take_action(self, env, states):
        states = [torch.tensor(np.array([states[i]]), dtype=torch.float, device=self.device) for i in range(env.K)]
        return [agent.select_action(state) for agent, state in zip(self.agents, states)]

    def update(self, i_agent):

        cur_agent = self.agents[i_agent]
        for j in range(self.num):
            agent = self.agents[j]
            if agent.replay_buffer.size() >= agent.minimal_size:
                b_s, b_a, b_r, b_ns, b_d = agent.replay_buffer.sample(agent.batch_size)
                agent.transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                         'dones': b_d}
        multi_state = []
        multi_action = []
        multi_next_state = []
        multi_reward = []
        multi_done = []
        for i in range(self.num):
            state = torch.tensor(self.agents[i].transition_dict['states'],
                                 dtype=torch.float).squeeze(1).to(self.device)
            action = torch.tensor(np.array(self.agents[i].transition_dict['actions']), dtype=torch.float).squeeze(1).to(
                self.device)
            next_state = torch.tensor(self.agents[i].transition_dict['next_states'],
                                      dtype=torch.float).squeeze(1).to(self.device)
            reward = torch.tensor(np.array(self.agents[i].transition_dict['rewards']),
                                  dtype=torch.float).view(-1, 1).to(self.device)
            done = torch.tensor(self.agents[i].transition_dict['dones'],
                                dtype=torch.float).view(-1, 1).to(self.device)
            multi_state.append(state)
            multi_next_state.append(next_state)
            multi_action.append(action)
            multi_reward.append(reward)
            multi_done.append(done)
        multi_state = [state for state in multi_state if state.numel() > 0]
        multi_next_state = [next_state for next_state in multi_next_state if next_state.numel() > 0]
        multi_action = [action for action in multi_action if action.numel() > 0]
        multi_reward = [reward for reward in multi_reward if reward.numel() > 0]
        multi_done = [done for done in multi_done if done.numel() > 0]
        multi_state = torch.stack(multi_state).to(self.device)  # torch.tensor()不能把包含tensor的list转成tensor，纯list就可以转tensor
        multi_next_state = torch.stack(multi_next_state).to(self.device)
        multi_action = torch.stack(multi_action).to(self.device)
        multi_reward = torch.stack(multi_reward).to(self.device)
        multi_done = torch.stack(multi_done).to(self.device)
        state_t = multi_state.cpu().numpy().transpose(1, 0, 2)
        multi_state = torch.tensor(state_t, dtype=torch.float).to(self.device)
        next_state_t = multi_next_state.cpu().numpy().transpose(1, 0, 2)
        multi_next_state = torch.tensor(next_state_t, dtype=torch.float).to(self.device)
        action_t = multi_action.cpu().numpy().transpose(1, 0, 2)
        multi_action = torch.tensor(action_t, dtype=torch.float).to(self.device)

        # Compute the target Q value
        target_act = [self.agents[agt].actor_target(multi_next_state[:, agt, :].squeeze(1)) for agt in range(self.num)]
        target_act = torch.stack(target_act).to(self.device)
        mns = multi_next_state.reshape(self.batchsize, -1)
        ta = torch.transpose(target_act, 0, 1).reshape(self.batchsize, -1)
        target_Q = cur_agent.critic_target(mns, ta)
        target_Q = multi_reward[i_agent, :, :] + ((1 - multi_done[i_agent, :, :]) * args.gamma * target_Q).detach()  # 若到终止状态，则只算reward

        # Get current Q estimate
        current_Q = cur_agent.critic(multi_state.reshape(self.batchsize, -1), multi_action.reshape(self.batchsize, -1))

        # Compute critic loss
        critic_loss = Fun.mse_loss(current_Q, target_Q)
        cur_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        cur_agent.critic_optimizer.step( )
        # Compute actor loss
        poli = []
        for j in range(self.num):
            if j == i_agent:
                poli.append(cur_agent.actor(multi_next_state[:, i_agent, :].squeeze(1)))
            else:
                poli.append(multi_action[:, j, :].squeeze(1))
        poli = torch.stack(poli).to(self.device)
        actor_loss = -cur_agent.critic(multi_state.reshape(self.batchsize, -1), torch.transpose(poli, 0, 1).reshape(self.batchsize, -1)).mean()
        # Optimize the actor
        cur_agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        cur_agent.actor_optimizer.step()
    def save(self, path):
        for agt in self.agents:
            torch.save(agt.actor.state_dict(), path)
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, path, map_location=None):
        for agt in self.agents:
            agt.actor.load_state_dict(torch.load(path, map_location=map_location))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


    def update_all_target(self):
        for agt in self.agents:
            agt.soft_update()

def main():
    # The number of users and servers can be adjusted accordingly.
    # num = [50, 100, 150, 200, 250]
    # server = [5, 10, 15, 20, 25]

    num = 5  # Number of UDs
    server = 3
    device = torch.device("cuda:7" if torch.cuda.is_available() else 'cpu')
    fi_m = np.random.uniform(3, 7, server)
    fi_l = np.random.uniform(0.8, 1.5, num)
    Di = np.random.uniform(300, 500, num)
    Ci = np.random.uniform(900, 1100, num)

    hi = pow(np.random.uniform(50, 200, (num, server)), -3)
    env = Env(alpha=0.6, beta=0.4, B=10, N0=pow(10, -174 / 10) * 0.001,
              hi=hi, pi=500, K=num, ser=server, Di=Di, Ci=Ci, fi_m=fi_m, fi_l=fi_l)
    state_dims = []
    action_dims = []
    for i in range(env.K):
        state_dims.append(2 * env.ser + 1)
        action_dims.append((2 * env.ser + 1))
    critic_dim = np.sum(state_dims) + np.sum(action_dims)
    agents = MADDPG(env, state_dims, action_dims, critic_dim, args.hidden_dim, device)
    reward_history = []
    start_time = time.time()
    for i in range(args.max_episode):
        states = env.reset()
        reward_t = []
        for t in range(args.max_step):
            action = np.array(agents.take_action(env, states)).reshape(env.K, env.ser * 2 + 1)
            action = (action + np.random.normal(0, args.exploration_noise, size=(env.K, 1 + env.ser * 2)).clip(
                    0, 1))
            next_state, reward_i, done, info = env.step(action)
            if t == 999:
                done = [True] * len(done)
            # store memory
            for agent_i in range(env.K):
                agent = agents.agents[agent_i]
                agent.replay_buffer.add(states[agent_i, :].reshape(1, -1), action[agent_i], reward_i[agent_i],
                                        next_state[agent_i, :].reshape(1, -1), done[agent_i])

            if (t + 1) % 50 == 0:
                for i_agent in range(env.K):
                    agents.update(i_agent)
                print("Total T:{} Episode: \t{} Reward: \t{:0.2f}".format(t, i, np.sum(reward_i)))
                agents.update_all_target()
                agents.save(path='MADDPG_CTDE-50.pth')
            states = next_state
            reward_t.append(np.sum(reward_i))
        torch.cuda.empty_cache()
        reward_history.append(np.average(reward_t))
        for agent_i in range(env.K):
            agent = agents.agents[agent_i]
            agent.replay_buffer.clear()
    total_time = time.time() - start_time






#main()
if __name__ == '__main__':
    main()


