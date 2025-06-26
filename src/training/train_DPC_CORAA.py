#Training of DPC-CORAA, i.e., Algorithm 2
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as Fun
import torch.optim as optim
import time
import torch.multiprocessing as mp
from partial_commu import DecPOSGPartialCommunication

class Env():
    def __init__(self, alpha, beta, B, N0, hi, pi, K, ser, Di, Ci, fi_m, fi_l):
        """
        B,N0：Bandwidth and the variance of Gaussian white noise（10MHz=10e6Hz， pow(10, -174 / 10) * 0.001）
        hi, pi: Channel gain, transmission power 0.001 * pow(np.random.uniform(50, 200, num), -3)、500mW=0.5W、100
        K, ser: Number of UDs, number of ESs
        Di, Ci: Task data size, the required number of CPU cycles (300~500kb) 1024kb=1Mb, (900, 1100) 1Mhz = 1000khz = 1000*1000hz
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
        self.state = np.random.rand(self.K, self.K * (2 * self.ser + 1))

    def step(self, action):
        self.Di = np.random.uniform(300, 500, self.K)
        self.Ci = np.random.uniform(900, 1100, self.K)
        self.done = [False] * self.K

        np.clip(action, 0, 1, out=action)
        action[np.isnan(action)] = 1

        stra, f = np.split(action, [self.ser + 1], axis=1)
        stra_sum = np.sum(stra, axis=1, keepdims=True)
        f_sum = np.sum(f, axis=0, keepdims=True)
        stra /= np.maximum(stra_sum, 1e-6)
        f /= np.maximum(f_sum, 1e-6)
        act = np.concatenate([stra, f], axis=1)
        action_dim = 2 * self.ser + 1
        for i in range(self.K):
            self.state[i, i * action_dim:(i + 1) * action_dim] = act[i]
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

        # 计算奖励
        self.reward_i = -(self.alpha * T + self.beta * E)[:, None]
        comm = DecPOSGPartialCommunication(N=self.K, M=self.ser, action_dim=2 * self.ser + 1,
                                           observation_space=self.K * (2 * self.ser + 1), lambda_step=0.1, k_max=500)

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


class Replay_buffer():
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        o, on, a, r, d = [], [], [], [], []
        for i in ind:
            O, On, A, R, D = self.storage[i]
            o.append(np.array(O, copy=False))#observation
            on.append(np.array(On, copy=False))  # next observation
            a.append(np.array(A, copy=False))  # action
            r.append(np.array(R, copy=False))  # reward
            d.append(np.array(D, copy=False))  # done
        return np.array(o), np.array(on), np.array(a), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

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
        self.device = device
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(critic_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(critic_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=2e-4)

        self.replay_buffer = Replay_buffer()

    def select_action(self, state):

        return self.actor(state).detach().cpu().numpy().flatten()

    def soft_update(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

class MADDPG:
    def __init__(self, env, state_dims, action_dims, critic_dim, hidden_dim):
        self.agents = []
        self.num_gpus = 8
        self.device_list = [f"cuda:{i}" for i in range(self.num_gpus)]

        for i in range(env.K):
            device = torch.device(self.device_list[i % self.num_gpus])
            self.agents.append(DDPG(state_dims[i], action_dims[i], critic_dim, hidden_dim, device))

        self.num = env.K
        self.ser = env.ser
        self.statedim = state_dims[0]
        self.actiondim = action_dims[0]
        self.criticdim = critic_dim


    def take_action(self,states):
        actions = []
        for i, agent in enumerate(self.agents):
            device = agent.device
            state = torch.tensor(np.array([states[i, :]]), dtype=torch.float, device=device)
            actions.append(agent.select_action(state))
        return actions

    def update(self, i_agent):
        cur_agent = self.agents[i_agent]
        x, y, u, r, d = cur_agent.replay_buffer.sample(args.batch_size)

        state = torch.FloatTensor(x).to(cur_agent.device)
        action = torch.FloatTensor(u).to(cur_agent.device)
        next_state = torch.FloatTensor(y).to(cur_agent.device)
        done = torch.FloatTensor(d).to(cur_agent.device)
        reward = torch.FloatTensor(r).to(cur_agent.device)
        target_act = cur_agent.actor_target(next_state)

        target_Q = cur_agent.critic_target(next_state, target_act)
        target_Q = reward + (
                (1 - done) * args.gamma * target_Q).detach()  # 若到终止状态，则只算reward

        # Get current Q estimate
        current_Q = cur_agent.critic(state, action)

        # Compute critic loss
        critic_loss = Fun.mse_loss(current_Q, target_Q.detach())
        cur_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        actor_loss = -cur_agent.critic(state, cur_agent.actor(state)).mean()
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
            agt.actor.eval()

        print("====================================")
        print("model has been loaded...")
        print("====================================")

    def update_all_target(self):
        for agt in self.agents:
            agt.soft_update()

def train_agent(agent_id, gpu_id, process, num, server):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    fi_m = np.random.uniform(3, 7 , server)
    fi_l = np.random.uniform(0.8, 1.5, num)
    Di = np.random.uniform(300, 500, num)
    Ci = np.random.uniform(900, 1100, num)

    hi = pow(np.random.uniform(50, 200, (num, server)), -3)
    env = Env(alpha=0.6, beta=0.4, B=10, N0=pow(10, -174 / 10) * 0.001,
              hi=hi, pi=500, K=num, ser=server, Di=Di, Ci=Ci, fi_m=fi_m, fi_l=fi_l)

    state_dims = []
    action_dims = []
    for i in range(env.K):
        state_dims.append((2 * env.ser + 1) * env.K)
        action_dims.append(2 * env.ser + 1)
    critic_dim = state_dims[0] + action_dims[0]
    agents = MADDPG(env, state_dims, action_dims, critic_dim, args.hidden_dim)
    reward_history = []
    start_time = time.time()
    for i in range(args.max_episode):
        states = env.reset()
        reward_t = []
        for t in range(args.max_step):
            action = np.array(agents.take_action(states)).reshape(env.K, env.ser * 2 + 1)
            action = (action + np.random.normal(0, args.exploration_noise, size=(env.K, 1 + env.ser * 2)).clip(
                0, 1))
            # execute action
            next_state, reward_i, done, info = env.step(action)
            if t == 999:
                done = [True] * len(done)
            # store memory
            for agent_i in range(env.K):
                agent1 = agents.agents[agent_i]
                agent1.replay_buffer.push(
                    (states[agent_i, :], next_state[agent_i, :], action[agent_i], reward_i[agent_i],
                     float(done[agent_i]))
                )
            if (t + 1) % 50 == 0:
                agents.update(agent_id)
                print(f"[Agent {agent_id}] Total T:{t} Episode: {i} Reward: {np.sum(reward_i):.2f}")
                if agent_id == env.K - 1:
                    agents.update_all_target()
                    agents.save(path='DPC-50.pth')
            states = next_state
            reward_t.append(np.sum(reward_i))
        process.put(agent_id)
        torch.cuda.empty_cache()
        reward_history.append(np.average(reward_t))

    total_time = time.time() - start_time



if __name__ == "__main__":
    #The number of users and servers can be adjusted accordingly.
    # num_agents = [50, 100, 150, 200, 250]
    # server = [5, 10, 15, 20, 25]

    num_agents = 5  # Number of UDs
    server = 3
    num_gpus = 8  # Number of GPU
    agents_per_batch = 5  # Each batch runs with 5 agents.
    process = mp.Queue()
    processes = []

    for batch_start in range(0, num_agents, agents_per_batch):
        for i in range(agents_per_batch):
            agent_id = batch_start + i
            if agent_id >= num_agents:
                break
            gpu_id = agent_id % num_gpus
            p = mp.Process(target=train_agent, args=(agent_id, gpu_id, process, num_agents, server))
            p.start()
            processes.append(p)


        for p in processes:
            p.join()
