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

    def step(self, action):
        # [RESTRICTED] This function is temporarily disabled due to confidentiality agreements.
        # Full implementation will be released upon paper acceptance.
        pass

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
        # [RESTRICTED] This function is temporarily disabled due to confidentiality agreements.
        # Full implementation will be released upon paper acceptance.
        pass
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
    agents = MADDPG(env, state_dims, action_dims, critic_dim, 64, device)
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

    print(f"已分配 GPU 内存: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")
    print(f"GPU 缓存内存: {torch.cuda.memory_reserved(device) / (1024 ** 2):.2f} MB")
    print(
        f"Time = {total_time:.2f}s")




#main()
if __name__ == '__main__':
    main()


