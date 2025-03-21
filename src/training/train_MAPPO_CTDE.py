#Training of MAPPO approach under the CTDE scheme
import numpy as np
import argparse
import torch
import torch.nn.functional as F
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
parser.add_argument('--gae_lambda', default=0.95, type=float, help='GAE lambda')
parser.add_argument('--policy_clip', default=0.1, type=float, help='policy clip')
parser.add_argument('--n_epochs', default=15, type=int, help='update number')
parser.add_argument('--hidden_dim', default=64, type=int, help='hidden dim')

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

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1)
        )


    def forward(self, x):
        x = self.critic(x)
        return x


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = F.softplus(self.fc_mu(x)) + 1e-6  # 输出均值
        std = F.softplus(self.fc_std(x)) + 1e-6
        mu = torch.nan_to_num(mu, nan=1.0, posinf=1.0, neginf=1.0)
        std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
        mu[mu < 1] += 1
        std[std < 1] += 1
        return mu, std

class PPOContinuous:
    def __init__(self, state_dim, action_dim, critic_dim, cfg, device):
        self.actor = PolicyNetContinuous(state_dim, cfg.hidden_dim,
                                         action_dim).to(device)
        self.critic = ValueNet(critic_dim, cfg.hidden_dim).to(device)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(),
                                                lr=1e-4, weight_decay=1e-2)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(),
                                                 lr=2e-4, weight_decay=1e-2)
        self.replay_buffer = ReplayBuffer(50000)
        self.minimal_size = cfg.batch_size
        self.batch_size = cfg.batch_size
        self.device = device
        self.transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}

    def select_action(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state.reshape(1, -1))
        action_dist = torch.distributions.Beta(mu, sigma)
        action = action_dist.sample()
        return action

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)

class MAPPO:
    def __init__(self, env, n_states, n_actions, n_critic, cfg, device):
        self.gamma = cfg.gamma
        self.eps = cfg.policy_clip
        self.gae_lambda = cfg.gae_lambda
        self.K = env.K  # 用户数量
        self.batchsize = cfg.batch_size
        self.device = device
        self.epochs = cfg.n_epochs
        self.loss = 0
        self.agents = []
        for i in range(self.K):
            self.agents.append(PPOContinuous(n_states[i], n_actions[i], n_critic, cfg, device))

    def take_action(self, state):
        actions = []
        states = torch.FloatTensor(np.array(state)).to(self.device)
        for i in range(self.K):
            state = states[i].clone().detach().unsqueeze(0).to(self.device)
            mu, sigma = self.agents[i].actor(state)
            action_dist = torch.distributions.Normal(mu, sigma)
            action = action_dist.sample()
            action = torch.sigmoid(action)
            actions.append(action.detach().cpu().numpy())
        return actions

    def update(self, i_agent, n_states):
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
    for i in range(num):
        state_dims.append(2 * server + 1)
        action_dims.append((2 * server + 1))
    critic_dim = np.sum(state_dims)
    agents = MAPPO(env, state_dims, action_dims, critic_dim, args, device)
    reward_history = []
    start_time = time.time()
    for i in range(args.max_episode):
        states = env.reset()
        reward_t = []
        for t in range(args.max_step):
            action = agents.take_action(states)
            next_state, reward, done, _ = env.step(np.array(action).squeeze(1))
            if t == 999:
                done = [True] * len(done)
            # store memory
            for agent_i in range(env.K):
                agent = agents.agents[agent_i]
                agent.replay_buffer.add(states[agent_i, :].reshape(1, -1), action[agent_i], reward[agent_i],
                                        next_state[agent_i, :].reshape(1, -1), done[agent_i])

            if (t + 1) % 50 == 0:
                for i_agent in range(env.K):
                    agents.update(i_agent, state_dims)
                print("Total T:{} Episode: \t{} Reward: \t{:0.2f}".format(t, i, np.sum(reward)))
                agents.save(path='MAPPO_CTDE-50.pth')
            states = next_state
            reward_t.append(np.sum(reward))
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


