# Performance analysis of CTDE-based MAPPO, including the impact of number of UDs, ESs
# on reward, time delay and energy cost.
import numpy as np
import torch
import random
import test_env as te
from training import train_MAPPO_CTDE as py
random.seed(10)


def main(env):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dims = []
    action_dims = []
    for i in range(env.K):
        state_dims.append(2 * env.ser + 1)
        action_dims.append((2 * env.ser + 1))
    critic_dim = np.sum(state_dims)
    agents = py.MAPPO(env, state_dims, action_dims, critic_dim, py.args, device)
    agents.load(path='MAPPO_CTDE.pth')
    reward_history = []
    for i in range(20):
        states = env.reset()
        reward_t = []
        for t in range(50):
            action = np.array(agents.take_action(states))
            next_state, reward_i, done, _ = env.step(np.array(action).squeeze(1))
            if t % 10 == 0:
                print("Episode: \t{} Reward: \t{:0.2f}".format(i, np.sum(reward_i)))
            reward_t.append(np.sum(reward_i))
            states = next_state
        reward_history.append(np.average(reward_t))
    return np.mean(reward_history)


# The number of users and servers can be adjusted accordingly.
num_agents = [50, 100, 150, 200, 250]
num_servers = [5, 10, 15, 20, 25]

num = 5  # Number of UDs
server = 3

reward1 = np.zeros((5, 1))
reward2 = np.zeros((5, 1))
energy1 = np.zeros((5, 1))
time1 = np.zeros((5, 1))
time2 = np.zeros((5, 1))
energy2 = np.zeros((5, 1))

Di = 400 * np.ones(num)
Ci = 900 * np.ones(num)
C = [900, 950, 1000, 1050, 1100]
D = [300, 350, 400, 450, 500]
Bi = 10
B = [5, 10, 15, 20, 25]

# The impact of number of UDs on reward, time delay and energy cost.
for i in range(5):
    reward1[i, 0] = main(py.Env(alpha=0.6, beta=0.4, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                hi=te.hi[: num_agents[i], : server], pi=500, K=num_agents[i], ser=server,
                                Di=Di, Ci=Ci,
                                fi_m=te.fi_m[: server], fi_l=te.fi_l[: num_agents[i]]))

    time1[i, 0] = (-1) * main(py.Env(alpha=1, beta=0, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                     hi=te.hi[: num_agents[i], : server], pi=500, K=num_agents[i], ser=server,
                                     Di=Di, Ci=Ci,
                                     fi_m=te.fi_m[: server], fi_l=te.fi_l[: num_agents[i]]))

    energy1[i, 0] = (-1) * main(py.Env(alpha=0, beta=1, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                       hi=te.hi[: num_agents[i], : server], pi=500, K=num_agents[i], ser=server,
                                       Di=Di, Ci=Ci,
                                       fi_m=te.fi_m[: server], fi_l=te.fi_l[: num_agents[i]]))

# The impact of number of ESs on reward, time delay and energy cost.
for i1 in range(5):
    reward2[i1, 0] = main(py.Env(alpha=0.6, beta=0.4, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                 hi=te.hi[: num, : num_servers[i1]], pi=500, K=num, ser=num_servers[i1],
                                 Di=Di, Ci=Ci,
                                 fi_m=te.fi_m[: num_servers[i1]], fi_l=te.fi_l[: num]))

    time2[i1, 0] = (-1) * main(py.Env(alpha=1, beta=0, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                      hi=te.hi[: num, : num_servers[i1]], pi=500, K=num, ser=num_servers[i1],
                                      Di=Di, Ci=Ci,
                                      fi_m=te.fi_m[: num_servers[i1]], fi_l=te.fi_l[: num]))

    energy2[i1, 0] = (-1) * main(py.Env(alpha=0, beta=1, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                        hi=te.hi[: num, : num_servers[i1]], pi=500, K=num, ser=num_servers[i1],
                                        Di=Di, Ci=Ci,
                                        fi_m=te.fi_m[: num_servers[i1]], fi_l=te.fi_l[: num]))
