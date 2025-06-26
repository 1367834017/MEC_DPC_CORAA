#Performance analysis of CTDE-based MADDPG approach, including the impact of task data, CPU cycles, and bandwidth on reward, time delay and energy cost.
import numpy as np
import torch

import random
import test_env as te
from training import train_MAPPO_CTDE as py
random.seed(10)


def main(env):
    device = torch.device("cuda:7" if torch.cuda.is_available() else 'cpu')
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
        initial = np.random.uniform(0, 1, (env.K, env.ser * 2 + 1))
        states, _, _, _ = env.step(initial)
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
# num_agents = [50, 100, 150, 200, 250]
# server = [5, 10, 15, 20, 25]

num = 5  # Number of UDs
server = 3

reward1 = np.zeros((5, 1))
reward2 = np.zeros((5, 1))
energy1 = np.zeros((5, 1))
time1 = np.zeros((5, 1))
time2 = np.zeros((5, 1))
energy2 = np.zeros((5, 1))
reward3 = np.zeros((5, 1))
time3 = np.zeros((5, 1))
energy3 = np.zeros((5, 1))

fi_m = te.fi_m[: server]
fi_l = te.fi_m[: num]
Di = 400*np.ones(num)
Ci = 900*np.ones(num)
C = [900, 950, 1000, 1050, 1100]
D = [300, 350, 400, 450, 500]
hi = te.hi[: num, : server]
Bi = 10
B = [5, 10, 15, 20, 25]

#The impact of task data on reward, time delay and energy cost.
for i1 in range(5):
    Di_test = D[i1] * np.ones(num)
    reward1[i1, 0] = main(py.Env(alpha=0.6, beta=0.4, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                  hi=hi, pi=500, K=num, ser=server,
                                  Di=Di_test, Ci=Ci,
                                  fi_m=fi_m, fi_l=fi_l))

    time1[i1, 0] = (-1) * main(py.Env(alpha=1, beta=0, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                  hi=hi, pi=500, K=num, ser=server,
                                  Di=Di_test, Ci=Ci,
                                  fi_m=fi_m, fi_l=fi_l))

    energy1[i1, 0] = (-1) * main(py.Env(alpha=0, beta=1, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                  hi=hi, pi=500, K=num, ser=server,
                                  Di=Di_test, Ci=Ci,
                                  fi_m=fi_m, fi_l=fi_l))


#The impact of CPU cycles on reward, time delay and energy cost.
for i2 in range(5):
    Ci_test = C[i2] * np.ones(num)
    reward2[i2, 0] = main(py.Env(alpha=0.6, beta=0.4, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                 hi=hi, pi=500, K=num, ser=server,
                                 Di=Di, Ci=Ci_test,
                                 fi_m=fi_m, fi_l=fi_l))

    time2[i2, 0] = (-1) * main(py.Env(alpha=1, beta=0, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                  hi=hi, pi=500, K=num, ser=server,
                                  Di=Di, Ci=Ci_test,
                                  fi_m=fi_m, fi_l=fi_l))

    energy2[i2, 0] = (-1) * main(py.Env(alpha=0, beta=1, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                  hi=hi, pi=500, K=num, ser=server,
                                  Di=Di, Ci=Ci_test,
                                  fi_m=fi_m, fi_l=fi_l))

#The impact of bandwidth on reward, time delay and energy cost.
for i3 in range(5):
    reward3[i3, 0] = main(py.Env(alpha=0.6, beta=0.4, B=B[i3], N0=pow(10, -174 / 10) * 0.001,
                       hi=hi, pi=500, K=num, ser=server, Di=Di, Ci=Ci,
                       fi_m=fi_m, fi_l=fi_l))
    time3[i3, 0] = (-1) * main(py.Env(alpha=1, beta=0, B=B[i3], N0=pow(10, -174 / 10) * 0.001,
                        hi=hi, pi=500, K=num, ser=server, Di=Di, Ci=Ci,
                        fi_m=fi_m, fi_l=fi_l))
    energy3[i3, 0] = (-1) * main(py.Env(alpha=0, beta=1, B=B[i3], N0=pow(10, -174 / 10) * 0.001,
                        hi=hi, pi=500, K=num, ser=server, Di=Di, Ci=Ci,
                        fi_m=fi_m, fi_l=fi_l))
