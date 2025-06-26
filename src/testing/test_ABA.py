import numpy as np
import test_env as te
from training import train_ABA as py


def main(env):
    state_dims = []
    action_dims = []
    for i in range(env.K):
        state_dims.append((env.ser + 1) * env.K)
        action_dims.append(env.ser + 1)
    critic_dim = state_dims[0] + action_dims[0]
    agents = py.MADDPG(env, state_dims, action_dims, critic_dim, 64)
    agents.load(path='ABA.pth')
    reward_history = []
    for i in range(20):
        initial = np.random.uniform(0, 1, (env.K, env.ser + 1))
        states, _, _, _ = env.step(initial)
        reward_t = []
        for t in range(50):
            action = np.array(agents.take_action(states)).reshape(env.K, env.ser + 1)
            action = (action + np.random.normal(0, 0.1, size=(env.K, 1 + env.ser)).clip(
                0, 1))
            next_state, reward_i, done, _ = env.step(action)
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

energy1 = np.zeros((5, 1))
time1 = np.zeros((5, 1))
time2 = np.zeros((5, 1))
energy2 = np.zeros((5, 1))
fi_m = te.fi_m[: server]
fi_l = np.ones(num)
Di = 400*np.ones(num)
Ci = 900*np.ones(num)
C = [900, 950, 1000, 1050, 1100]
D = [300, 350, 400, 450, 500]
hi = te.hi[: num, : server]
Bi = 10

for i in range(5):
    Di_test = D[i] * np.ones(num)
    time1[i, 0] = (-1) * main(py.Env(alpha=1, beta=0, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                  hi=hi, pi=500, K=num, ser=server,
                                  Di=Di_test, Ci=Ci,
                                  fi_m=fi_m, fi_l=fi_l))

    energy1[i, 0] = (-1) * main(py.Env(alpha=0, beta=1, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                  hi=hi, pi=500, K=num, ser=server,
                                  Di=Di_test, Ci=Ci,
                                   fi_m=fi_m, fi_l=fi_l))


for i1 in range(5):
    Ci_test = C[i1] * np.ones(num)
    time2[i1, 0] = (-1) * main(py.Env(alpha=1, beta=0, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                  hi=hi, pi=500, K=num, ser=server,
                                  Di=Di, Ci=Ci_test,
                                  fi_m=fi_m, fi_l=fi_l))

    energy2[i1, 0] = (-1) * main(py.Env(alpha=0, beta=1, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                  hi=hi, pi=500, K=num, ser=server,
                                  Di=Di, Ci=Ci_test,
                                   fi_m=fi_m, fi_l=fi_l))


