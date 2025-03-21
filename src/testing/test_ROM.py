import numpy as np
import test_env as te


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

    def step(self, action):
        np.clip(action, 0, 1, out=action)
        action[np.isnan(action)] = 1
        stra, f = np.split(action, [self.ser], axis=1)
        stra_sum = np.sum(stra, axis=1, keepdims=True)
        f_sum = np.sum(f, axis=0, keepdims=True)
        stra /= np.maximum(stra_sum, 1e-6)
        f /= np.maximum(f_sum, 1e-6)
        a = self.pi * 0.001 * self.hi
        r_1 = f * self.B * 1e6 * np.log2(1 + (a / self.N0))  # 直接矩阵运算
        T1_ij = stra[:, :self.ser] * self.Di[:, None] * 1024 / (1 + r_1)
        E1_ij = T1_ij * self.pi * 1e-3
        T2_ij = stra[:, :self.ser] * self.Ci[:, None] / (self.fi_m * 1000)
        E2_ij = stra[:, :self.ser] * self.Ci[:, None] * (self.fi_m ** 2) * 1e-3
        T1 = np.max(T1_ij, axis=1)
        T2 = np.max(T2_ij, axis=1)
        T = T1 + T2  # 总时延
        E = np.sum(E1_ij + E2_ij, axis=1)
        self.reward_i = -(self.alpha * T + self.beta * E)[:, None]


        return np.sum(self.reward_i)


    def reset(self, initial):
        reward = self.step(initial)
        return reward


def main():
    # The number of users and servers can be adjusted accordingly.
    # num_agents = [50, 100, 150, 200, 250]
    # server = [5, 10, 15, 20, 25]

    num = 5  # Number of UDs
    server = 3

    energy1 = np.zeros((5, 1))
    time1 = np.zeros((5, 1))
    time2 = np.zeros((5, 1))
    energy2 = np.zeros((5, 1))
    env1 = []
    env2 = []
    env3 = []
    env4 = []
    fi_m = te.fi_m[: server]
    fi_l = te.fi_l[: num]
    Di = 400 * np.ones((num))
    Ci = 900 * np.ones((num))
    C = [900, 950, 1000, 1050, 1100]
    Bi = 10
    D = [300, 350, 400, 450, 500]
    hi = te.hi[: num, : server]
    initial = np.random.uniform(0, 1, (num, server * 2))
    for i1 in range(5):

        env1.append(Env(alpha=1, beta=0, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                         hi=hi, pi=500, K=num, ser=server, Di=D[i1] * np.ones(num), Ci=Ci,
                         fi_m=fi_m, fi_l=fi_l))
        time1[i1, 0] = (-1) * env1[i1].reset(initial)
        env2.append(Env(alpha=0, beta=1, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                         hi=hi, pi=500, K=num, ser=server, Di=D[i1] * np.ones(num), Ci=Ci,
                         fi_m=fi_m, fi_l=fi_l))
        energy1[i1, 0] = (-1) * env2[i1].reset(initial)
    #
    # data1 = {'Task Data(kbits)': D,
    #          'Time delay_ROM': time1.flatten().tolist(), 'Energy_ROM': energy1.flatten().tolist()}


    for i2 in range(5):
        env3.append(Env(alpha=1, beta=0, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                         hi=hi, pi=500, K=num, ser=server, Di=Di, Ci=C[i2] * np.ones(num),
                         fi_m=fi_m, fi_l=fi_l))
        time2[i2, 0] = (-1) * env3[i2].reset(initial)
        env4.append(Env(alpha=0, beta=1, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                         hi=hi, pi=500, K=num, ser=server, Di=Di, Ci=C[i2] * np.ones(num),
                         fi_m=fi_m, fi_l=fi_l))
        energy2[i2, 0] = (-1) * env4[i2].reset(initial)
    # data2 = {'The amount of CPU cycles(megacycles)': C,
    #          'Time delay_ROM': time2.flatten().tolist(), 'Energy_ROM': energy2.flatten().tolist()}
    #
    #
    # print(data1)
    # print(data2)



# main()
if __name__ == '__main__':
    main()






