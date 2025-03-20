#Testing of FLO
import numpy as np
import test_env as te

class Env():
    def __init__(self, alpha, beta, K, ser, Di, Ci, fi_l):
        """
        K, ser: Number of UDs, number of ESs
        Di, Ci: Task data size, the required number of CPU cycles (300~500kb) 1024kb=1Mb, (900, 1100)兆周期数 1Mhz = 1000khz = 1000*1000hz
        fi_m: Maximum computational capacity of the server 3-7 GHz/s 10e9Hz/s
        """
        self.alpha, self.beta = alpha, beta
        self.K, self.ser = K, ser
        self.Di, self.Ci = Di, Ci
        self.fi_l = fi_l

    def step(self):

        T = self.Ci  / (self.fi_l * 1000)
        E = self.Ci * (self.fi_l ** 2) * 1e-2

        self.reward_i = -(self.alpha * T + self.beta * E)[:, None]

        return np.sum(self.reward_i)

    def reset(self):
        reward = self.step()
        return reward



def main():
    # The number of users and servers can be adjusted accordingly.
    # num_agents = [50, 100, 150, 200, 250]
    # server = [5, 10, 15, 20, 25]

    num = 5  # Number of UDs
    server = 3
    energy1_local = np.zeros((5, 1))
    time1_local = np.zeros((5, 1))
    time2_local = np.zeros((5, 1))
    energy2_local = np.zeros((5, 1))

    env1 = []
    env2 = []
    env3 = []
    env4 = []

    fi_l = te.fi_l[: num]
    Di = 400 * np.ones((num))
    Ci = 900 * np.ones((num))
    C = [900, 950, 1000, 1050, 1100]
    D = [300, 350, 400, 450, 500]

    for i1 in range(5):
        env1.append(Env(alpha=1, beta=0,
                            K=num, ser=server, Di=D[i1] * np.ones(num), Ci=Ci,
                        fi_l=fi_l))
        time1_local[i1, 0] = (-1) * env1[i1].reset()
        env2.append(Env(alpha=0, beta=1,
                            K=num, ser=server, Di=D[i1] * np.ones(num), Ci=Ci,
                            fi_l=fi_l))
        energy1_local[i1, 0] = (-1) * env2[i1].reset()
    # data1 = {'Task Data(kbits)': D,
    #          'Time delay_local': time1_local.flatten().tolist(), 'Energy_local': energy1_local.flatten().tolist()}
    # print(data1)
    for i2 in range(5):

        env3.append(Env(alpha=1, beta=0,
                           K=num, ser=server, Di=Di, Ci=C[i2] * np.ones(num),
                            fi_l=fi_l))
        time2_local[i2, 0] = (-1) * env3[i2].reset()
        env4.append(Env(alpha=0, beta=1,
                            K=num, ser=server, Di=Di, Ci=C[i2] * np.ones(num),
                            fi_l=fi_l))
        energy2_local[i2, 0] = (-1) * env4[i2].reset()
    # data2 = {'The amount of CPU cycles(megacycles)': C,
    #              'Time delay_Local': time2_local.flatten().tolist(), 'Energy_Local': energy2_local.flatten().tolist()}
    # print(data2)


#main()
if __name__ == '__main__':
    main()



