import numpy as np

num = 250
server = 25
hi = np.zeros((num, server))
for i in range(num):
    for j in range(server):
        hi[i, j] = 0.001 * pow(np.random.uniform(50, 200), -3)
fi_m = np.random.uniform(3, 7, server)
fi_l = np.random.uniform(0.8, 1.5, num)
Di = np.random.uniform(300, 500, num)
Ci = np.random.uniform(900, 1100, num)
