#partial communication model, i.e., Algorithm 2
import numpy as np


class DecPOSGPartialCommunication:
    def __init__(self, N, action_dim, observation_space, lambda_step=0.1, k_max=200):
        """
        :param N: Number of user devices (UDs)
        :param action_dim: The dimensionality of each UD's observation
        :param observation_space: The dimensionality of each UD’s observation
        :param lambda_step: Step size for gradient descent
        :param k_max: Maximum number of communication iterations
        """
        self.N = N
        self.action_dim = action_dim
        self.obs_dim = observation_space
        self.lambda_step = lambda_step
        self.k_max = k_max

        self.current_actions = np.zeros(N)
        self.current_obs = np.zeros((N, observation_space))
        self.next_obs = np.zeros_like(self.current_obs)

#注释掉投影和计算邻接矩阵部分
#-------------------------------------------------------------
    def _projection(self, x, ud_idx):

        # """投影操作到动作空间 (高效实现)"""
        # # 假设action_space为[min, max]区间
        # return np.clip(x, 0, 1)
        # [RESTRICTED] This function is temporarily disabled due to confidentiality agreements.
        # Full implementation will be released upon paper acceptance.
        pass
    def _compute_weights(self, t):
        # """计算通信权重矩阵 (示例实现)"""
        # # 这里简化为全连接网络，实际应根据拓扑动态生成
        # return np.ones((self.N, self.N)) / self.N
        # [RESTRICTED] This function is temporarily disabled due to confidentiality agreements.
        # Full implementation will be released upon paper acceptance.
        pass
    #------------------------------------------------------------
    def update_step(self, t, current_actions, current_obs):
        """
        # Perform a complete observation update process
        :param t: Current time slot
        :param current_actions: Array of current actions
        :param current_obs: Matrix of current observations
        :return: Observation matrix for the next time step
        """

        np.copyto(self.next_obs, current_obs)
        for i in range(self.N):
            self.next_obs[i, i*self.action_dim:(i+1)*self.action_dim] = current_actions[i]  # o_{i,i}^{0,(t)}

        W = self._compute_weights(t)
        er = []

        for k in range(self.k_max):
            temp_obs = np.zeros_like(self.next_obs)
            for i in range(self.N):
                # 式(18): 通信更新
                delta = W[i, None, :] @ (self.next_obs - self.next_obs[i])
                temp_obs[i] = self.next_obs[i] + delta

                grad = (temp_obs[i, i*self.action_dim:(i+1)*self.action_dim] - current_actions[i])
                updated_action = temp_obs[i, i*self.action_dim:(i+1)*self.action_dim] - self.lambda_step * grad
                temp_obs[i, i*self.action_dim:(i+1)*self.action_dim] = self._projection(updated_action, i)
            self.next_obs[:] = temp_obs
            action = current_actions.reshape(1, -1)
            er.append(np.linalg.norm(self.next_obs - action, ord='fro') / self.next_obs.size)


        # plt.pause(0.01)
        # plt.cla()
        # plt.plot(range(len(reward_history)), reward_history, label="MADDPG")
        #绘图
        # plt.rcParams['figure.dpi'] = 500
        # plt.figure(figsize=(9, 7))
        # plt.plot(range(len(er)), er, linewidth=1.5)
        # plt.axhline(y=0, color="gray", linestyle="dashed", linewidth=1.5)
        # plt.xlabel("Iteration k", fontsize=16)
        # plt.ylabel(r"$\|\mathbf{o}^k - \mathbf{a}\|^2$", fontsize=16)
        # plt.xticks(fontsize=14)  # 调整 x 轴刻度字体大小
        # plt.yticks(fontsize=14)  # 调整 y 轴刻度字体大小        #plt.legend()
        #
        # plt.savefig("./conv.jpg")
        # plt.show()
        return self.next_obs.copy()


# 使用示例 ---------------------------------------------------
# if __name__ == "__main__":
#     # 系统参数
#     N = 5  # 用户设备数量
#     ser = 3
#     #action_space = [(0, 1.0) for _ in range(N)]  # 每个UD的动作空间
#     action_dim = 2 * ser + 1
#     obs_dim = N * (2 * ser + 1) # 观测维度5*7
#
#     # 初始化通信模块
#     comm = DecPOSGPartialCommunication(
#         N=N,
#         action_dim=action_dim,
#         observation_space=obs_dim,
#         lambda_step=0.1,
#         k_max=500
#     )
#
#     # 模拟输入数据
#     current_actions = np.random.uniform(0, 1, (N, 2 * ser + 1))
#     current_obs = np.random.uniform(0, 1, (N, obs_dim))
#     obs = []
#     er = []
#     #for t in range(500):
#     # 执行更新
#     next_obs = comm.update_step(t=0, current_actions=current_actions, current_obs=current_obs)
#         #current_obs = next_obs
#         # obs.append(next_obs.reshape(-1, 1))
#         # action = np.kron(np.ones((N, 1)), current_actions).reshape(-1, 1)
#         # er.append( np.linalg.norm(obs[t] - action, 2))
#         # if t % 20 == 0:
#         #     plt.pause(0.01)
#         #     plt.cla()
#         #     # plt.plot(range(len(reward_history)), reward_history, label="MADDPG")
#         #     plt.plot(range(len(er)), er)
#         #     plt.draw()