#partial communication model, i.e., Algorithm 2
import numpy as np


class DecPOSGPartialCommunication:
    def __init__(self, N, M, action_dim, observation_space, lambda_step=0.1, k_max=200):
        """
        :param N: Number of user devices (UDs)
        :param action_dim: The dimensionality of each UD's observation
        :param observation_space: The dimensionality of each UD’s observation
        :param lambda_step: Step size for gradient descent
        :param k_max: Maximum number of communication iterations
        """
        self.N = N
        self.M = M
        self.action_dim = action_dim
        self.obs_dim = observation_space

        self.lambda_step = lambda_step
        self.k_max = k_max

        self.current_actions = np.zeros(N)
        self.current_obs = np.zeros((N, observation_space))
        self.next_obs = np.zeros_like(self.current_obs)

    def _projection(self, x, ud_idx):

        return np.clip(x, 0, 1)
    def _compute_weights(self, t, O):
        share_matrix = np.zeros((self.N, self.N), dtype=int)
        for j in range(self.M):
            for i in range(self.N):
                for h in range(i + 1, self.N):
                    if np.any(O[i, h, j] != 0) or np.any(O[h, i, j] != 0):
                        share_matrix[i, h] = 1
                        share_matrix[h, i] = 1
        np.fill_diagonal(share_matrix, 0)

        d = share_matrix.sum(axis=1)  # d_i^t

        W = np.zeros((self.N, self.N))
        for i in range(self.N):
            for h in range(self.N):
                if i != h and share_matrix[i, h] > 0:
                    W[i, h] = share_matrix[i, h] / (max(d[i], d[h]) + 1)
            W[i, i] = 1 - W[i].sum()
        return W

    def update_step(self, t, current_actions, current_obs):
        """
        # Perform a complete observation update process
        :param t: Current time slot
        :param current_actions: Array of current actions
        :param current_obs: Matrix of current observations
        :return: Observation matrix for the next time step
        """
        np.copyto(self.next_obs, current_obs)
        O = self.next_obs.reshape((self.N, self.N, -1))  # [5,5,7]
        O = O[:, :, :self.M + 1]  # [5,5,4]
        W = self._compute_weights(t, O)
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

        return self.next_obs.copy()



