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

    def _projection(self, x, ud_idx):
        # [RESTRICTED] This function is temporarily disabled due to confidentiality agreements.
        # Full implementation will be released upon paper acceptance.
        pass
    def _compute_weights(self, t):
        # [RESTRICTED] This function is temporarily disabled due to confidentiality agreements.
        # Full implementation will be released upon paper acceptance.
        pass

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
        return self.next_obs.copy()


