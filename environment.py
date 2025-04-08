import numpy as np
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import Discrete

from kernel import PennylaneKernel, FidelityLossEvaluator


class RLKernelEnvironment(Environment):
    def __init__(self, initial_kernel: PennylaneKernel, X: np.ndarray, y: np.ndarray,
                 ke: FidelityLossEvaluator, bw_possible=1, convert_to_int=False, mode="wide"):
        self.initial_kernel = initial_kernel
        self.X = X
        self.y = y
        self.ke = ke
        self.convert_to_int = convert_to_int
        self.bw = bw_possible
        self.mode = mode.lower()

        self.n_operations = initial_kernel.ansatz.n_operations
        self.n_features = initial_kernel.ansatz.n_features
        self.n_qubits = initial_kernel.ansatz.n_qubits
        self.allow_midcircuit_measurement = initial_kernel.ansatz.allow_midcircuit_measurement

        self.allowed_ops = self._get_allowed_operations()
        self._viewer = None

        action_space = Discrete(len(self.allowed_ops) * self.n_qubits * (self.n_qubits - 1)
                                 * (self.n_features + 1) * self.bw)
        observation_space = Discrete(len(self.allowed_ops) * self.n_qubits * (self.n_qubits - 1)
                                     * (self.n_features + 1) * self.n_operations)

        mdp_info = MDPInfo(observation_space, action_space, gamma=0.99, horizon=100)
        super().__init__(mdp_info)

        self._state = self._serialize_state(0, initial_kernel)
        self.last_reward = None

    def _get_allowed_operations(self):
        if self.mode == "tight":
            return ['II', 'IX', 'IY', 'IZ', 'XX', 'XY', 'XZ', 'YY', 'YZ', 'ZZ']
        return self.initial_kernel.ansatz.get_allowed_operations()

    def _serialize_state(self, n_ops, kernel):
        state = np.concatenate([[n_ops], kernel.to_numpy()], dtype=object)
        if self.convert_to_int:
            for i in range(self.n_operations):
                idx = 5 + i * 5
                if state[idx] < 1:
                    state[idx] *= self.bw
            return state.astype(int)
        return state

    def _deserialize_state(self, array):
        array = array.astype(float)
        if self.convert_to_int:
            for i in range(self.n_operations):
                array[5 + i * 5] /= self.bw
        kernel = PennylaneKernel.from_numpy(
            array[1:], self.n_features, self.n_qubits,
            self.n_operations, self.allow_midcircuit_measurement
        )
        return int(array[0]), kernel

    def unpack_action(self, action):
        a = action[0]
        g_idx = a % len(self.allowed_ops); a //= len(self.allowed_ops)
        w0 = a % self.n_qubits; a //= self.n_qubits
        w1 = a % (self.n_qubits - 1); a //= (self.n_qubits - 1)
        if w1 >= w0:
            w1 += 1
        f = a % (self.n_features + 1); a //= (self.n_features + 1)
        bw = a % self.bw
        bandwidth = (bw + 1) if self.convert_to_int else float(bw + 1) / self.bw
        return {
            'generator': self.allowed_ops[g_idx],
            'wires': [w0, w1],
            'feature': f,
            'bandwidth': bandwidth
        }

    def step(self, action):
        op = self.unpack_action(action)
        n_op, kernel = self._deserialize_state(self._state)
        kernel.ansatz.change_operation(n_op, op['feature'], op['wires'], op['generator'], op['bandwidth'])
        n_op += 1
        self._state = self._serialize_state(n_op, kernel)

        reward = -1 * self.ke.evaluate(kernel, None, self.X, self.y)
        self.last_reward = reward
        absorbing = (n_op == self.n_operations)
        return self._state, reward, absorbing, {}

    def reset(self, state=None):
        if state is None:
            self.initial_kernel.ansatz.initialize_to_identity()
            self._state = self._serialize_state(0, self.initial_kernel)
        else:
            self._state = state
        return self._state

    def render(self):
        n_op, kernel = self._deserialize_state(self._state)
        print(f"{self.last_reward=:2.4f} {n_op=:2d} {kernel=}")


RLKernelEnvironment.register()
