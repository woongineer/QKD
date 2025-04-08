import copy
import numpy as np

from mushroom_rl.algorithms.value import SARSALambda
from mushroom_rl.core import Core, Environment
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.parameters import ExponentialParameter
import matplotlib.pyplot as plt

from kernel import PennylaneKernel, FidelityLossEvaluator
from environment import RLKernelEnvironment


def plot_fidelity_losses(fidelity_losses, filename="fidelity_loss_plot.png"):
    plt.figure(figsize=(6, 4))
    plt.plot(fidelity_losses, marker='o')
    plt.title("Fidelity Loss over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Fidelity Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved fidelity loss plot to '{filename}'")


class ReinforcementLearningOptimizer:
    def __init__(self, initial_kernel: PennylaneKernel, X: np.ndarray, y: np.ndarray,
                 ke: FidelityLossEvaluator, env_mode="wide", bw_possible=10):
        self.initial_kernel = copy.deepcopy(initial_kernel)
        self.X = X
        self.y = y
        self.ke = ke
        self.bw = bw_possible

        self.mdp: RLKernelEnvironment = Environment.make(
            'RLKernelEnvironment',
            initial_kernel=self.initial_kernel,
            X=X, y=y, ke=ke,
            bw_possible=bw_possible,
            convert_to_int=True,
            mode=env_mode
        )

        self.agent = None
        self.core = None

    def optimize(self, initial_episodes=3, n_episodes=100, n_steps_per_fit=1, final_episodes=3):
        epsilon = ExponentialParameter(1.0, 0.995)
        learning_rate = Parameter(.1)
        pi = EpsGreedy(epsilon=epsilon)

        self.agent = SARSALambda(self.mdp.info, pi, learning_rate=learning_rate, lambda_coeff=0.9)
        self.core = Core(self.agent, self.mdp)

        print("Initial MDP and agent setup complete.")
        print(self.mdp)
        print(self.mdp.info)
        print(self.core)

        dataset = self.core.evaluate(n_episodes=initial_episodes, render=False)
        J_before = np.mean(compute_J(dataset, self.mdp.info.gamma))
        print(f"Objective function before learning: {J_before:.4f}")

        # self.core.learn(n_episodes=n_episodes, n_steps_per_fit=n_steps_per_fit, render=True)
        fidelity_losses = []
        for ep in range(n_episodes):
            self.core.learn(n_episodes=1, n_steps_per_fit=n_steps_per_fit, render=False)

            # 현재 kernel 상태 가져오기
            state = self.mdp._state.astype(float)
            for i in range(self.mdp.n_operations):
                state[5 + i * 5] /= self.bw
            kernel = PennylaneKernel.from_numpy(
                state[1:], self.mdp.n_features, self.mdp.n_qubits,
                self.mdp.n_operations, self.mdp.allow_midcircuit_measurement
            )

            # fidelity loss 찍기
            fidelity_loss = self.ke.evaluate(kernel, None, self.X, self.y)
            fidelity_losses.append(fidelity_loss)
            print(f"[Episode {ep + 1}] Fidelity loss: {fidelity_loss:.4f}")

        dataset = self.core.evaluate(n_episodes=final_episodes, render=False)
        J_after = np.mean(compute_J(dataset, self.mdp.info.gamma))
        print(f"Objective function after learning: {J_after:.4f}")

        print("Final state debug:")
        state = self.mdp._state.astype(float)
        for i in range(self.mdp.n_operations):
            state[5 + i * 5] /= self.bw

        kernel = PennylaneKernel.from_numpy(
            state[1:],
            self.mdp.n_features,
            self.mdp.n_qubits,
            self.mdp.n_operations,
            self.mdp.allow_midcircuit_measurement
        )
        plot_fidelity_losses(fidelity_losses, "fidelity_loss_plot.png")
        return kernel
