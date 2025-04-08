from data import data_load_and_process
from kernel import Ansatz, KernelType, PennylaneKernel, FidelityLossEvaluator
from optimizer import ReinforcementLearningOptimizer


if __name__ == "__main__":
    n_qubits = 4
    n_operations = 10

    x_train, x_test, y_train, y_test = data_load_and_process(dataset="kmnist", reduction_sz=n_qubits)

    ansatz = Ansatz(n_features=n_qubits, n_qubits=n_qubits, n_operations=n_operations)
    ansatz.initialize_to_identity()
    measurement = "Z" * n_qubits

    kernel = PennylaneKernel(ansatz, measurement, KernelType.FIDELITY)
    evaluator = FidelityLossEvaluator(batch_size=30)

    rl_opt = ReinforcementLearningOptimizer(initial_kernel=kernel, X=x_train, y=y_train,
                                            ke=evaluator, env_mode="wide", bw_possible=5)

    optimized_kernel = rl_opt.optimize(initial_episodes=5, n_episodes=100, n_steps_per_fit=1, final_episodes=10)

    print(f"Optimized kernel:{optimized_kernel}")
