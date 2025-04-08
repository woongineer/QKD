import numpy as np
import itertools
import pennylane as qml
from enum import Enum
import torch


class Operation:
    PAULI_GENERATORS = [a + b for a, b in itertools.product("IXYZ", repeat=2)]
    MEASUREMENT_OPERATIONS = ["IM", "MI"]
    OPERATIONS = PAULI_GENERATORS + MEASUREMENT_OPERATIONS

    def __init__(self, generator: str, wires: list, feature: int, bandwidth: float):
        self.generator = generator
        self.wires = wires
        self.feature = feature
        self.bandwidth = bandwidth

    def to_numpy(self):
        return np.array([
            Operation.OPERATIONS.index(self.generator),
            self.wires[0], self.wires[1],
            self.feature,
            self.bandwidth
        ])

    @staticmethod
    def from_numpy(array):
        return Operation(
            generator=Operation.OPERATIONS[int(array[0])],
            wires=[int(array[1]), int(array[2])],
            feature=int(array[3]),
            bandwidth=float(array[4])
        )

    def __repr__(self):
        return f"-i {self.bandwidth:.2f} * x[{self.feature}] {self.generator}^({self.wires[0]},{self.wires[1]})"


class Ansatz:
    def __init__(self, n_features: int, n_qubits: int, n_operations: int, allow_midcircuit_measurement=False):
        assert n_qubits >= 2
        assert n_features > 0
        assert n_operations > 0
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.n_operations = n_operations
        self.operation_list = [None] * n_operations
        self.allow_midcircuit_measurement = allow_midcircuit_measurement

    def change_operation(self, i, f, w, g, b):
        self.change_feature(i, f)
        self.change_wires(i, w)
        self.change_generators(i, g)
        self.change_bandwidth(i, b)

    def change_feature(self, i, f):
        assert 0 <= i < self.n_operations
        assert 0 <= f <= self.n_features
        self.operation_list[i].feature = f

    def change_wires(self, i, w):
        assert 0 <= i < self.n_operations and len(w) == 2
        assert 0 <= w[0] < self.n_qubits and 0 <= w[1] < self.n_qubits and w[0] != w[1]
        self.operation_list[i].wires = w

    def change_generators(self, i, g):
        assert 0 <= i < self.n_operations
        assert g in Operation.OPERATIONS
        if not self.allow_midcircuit_measurement:
            assert g not in Operation.MEASUREMENT_OPERATIONS
        self.operation_list[i].generator = g

    def change_bandwidth(self, i, b):
        assert 0 <= i < self.n_operations
        self.operation_list[i].bandwidth = b

    def get_allowed_operations(self):
        return Operation.OPERATIONS if self.allow_midcircuit_measurement else Operation.PAULI_GENERATORS

    def initialize_to_identity(self):
        for i in range(self.n_operations):
            self.operation_list[i] = Operation("II", [0, 1], -1, 1.0)

    def initialize_to_random_circuit(self):
        for i in range(self.n_operations):
            g = np.random.choice(self.get_allowed_operations())
            w = np.random.choice(self.n_qubits, 2, replace=False)
            f = np.random.choice(self.n_features + 1)
            b = np.random.uniform(0.0, 1.0)
            self.operation_list[i] = Operation(g, list(w), f, b)

    def to_numpy(self):
        return np.array([op.to_numpy() for op in self.operation_list]).ravel()

    @staticmethod
    def from_numpy(arr, n_feat, n_qubit, n_ops, allow_mid=False, shift_wire=False):
        ans = Ansatz(n_feat, n_qubit, n_ops, allow_mid)
        ans.initialize_to_identity()
        for i in range(n_ops):
            gen = int(np.rint(arr[i*5]))
            wires = [int(np.rint(arr[i*5+1])), int(np.rint(arr[i*5+2]))]
            feat = int(np.rint(arr[i*5+3]))
            bw = round(arr[i*5+4], 4)
            if shift_wire and wires[1] >= wires[0]:
                wires[1] += 1
            if feat == -1:
                feat = n_feat
            ans.change_operation(i, feat, wires, Operation.OPERATIONS[gen], bw)
        return ans

    def __repr__(self):
        return str(self.operation_list)


class KernelType(Enum):
    FIDELITY = 0
    OBSERVABLE = 1

    @staticmethod
    def convert(item):
        return KernelType.FIDELITY if item < 0.5 else KernelType.OBSERVABLE


class PennylaneKernel:
    PAULIS = ['I', 'X', 'Y', 'Z']

    def __init__(self, ansatz: Ansatz, measurement: str, type: KernelType):
        assert len(measurement) == ansatz.n_qubits
        self.ansatz = ansatz
        self.measurement = measurement
        self.type = type
        self.last_probabilities = None

        dev = qml.device("default.qubit", wires=ansatz.n_qubits, shots=None)
        wires = np.arange(ansatz.n_qubits)
        meas_wires = [i for i, p in enumerate(measurement) if p != 'I'] or list(wires)

        @qml.qnode(dev)
        def fidelity_kernel(x1, x2):
            self._apply_ansatz(x1, wires)
            qml.adjoint(self._apply_ansatz)(x2, wires)
            self._apply_measurement_basis()
            return qml.probs(wires=meas_wires)

        @qml.qnode(dev)
        def observable_phi(x):
            self._apply_ansatz(x, wires)
            self._apply_measurement_basis()
            return qml.probs(wires=meas_wires)

        self._fidelity_kernel = fidelity_kernel
        self._observable_phi = observable_phi

    def to_qiskit_circuit(self, x):
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import SparsePauliOp

        qc = QuantumCircuit(self.ansatz.n_qubits)
        params = ParameterVector("x", self.ansatz.n_features)

        for op in self.ansatz.operation_list:
            feature = 1.0 if op.feature == self.ansatz.n_features else x[op.feature]
            operator = SparsePauliOp(op.generator)
            gate = PauliEvolutionGate(operator, time=op.bandwidth * feature)
            if op.generator != "II":
                qc.append(gate, op.wires)
        return qc

    def _apply_ansatz(self, x, wires):
        for op in self.ansatz.operation_list:
            if 'M' not in op.generator:
                theta = np.pi if op.feature == self.ansatz.n_features else x[op.feature]
                qml.PauliRot(op.bandwidth * theta, op.generator, wires=op.wires)
            elif op.generator[0] == "M":
                qml.measure(op.wires[0])
            else:
                qml.measure(op.wires[1])

    def _apply_measurement_basis(self):
        for i, p in enumerate(self.measurement):
            if p == 'X':
                qml.Hadamard(wires=i)
            elif p == 'Y':
                qml.S(wires=i)
                qml.Hadamard(wires=i)

    def kappa(self, x1, x2):
        if self.type == KernelType.FIDELITY:
            probs = self._fidelity_kernel(x1, x2)
            self.last_probabilities = probs
            return probs[0]
        return self.phi(x1) * self.phi(x2)

    def phi(self, x):
        if self.type == KernelType.OBSERVABLE:
            probs = self._observable_phi(x)
            self.last_probabilities = probs
            return sum([(-1)**bin(i).count("1") * p for i, p in enumerate(probs)])
        raise ValueError("phi not available for FIDELITY kernel")

    def to_numpy(self):
        a_np = self.ansatz.to_numpy()
        m_np = np.array([self.PAULIS.index(p) for p in self.measurement])
        t_np = np.array([self.type.value])
        return np.concatenate([a_np, m_np, t_np], dtype=object).ravel()

    @staticmethod
    def from_numpy(array, n_feat, n_qubit, n_ops, allow_mid=False, shift_wire=False):
        assert len(array) == 5 * n_ops + n_qubit + 1
        a_np = array[:n_ops*5]
        m_np = array[n_ops*5:-1]
        t_np = array[-1]
        ansatz = Ansatz.from_numpy(a_np, n_feat, n_qubit, n_ops, allow_mid, shift_wire)
        measurement = "".join(PennylaneKernel.PAULIS[int(i)] for i in m_np)
        type_enum = KernelType.convert(t_np)
        return PennylaneKernel(ansatz, measurement, type_enum)

    def __repr__(self):
        return f"{self.ansatz} -> {self.measurement}"


# class FidelityLossEvaluator:
#     def __init__(self, batch_size=50):
#         self.batch_size = batch_size
#
#     def evaluate(self, kernel: PennylaneKernel, K, X, y):
#         from qiskit import Aer, transpile
#         from qiskit.quantum_info import Statevector
#
#         fidelities = []
#         sim = Aer.get_backend('aer_simulator')
#
#         for _ in range(self.batch_size):
#             i, j = np.random.randint(0, len(X), size=2)
#             qc1 = kernel.to_qiskit_circuit(X[i])
#             qc2 = kernel.to_qiskit_circuit(X[j])
#             qc1 = transpile(qc1, sim)
#             qc2 = transpile(qc2, sim)
#             sv1 = Statevector(qc1)
#             sv2 = Statevector(qc2)
#             f = np.abs(np.vdot(sv1.data, sv2.data))**2
#             fidelities.append(f)
#
#         return 1 - np.mean(fidelities)


class FidelityLossEvaluator:
    def __init__(self, batch_size=50):
        self.batch_size = batch_size

    def evaluate(self, kernel, K, X, y):
        """
        Reward = - MSE loss between predicted fidelity and similarity label (1/0)
        """
        # Define new data sampler (similar to your new_data)
        def sample_similarity_pairs(X, Y, batch_size):
            X1, X2, Y_label = [], [], []
            for _ in range(batch_size):
                i, j = np.random.randint(len(X)), np.random.randint(len(X))
                X1.append(X[i])
                X2.append(X[j])
                Y_label.append(1.0 if Y[i] == Y[j] else 0.0)
            return np.array(X1), np.array(X2), np.array(Y_label)

        # Sample similarity batch
        X1_batch, X2_batch, Y_batch = sample_similarity_pairs(X, y, self.batch_size)

        dev = qml.device('default.qubit', wires=kernel.ansatz.n_qubits)

        @qml.qnode(dev)
        def fidelity_qnode(x1, x2):
            kernel._apply_ansatz(x1, list(range(kernel.ansatz.n_qubits)))
            qml.adjoint(kernel._apply_ansatz)(x2, list(range(kernel.ansatz.n_qubits)))
            return qml.probs(wires=range(kernel.ansatz.n_qubits))

        fidelities = []
        for i in range(self.batch_size):
            probs = fidelity_qnode(X1_batch[i], X2_batch[i])
            fidelities.append(float(probs[0]))  # Fidelity = P(0...0)

        fidelities = torch.tensor(fidelities)
        labels = torch.tensor(Y_batch)

        # MSE loss = similarity loss
        loss = torch.nn.functional.mse_loss(fidelities, labels)

        return loss.item()  # RL에서는 reward = - loss 로 사용됨
