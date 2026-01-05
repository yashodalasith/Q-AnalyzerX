import numpy as np
from qiskit import QuantumCircuit


# -----------------------------
# Helper: Inverse Quantum Fourier Transform
# -----------------------------
def qft_dagger(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for i in range(n // 2):
        qc.swap(i, n - i - 1)

    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi / (2 ** (j - m)), m, j)
        qc.h(j)

    qc.name = "QFT†"
    return qc


# -----------------------------
# Helper: Controlled modular exponentiation (mod 15)
# -----------------------------
def c_amod15(a: int, power: int):
    if a not in [2, 4, 7, 8, 11, 13]:
        raise ValueError("Invalid a for mod-15 demo")

    U = QuantumCircuit(4)

    for _ in range(power):
        if a in [2, 13]:
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)

        if a in [7, 8]:
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)

        if a in [4, 11]:
            U.swap(1, 3)
            U.swap(0, 2)

        if a in [7, 11, 13]:
            for q in range(4):
                U.x(q)

    U = U.to_gate()
    U.name = f"{a}^{power} mod 15"
    return U.control()


# -----------------------------
# Shor's Order Finding Circuit
# -----------------------------
def shor_order_finding(a: int = 7, n_count: int = 8) -> QuantumCircuit:
    qc = QuantumCircuit(n_count + 4, n_count)

    # Initialize counting qubits |+>
    for q in range(n_count):
        qc.h(q)

    # Initialize auxiliary register |1>
    qc.x(n_count)

    # Controlled modular exponentiation
    for q in range(n_count):
        qc.append(
            c_amod15(a, 2 ** q),
            [q] + [i + n_count for i in range(4)]
        )

    # Inverse QFT
    qc.append(qft_dagger(n_count), range(n_count))

    # Measurement
    qc.measure(range(n_count), range(n_count))

    return qc


# -----------------------------
# Execution (optional, minimal)
# -----------------------------
if __name__ == "__main__":
    circuit = shor_order_finding()
