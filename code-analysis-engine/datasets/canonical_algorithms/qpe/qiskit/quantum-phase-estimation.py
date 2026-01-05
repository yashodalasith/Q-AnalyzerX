import math
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

# Number of counting qubits
n_count = 3

# Create QPE circuit
qpe = QuantumCircuit(n_count + 1, n_count)

# Prepare eigenstate |1>
qpe.x(n_count)

# Apply Hadamard to counting qubits
for q in range(n_count):
    qpe.h(q)

# Controlled-U operations (U = phase gate)
angle = math.pi / 4
repetitions = 1
for q in range(n_count):
    for _ in range(repetitions):
        qpe.cp(angle, q, n_count)
    repetitions *= 2

# Inverse QFT
qpe.append(QFT(n_count, inverse=True), range(n_count))

# Measurement
for q in range(n_count):
    qpe.measure(q, q)
