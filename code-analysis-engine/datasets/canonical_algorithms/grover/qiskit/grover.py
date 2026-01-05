from qiskit import QuantumCircuit, Aer, transpile

n = 2
qc = QuantumCircuit(n, n)

# Initialize |s>
qc.h(range(n))

# Oracle: mark |11>
qc.cz(0, 1)

# Diffusion
qc.h(range(n))
qc.z(range(n))
qc.cz(0, 1)
qc.h(range(n))

# Measure
qc.measure(range(n), range(n))
