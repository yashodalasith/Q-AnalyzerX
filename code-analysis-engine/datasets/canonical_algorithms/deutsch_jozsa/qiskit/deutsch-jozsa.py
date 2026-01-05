from qiskit import QuantumCircuit

# Number of input qubits
n = 3

# -------------------------
# Balanced Oracle
# -------------------------
oracle = QuantumCircuit(n + 1)

# Fixed balanced function (e.g., b_str = "101")
b_str = "101"

# Apply X-gates
for qubit in range(n):
    if b_str[qubit] == "1":
        oracle.x(qubit)

# Controlled-NOTs
for qubit in range(n):
    oracle.cx(qubit, n)

# Uncompute X-gates
for qubit in range(n):
    if b_str[qubit] == "1":
        oracle.x(qubit)

# -------------------------
# Deutsch–Jozsa Circuit
# -------------------------
dj_circuit = QuantumCircuit(n + 1, n)

# Prepare |+>^n |-> state
for qubit in range(n):
    dj_circuit.h(qubit)
dj_circuit.x(n)
dj_circuit.h(n)

# Apply oracle
dj_circuit.compose(oracle, inplace=True)

# Apply Hadamards again
for qubit in range(n):
    dj_circuit.h(qubit)

# Measurement
for i in range(n):
    dj_circuit.measure(i, i)
