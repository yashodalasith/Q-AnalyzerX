from qiskit import QuantumCircuit

# Problem parameters
n = 3
s = "011"

# Circuit: n input qubits + 1 auxiliary, n classical bits
bv_circuit = QuantumCircuit(n + 1, n)

# Prepare auxiliary qubit in |-> state
bv_circuit.h(n)
bv_circuit.z(n)

# Apply Hadamards to input qubits
for i in range(n):
    bv_circuit.h(i)

# Oracle
s = s[::-1]  # reverse for Qiskit qubit ordering
for q in range(n):
    if s[q] == "1":
        bv_circuit.cx(q, n)

# Apply Hadamards again
for i in range(n):
    bv_circuit.h(i)

# Measurement
for i in range(n):
    bv_circuit.measure(i, i)
