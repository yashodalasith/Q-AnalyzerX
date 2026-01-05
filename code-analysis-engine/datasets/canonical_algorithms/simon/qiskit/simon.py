from qiskit import QuantumCircuit
from qiskit_textbook.tools import simon_oracle

# Hidden bitstring
b = '110'
n = len(b)

# Simon circuit
qc = QuantumCircuit(2*n, n)

# Step 1: Hadamard on input register
qc.h(range(n))

# Step 2: Oracle
qc.compose(simon_oracle(b), inplace=True)

# Step 3: Hadamard again
qc.h(range(n))

# Step 4: Measure input register
qc.measure(range(n), range(n))
