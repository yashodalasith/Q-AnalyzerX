from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# Grover's algorithm for 2 qubits
qr = QuantumRegister(2, 'q')
cr = ClassicalRegister(2, 'c')
qc = QuantumCircuit(qr, cr)

# Initialize
qc.h(0)
qc.h(1)

# Oracle (marking |11>)
qc.cz(0, 1)

# Diffusion operator
qc.h(0)
qc.h(1)
qc.x(0)
qc.x(1)
qc.cz(0, 1)
qc.x(0)
qc.x(1)
qc.h(0)
qc.h(1)

# Measure
qc.measure(qr, cr)