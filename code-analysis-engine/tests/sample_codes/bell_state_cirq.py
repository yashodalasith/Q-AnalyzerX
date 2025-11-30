import cirq

# Bell state creation
qubits = cirq.LineQubit.range(2)
circuit = cirq.Circuit()

# Create Bell state
circuit.append(cirq.H(qubits[0]))
circuit.append(cirq.CNOT(qubits[0], qubits[1]))

# Measure
circuit.append(cirq.measure(*qubits, key='result'))