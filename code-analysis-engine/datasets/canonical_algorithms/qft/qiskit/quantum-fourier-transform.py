from numpy import pi
from qiskit import QuantumCircuit

def qft(circuit, n):
    for j in range(n):
        circuit.h(j)
        for k in range(j+1, n):
            circuit.cp(pi / 2**(k-j), j, k)
    for i in range(n//2):
        circuit.swap(i, n-i-1)
    return circuit

# Example usage
n = 3
qc = QuantumCircuit(n, n)

# Prepare example input |5⟩
qc.x(0)
qc.x(2)

qft(qc, n)

qc.measure(range(n), range(n))
