"""Test language detection"""
from modules.language_detector import LanguageDetector, SupportedLanguage

def test_qiskit_detection():
    detector = LanguageDetector()
    
    qiskit_code = """
from qiskit import QuantumCircuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0,1], [0,1])
    """
    
    result = detector.detect(qiskit_code)
    print(f"✓ Qiskit Detection: {result}")
    assert result["language"] == SupportedLanguage.QISKIT

def test_cirq_detection():
    detector = LanguageDetector()
    
    cirq_code = """
import cirq
qubits = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1])
)
    """
    
    result = detector.detect(cirq_code)
    print(f"✓ Cirq Detection: {result}")
    assert result["language"] == SupportedLanguage.CIRQ

if __name__ == "__main__":
    test_qiskit_detection()
    test_cirq_detection()
    print("\n✅ All tests passed!")