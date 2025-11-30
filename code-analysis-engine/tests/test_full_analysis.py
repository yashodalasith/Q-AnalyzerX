"""
Full Analysis Test Suite
Tests the complete pipeline
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.language_detector import LanguageDetector, SupportedLanguage
from modules.ast_builder import ASTBuilder
from modules.complexity_analyzer import ComplexityAnalyzer
from modules.quantum_analyzer import QuantumAnalyzer

def test_qiskit_grover():
    """Test Grover's algorithm (Qiskit)"""
    print("\n" + "="*60)
    print("TEST 1: Qiskit Grover's Algorithm")
    print("="*60)
    
    code = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.h(1)
qc.cz(0, 1)
qc.h(0)
qc.h(1)
qc.measure([0,1], [0,1])
    """
    
    # Language detection
    detector = LanguageDetector()
    lang_result = detector.detect(code)
    print(f"✓ Language: {lang_result['language']} ({lang_result['confidence']:.1%} confidence)")
    
    # Build AST
    ast_builder = ASTBuilder()
    unified_ast = ast_builder.build(code, SupportedLanguage.QISKIT)
    print(f"✓ Qubits: {unified_ast.total_qubits}")
    print(f"✓ Gates: {unified_ast.total_gates}")
    print(f"✓ Entangling gates: {len(unified_ast.get_entangling_gates())}")
    
    # Quantum analysis
    q_analyzer = QuantumAnalyzer()
    q_metrics = q_analyzer.analyze(unified_ast)
    print(f"✓ Circuit depth: {q_metrics.circuit_depth}")
    print(f"✓ Superposition score: {q_metrics.superposition_score}")
    print(f"✓ Entanglement score: {q_metrics.entanglement_score}")
    print(f"✓ Quantum volume: {q_metrics.quantum_volume}")

def test_cirq_bell_state():
    """Test Bell state (Cirq)"""
    print("\n" + "="*60)
    print("TEST 2: Cirq Bell State")
    print("="*60)
    
    code = """
import cirq

qubits = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1]),
    cirq.measure(*qubits)
)
    """
    
    detector = LanguageDetector()
    lang_result = detector.detect(code)
    print(f"✓ Language: {lang_result['language']}")
    
    ast_builder = ASTBuilder()
    unified_ast = ast_builder.build(code, SupportedLanguage.CIRQ)
    print(f"✓ Qubits: {unified_ast.total_qubits}")
    print(f"✓ Gates: {unified_ast.total_gates}")
    print(f"✓ Has entanglement: {unified_ast.has_entanglement()}")

def test_openqasm_qft():
    """Test QFT (OpenQASM)"""
    print("\n" + "="*60)
    print("TEST 3: OpenQASM Quantum Fourier Transform")
    print("="*60)
    
    code = """
OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

h q[0];
h q[1];
h q[2];
cx q[0], q[1];
cx q[1], q[2];
measure q -> c;
    """
    
    detector = LanguageDetector()
    lang_result = detector.detect(code)
    print(f"✓ Language: {lang_result['language']}")
    
    ast_builder = ASTBuilder()
    unified_ast = ast_builder.build(code, SupportedLanguage.OPENQASM)
    print(f"✓ Qubits: {unified_ast.total_qubits}")
    print(f"✓ Gates: {unified_ast.total_gates}")
    print(f"✓ Measurements: {len(unified_ast.measurements)}")

def test_python_classical():
    """Test classical Python"""
    print("\n" + "="*60)
    print("TEST 4: Classical Python Code")
    print("="*60)
    
    code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

for i in range(10):
    print(fibonacci(i))
    """
    
    detector = LanguageDetector()
    lang_result = detector.detect(code)
    print(f"✓ Language: {lang_result['language']}")
    
    analyzer = ComplexityAnalyzer()
    parser = ASTBuilder().parsers[SupportedLanguage.PYTHON]
    parsed = parser.parse(code)
    metadata = ASTBuilder().get_metadata(parsed)
    
    metrics = analyzer.analyze(code, metadata)
    print(f"✓ Cyclomatic complexity: {metrics.cyclomatic_complexity}")
    print(f"✓ Time complexity: {metrics.time_complexity.value}")
    print(f"✓ Loops: {metrics.loop_count}")
    print(f"✓ Functions: {metrics.function_count}")

if __name__ == "__main__":
    print("\n🚀 Running Code Analysis Engine Tests\n")
    
    try:
        test_qiskit_grover()
        test_cirq_bell_state()
        test_openqasm_qft()
        test_python_classical()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()