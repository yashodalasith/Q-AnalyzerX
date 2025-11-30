"""
Quantum Complexity Analyzer
Calculates quantum-specific metrics
"""
import math
from typing import Dict, Any
from models.unified_ast import UnifiedAST, GateType
from models.analysis_result import QuantumComplexity

class QuantumAnalyzer:
    """Analyzes quantum circuit complexity"""
    
    def __init__(self):
        # Entangling gates
        self.entangling_gates = {
            GateType.CNOT, GateType.CX, GateType.CZ,
            GateType.SWAP, GateType.TOFFOLI, GateType.FREDKIN
        }
        
        # Superposition-creating gates
        self.superposition_gates = {
            GateType.H, GateType.RX, GateType.RY
        }
    
    def analyze(self, unified_ast: UnifiedAST) -> QuantumComplexity:
        """
        Analyze quantum circuit complexity
        
        Args:
            unified_ast: Unified AST representation
            
        Returns:
            QuantumComplexity object
        """
        # Basic counts
        total_gates = len(unified_ast.gates)
        single_qubit_gates = len(unified_ast.get_single_qubit_gates())
        entangling_gates = unified_ast.get_entangling_gates()
        two_qubit_gates = len(entangling_gates)
        
        # CX gate ratio
        cx_gates = len([g for g in entangling_gates 
                       if g.gate_type in {GateType.CNOT, GateType.CX}])
        cx_ratio = cx_gates / max(total_gates, 1)
        
        # Circuit characteristics
        has_superposition = unified_ast.has_superposition()
        has_entanglement = unified_ast.has_entanglement()
        
        # Scores (0.0 to 1.0)
        superposition_score = self.calculate_superposition_score(unified_ast)
        entanglement_score = self.calculate_entanglement_score(unified_ast)
        
        # Circuit depth (simplified)
        circuit_depth = self.calculate_circuit_depth(unified_ast)
        
        # Quantum volume estimation
        quantum_volume = self.estimate_quantum_volume(
            unified_ast.total_qubits, circuit_depth
        )
        
        # Runtime estimation (simplified)
        estimated_runtime = self.estimate_runtime(unified_ast)
        
        return QuantumComplexity(
            qubits_required=unified_ast.total_qubits,
            circuit_depth=circuit_depth,
            gate_count=total_gates,
            single_qubit_gates=single_qubit_gates,
            two_qubit_gates=two_qubit_gates,
            cx_gate_count=cx_gates,
            cx_gate_ratio=cx_ratio,
            measurement_count=len(unified_ast.measurements),
            superposition_score=superposition_score,
            entanglement_score=entanglement_score,
            has_superposition=has_superposition,
            has_entanglement=has_entanglement,
            quantum_volume=quantum_volume,
            estimated_runtime_ms=estimated_runtime
        )
    
    def calculate_superposition_score(self, unified_ast: UnifiedAST) -> float:
        """
        Calculate superposition potential score
        Based on ratio of superposition gates to total gates
        """
        superposition_gate_count = sum(
            1 for gate in unified_ast.gates 
            if gate.gate_type in self.superposition_gates
        )
        
        total_gates = max(len(unified_ast.gates), 1)
        
        # Normalize score (0.0 to 1.0)
        score = min(superposition_gate_count / total_gates, 1.0)
        
        # Boost if Hadamard is used (creates even superposition)
        if any(g.gate_type == GateType.H for g in unified_ast.gates):
            score = min(score * 1.2, 1.0)
        
        return round(score, 3)
    
    def calculate_entanglement_score(self, unified_ast: UnifiedAST) -> float:
        """
        Calculate entanglement potential score
        Based on ratio of entangling gates and qubit connectivity
        """
        entangling_gate_count = len(unified_ast.get_entangling_gates())
        total_gates = max(len(unified_ast.gates), 1)
        
        # Base score from gate ratio
        score = entangling_gate_count / total_gates
        
        # Boost for multiple qubits (more entanglement potential)
        if unified_ast.total_qubits > 2:
            qubit_factor = min(unified_ast.total_qubits / 10, 1.5)
            score *= qubit_factor
        
        # Cap at 1.0
        return min(round(score, 3), 1.0)
    
    def calculate_circuit_depth(self, unified_ast: UnifiedAST) -> int:
        """
        Calculate circuit depth (simplified sequential model)
        Real depth requires gate scheduling and qubit topology
        """
        # Simplified: assume sequential execution
        # In reality, gates on different qubits can be parallel
        
        total_gates = len(unified_ast.gates)
        
        # Rough estimate: depth = gates / parallelism_factor
        parallelism_factor = max(unified_ast.total_qubits // 2, 1)
        estimated_depth = total_gates // parallelism_factor
        
        return max(estimated_depth, total_gates // 3)  # Conservative estimate
    
    def estimate_quantum_volume(self, n_qubits: int, depth: int) -> float:
        """
        Estimate quantum volume
        QV = min(n, d)^2 where n=qubits, d=depth
        """
        if n_qubits == 0 or depth == 0:
            return 0.0
        
        return float(min(n_qubits, depth) ** 2)
    
    def estimate_runtime(self, unified_ast: UnifiedAST) -> float:
        """
        Estimate runtime in milliseconds
        Based on gate count and quantum hardware characteristics
        """
        # Typical gate times (microseconds)
        single_qubit_gate_time = 0.1  # 100 ns
        two_qubit_gate_time = 0.5     # 500 ns
        measurement_time = 1.0        # 1 microsecond
        
        single_qubit_gates = len(unified_ast.get_single_qubit_gates())
        two_qubit_gates = len(unified_ast.get_entangling_gates())
        measurements = len(unified_ast.measurements)
        
        # Calculate total time (in microseconds)
        total_time_us = (
            single_qubit_gates * single_qubit_gate_time +
            two_qubit_gates * two_qubit_gate_time +
            measurements * measurement_time
        )
        
        # Convert to milliseconds
        return round(total_time_us / 1000, 3)
    
    def estimate_memory_requirement(self, n_qubits: int) -> float:
        """
        Estimate memory requirement for simulation
        Classical simulation requires 2^n complex numbers
        """
        if n_qubits == 0:
            return 0.01  # Minimal overhead
        
        # Each state: 2 * 8 bytes (complex128)
        # Total states: 2^n
        bytes_required = 2 ** n_qubits * 16
        
        # Convert to MB
        mb_required = bytes_required / (1024 ** 2)
        
        return round(mb_required, 3)