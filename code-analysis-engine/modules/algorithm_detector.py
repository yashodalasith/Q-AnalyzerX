"""
Quantum Algorithm Pattern Detection
Identifies specific quantum algorithms from circuit structure
"""
from typing import Dict, List, Set
from models.unified_ast import UnifiedAST, GateType
from models.analysis_result import ProblemType

class QuantumAlgorithmDetector:
    """
    Detects quantum algorithms through pattern matching
    on circuit structure and gate sequences
    """
    
    def __init__(self):
        self.patterns = {
            'grover': self._detect_grover,
            'qft': self._detect_qft,
            'vqe': self._detect_vqe,
            'qaoa': self._detect_qaoa,
            'shor': self._detect_shor,
            'phase_estimation': self._detect_phase_estimation,
            'amplitude_amplification': self._detect_amplitude_amplification
        }
    
    def detect(self, unified_ast: UnifiedAST) -> Dict[str, any]:
        """
        Detect quantum algorithm and problem type
        
        Returns:
            {
                'problem_type': ProblemType,
                'detected_algorithms': List[str],
                'confidence': float,
                'characteristics': Dict
            }
        """
        results = {}
        detected = []
        
        for algo_name, detector_func in self.patterns.items():
            match_result = detector_func(unified_ast)
            if match_result['matched']:
                detected.append({
                    'algorithm': algo_name,
                    'confidence': match_result['confidence'],
                    'evidence': match_result['evidence']
                })
        
        # Determine problem type from detected algorithms
        problem_type = self._map_to_problem_type(detected)
        
        # Overall confidence (max of all detections)
        confidence = max([d['confidence'] for d in detected], default=0.0)
        
        return {
            'problem_type': problem_type,
            'detected_algorithms': [d['algorithm'] for d in detected],
            'confidence': confidence,
            'algorithm_details': detected
        }
    
    def _detect_grover(self, ast: UnifiedAST) -> Dict:
        """
        Detect Grover's Search Algorithm
        
        Pattern:
        1. Hadamard gates on all qubits (superposition)
        2. Oracle (controlled gates)
        3. Diffusion operator (H-X-CZ-X-H pattern)
        4. Repetition of oracle + diffusion
        """
        evidence = []
        score = 0.0
        
        gates = ast.gates
        gate_sequence = [g.gate_type for g in gates]
        
        # Check for initial Hadamards
        h_count_start = sum(1 for g in gates[:ast.total_qubits] if g.gate_type == GateType.H)
        if h_count_start >= ast.total_qubits * 0.8:  # At least 80% of qubits
            evidence.append("Initial superposition with H gates")
            score += 0.3
        
        # Check for oracle pattern (multiple CNOT/CX gates)
        cx_gates = [g for g in gates if g.gate_type in {GateType.CNOT, GateType.CX}]
        if len(cx_gates) >= 2:
            evidence.append(f"Oracle pattern detected ({len(cx_gates)} CX gates)")
            score += 0.2
        
        # Check for diffusion operator pattern (H-X-multi-controlled-Z-X-H)
        if self._has_diffusion_pattern(gate_sequence):
            evidence.append("Diffusion operator detected")
            score += 0.3
        
        # Check for iteration (repeated pattern)
        if self._has_repeated_pattern(gate_sequence):
            evidence.append("Repeated oracle-diffusion pattern")
            score += 0.2
        
        return {
            'matched': score >= 0.5,
            'confidence': min(score, 1.0),
            'evidence': evidence
        }
    
    def _detect_qft(self, ast: UnifiedAST) -> Dict:
        """
        Detect Quantum Fourier Transform
        
        Pattern:
        1. Hadamard gates
        2. Controlled rotation gates (RZ, controlled phase)
        3. SWAP gates for reversing qubit order
        """
        evidence = []
        score = 0.0
        
        gates = ast.gates
        
        # Count Hadamards
        h_count = sum(1 for g in gates if g.gate_type == GateType.H)
        if h_count >= ast.total_qubits * 0.5:
            evidence.append(f"Multiple Hadamard gates ({h_count})")
            score += 0.3
        
        # Count rotation gates
        rotation_gates = [g for g in gates if g.gate_type in {GateType.RZ, GateType.RY}]
        if len(rotation_gates) >= ast.total_qubits:
            evidence.append(f"Rotation gates present ({len(rotation_gates)})")
            score += 0.3
        
        # Check for SWAP gates (qubit reordering)
        swap_count = sum(1 for g in gates if g.gate_type == GateType.SWAP)
        if swap_count >= ast.total_qubits // 2:
            evidence.append(f"SWAP gates for qubit reordering ({swap_count})")
            score += 0.4
        
        return {
            'matched': score >= 0.6,
            'confidence': min(score, 1.0),
            'evidence': evidence
        }
    
    def _detect_vqe(self, ast: UnifiedAST) -> Dict:
        """
        Detect Variational Quantum Eigensolver
        
        Pattern:
        1. Parameterized rotation gates (RX, RY, RZ)
        2. Entangling gates (CNOT)
        3. Measurement
        4. Typically shallow circuits
        """
        evidence = []
        score = 0.0
        
        gates = ast.gates
        
        # Check for parameterized gates
        param_gates = [g for g in gates if g.gate_type in {GateType.RX, GateType.RY, GateType.RZ}]
        if len(param_gates) >= ast.total_qubits:
            evidence.append(f"Parameterized gates ({len(param_gates)})")
            score += 0.4
        
        # Check for entangling layer
        if ast.has_entanglement():
            evidence.append("Entangling layer present")
            score += 0.3
        
        # Check for measurements
        if len(ast.measurements) > 0:
            evidence.append("Circuit includes measurements")
            score += 0.2
        
        # Check circuit depth (VQE typically has shallow circuits)
        depth = ast.calculate_circuit_depth()
        if depth < ast.total_qubits * 3:
            evidence.append(f"Shallow circuit (depth={depth})")
            score += 0.1
        
        return {
            'matched': score >= 0.5,
            'confidence': min(score, 1.0),
            'evidence': evidence
        }
    
    def _detect_qaoa(self, ast: UnifiedAST) -> Dict:
        """
        Detect Quantum Approximate Optimization Algorithm
        
        Pattern similar to VQE but with:
        1. Mixing layer (RX gates)
        2. Problem Hamiltonian layer (ZZ interactions)
        3. Alternating layers
        """
        evidence = []
        score = 0.0
        
        gates = ast.gates
        
        # Check for RX gates (mixing layer)
        rx_gates = [g for g in gates if g.gate_type == GateType.RX]
        if len(rx_gates) >= ast.total_qubits:
            evidence.append(f"Mixing layer with RX gates ({len(rx_gates)})")
            score += 0.3
        
        # Check for ZZ interactions (CZ or CNOT-RZ-CNOT)
        cz_gates = [g for g in gates if g.gate_type == GateType.CZ]
        if len(cz_gates) > 0:
            evidence.append(f"Problem Hamiltonian layer (CZ gates: {len(cz_gates)})")
            score += 0.4
        
        # Check for layered structure
        if self._has_layered_pattern(gates):
            evidence.append("Alternating layer structure detected")
            score += 0.3
        
        return {
            'matched': score >= 0.6,
            'confidence': min(score, 1.0),
            'evidence': evidence
        }
    
    def _detect_shor(self, ast: UnifiedAST) -> Dict:
        """
        Detect Shor's Algorithm
        
        Pattern:
        1. QFT (see _detect_qft)
        2. Modular exponentiation (controlled operations)
        3. Inverse QFT
        """
        evidence = []
        score = 0.0
        
        # Check for QFT pattern
        qft_result = self._detect_qft(ast)
        if qft_result['matched']:
            evidence.extend(qft_result['evidence'])
            score += 0.5
        
        # Check for controlled operations (modular exponentiation)
        controlled_gates = [g for g in ast.gates if g.is_controlled]
        if len(controlled_gates) >= ast.total_qubits:
            evidence.append(f"Modular exponentiation ({len(controlled_gates)} controlled ops)")
            score += 0.3
        
        # Shor requires many qubits
        if ast.total_qubits >= 4:
            evidence.append(f"Sufficient qubits for factorization ({ast.total_qubits})")
            score += 0.2
        
        return {
            'matched': score >= 0.7,
            'confidence': min(score, 1.0),
            'evidence': evidence
        }
    
    def _detect_phase_estimation(self, ast: UnifiedAST) -> Dict:
        """Detect Quantum Phase Estimation"""
        evidence = []
        score = 0.0
        
        # Similar to QFT but with controlled-U operations
        qft_result = self._detect_qft(ast)
        if qft_result['matched']:
            score += 0.4
            evidence.append("Contains QFT")
        
        # Check for controlled unitaries
        controlled_gates = [g for g in ast.gates if g.is_controlled]
        if len(controlled_gates) >= ast.total_qubits // 2:
            evidence.append(f"Controlled unitary operations ({len(controlled_gates)})")
            score += 0.6
        
        return {
            'matched': score >= 0.6,
            'confidence': min(score, 1.0),
            'evidence': evidence
        }
    
    def _detect_amplitude_amplification(self, ast: UnifiedAST) -> Dict:
        """Detect Amplitude Amplification (generalized Grover)"""
        # Very similar to Grover
        grover_result = self._detect_grover(ast)
        return {
            'matched': grover_result['matched'],
            'confidence': grover_result['confidence'] * 0.9,  # Slightly lower confidence
            'evidence': grover_result['evidence']
        }
    
    # Helper methods
    
    def _has_diffusion_pattern(self, gate_sequence: List[GateType]) -> bool:
        """Check for H-X-CZ-X-H pattern"""
        pattern = [GateType.H, GateType.X, GateType.CZ, GateType.X, GateType.H]
        return self._find_subsequence(gate_sequence, pattern)
    
    def _has_repeated_pattern(self, gate_sequence: List[GateType]) -> bool:
        """Check if there's a repeated pattern in gate sequence"""
        n = len(gate_sequence)
        for length in range(3, n // 2 + 1):
            pattern = gate_sequence[:length]
            if gate_sequence[length:2*length] == pattern:
                return True
        return False
    
    def _has_layered_pattern(self, gates: List) -> bool:
        """Check for alternating layers (e.g., all RX then all CZ)"""
        if len(gates) < 4:
            return False
        
        # Simple check: gates of same type clustered together
        prev_type = gates[0].gate_type
        transitions = 0
        for gate in gates[1:]:
            if gate.gate_type != prev_type:
                transitions += 1
                prev_type = gate.gate_type
        
        # If transitions > 2, likely has layered structure
        return transitions >= 2
    
    def _find_subsequence(self, sequence: List, pattern: List) -> bool:
        """Check if pattern exists as subsequence"""
        n, m = len(sequence), len(pattern)
        for i in range(n - m + 1):
            if sequence[i:i+m] == pattern:
                return True
        return False
    
    def _map_to_problem_type(self, detected: List[Dict]) -> ProblemType:
        """Map detected algorithms to problem types"""
        if not detected:
            return ProblemType.UNKNOWN
        
        # Get highest confidence algorithm
        best = max(detected, key=lambda x: x['confidence'])
        algo = best['algorithm']
        
        mapping = {
            'grover': ProblemType.SEARCH,
            'amplitude_amplification': ProblemType.SEARCH,
            'vqe': ProblemType.OPTIMIZATION,
            'qaoa': ProblemType.OPTIMIZATION,
            'shor': ProblemType.FACTORIZATION,
            'qft': ProblemType.SIMULATION,
            'phase_estimation': ProblemType.SIMULATION
        }
        
        return mapping.get(algo, ProblemType.UNKNOWN)