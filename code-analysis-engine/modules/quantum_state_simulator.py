"""
Quantum State Simulator for Accurate Superposition and Entanglement Scores
Uses state vector simulation to track actual quantum state evolution
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from models.unified_ast import UnifiedAST, QuantumGateNode, GateType
from scipy.stats import entropy

class QuantumStateSimulator:
    """
    Simulates quantum circuits to accurately measure:
    1. Superposition score (based on state entropy)
    2. Entanglement score (based on entanglement entropy)
    3. Actual quantum properties after gate application
    """
    
    def __init__(self, max_qubits: int = 20):
        """
        Initialize simulator
        
        Args:
            max_qubits: Maximum qubits to simulate (limited by memory: 2^n)
        """
        self.max_qubits = max_qubits
        self.state_vector = None
        self.num_qubits = 0
    
    def simulate(self, unified_ast: UnifiedAST) -> Dict[str, float]:
        """
        Simulate the quantum circuit and calculate scores
        
        Returns:
            {
                'superposition_score': float (0.0 to 1.0),
                'entanglement_score': float (0.0 to 1.0),
                'state_entropy': float,
                'max_entanglement_achieved': float
            }
        """
        self.num_qubits = unified_ast.total_qubits
        
        # Check if circuit is too large to simulate
        if self.num_qubits > self.max_qubits:
            # Fall back to heuristic for large circuits
            return self._heuristic_analysis(unified_ast)
        
        # Initialize state to |00...0⟩
        self._initialize_state()
        
        # Track entropy throughout circuit
        entropy_history = [self._calculate_entropy()]
        entanglement_history = [0.0]
        
        # Apply gates sequentially
        for gate in unified_ast.gates:
            self._apply_gate(gate)
            entropy_history.append(self._calculate_entropy())
            entanglement_history.append(self._calculate_entanglement())
        
        # Calculate final scores
        superposition_score = self._normalize_entropy(entropy_history[-1])
        entanglement_score = max(entanglement_history)
        
        return {
            'superposition_score': round(superposition_score, 3),
            'entanglement_score': round(entanglement_score, 3),
            'state_entropy': round(entropy_history[-1], 3),
            'max_entanglement_achieved': round(max(entanglement_history), 3),
            'final_state_purity': round(self._calculate_purity(), 3)
        }
    
    def _initialize_state(self):
        """Initialize quantum state to |00...0⟩"""
        dim = 2 ** self.num_qubits
        self.state_vector = np.zeros(dim, dtype=complex)
        self.state_vector[0] = 1.0  # |00...0⟩
    
    def _apply_gate(self, gate: QuantumGateNode):
        """Apply a quantum gate to the state vector"""
        gate_type = gate.gate_type
        
        # Get target qubits (convert to 0-indexed)
        targets = gate.qubits
        controls = gate.control_qubits
        
        if gate_type == GateType.H:
            self._apply_hadamard(targets[0] if targets else 0)
        
        elif gate_type == GateType.X:
            self._apply_pauli_x(targets[0] if targets else 0)
        
        elif gate_type == GateType.Y:
            self._apply_pauli_y(targets[0] if targets else 0)
        
        elif gate_type == GateType.Z:
            self._apply_pauli_z(targets[0] if targets else 0)
        
        elif gate_type == GateType.S:
            self._apply_s_gate(targets[0] if targets else 0)
        
        elif gate_type == GateType.T:
            self._apply_t_gate(targets[0] if targets else 0)
        
        elif gate_type in {GateType.RX, GateType.RY, GateType.RZ}:
            angle = gate.parameters[0] if gate.parameters else np.pi/4
            if gate_type == GateType.RX:
                self._apply_rx(targets[0] if targets else 0, angle)
            elif gate_type == GateType.RY:
                self._apply_ry(targets[0] if targets else 0, angle)
            else:  # RZ
                self._apply_rz(targets[0] if targets else 0, angle)
        
        elif gate_type in {GateType.CNOT, GateType.CX}:
            if controls and targets:
                self._apply_cnot(controls[0], targets[0])
        
        elif gate_type == GateType.CZ:
            if controls and targets:
                self._apply_cz(controls[0], targets[0])
        
        elif gate_type == GateType.SWAP:
            if len(targets) >= 2:
                self._apply_swap(targets[0], targets[1])
        
        elif gate_type == GateType.TOFFOLI:
            if len(controls) >= 2 and targets:
                self._apply_toffoli(controls[0], controls[1], targets[0])
    
    # Single-qubit gate implementations
    
    def _apply_hadamard(self, qubit: int):
        """Apply Hadamard gate to qubit"""
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        self._apply_single_qubit_gate(qubit, H)
    
    def _apply_pauli_x(self, qubit: int):
        """Apply Pauli-X (NOT) gate"""
        X = np.array([[0, 1], [1, 0]])
        self._apply_single_qubit_gate(qubit, X)
    
    def _apply_pauli_y(self, qubit: int):
        """Apply Pauli-Y gate"""
        Y = np.array([[0, -1j], [1j, 0]])
        self._apply_single_qubit_gate(qubit, Y)
    
    def _apply_pauli_z(self, qubit: int):
        """Apply Pauli-Z gate"""
        Z = np.array([[1, 0], [0, -1]])
        self._apply_single_qubit_gate(qubit, Z)
    
    def _apply_s_gate(self, qubit: int):
        """Apply S gate (phase gate)"""
        S = np.array([[1, 0], [0, 1j]])
        self._apply_single_qubit_gate(qubit, S)
    
    def _apply_t_gate(self, qubit: int):
        """Apply T gate (π/8 gate)"""
        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
        self._apply_single_qubit_gate(qubit, T)
    
    def _apply_rx(self, qubit: int, angle: float):
        """Apply RX rotation gate"""
        cos = np.cos(angle / 2)
        sin = np.sin(angle / 2)
        RX = np.array([[cos, -1j * sin], [-1j * sin, cos]])
        self._apply_single_qubit_gate(qubit, RX)
    
    def _apply_ry(self, qubit: int, angle: float):
        """Apply RY rotation gate"""
        cos = np.cos(angle / 2)
        sin = np.sin(angle / 2)
        RY = np.array([[cos, -sin], [sin, cos]])
        self._apply_single_qubit_gate(qubit, RY)
    
    def _apply_rz(self, qubit: int, angle: float):
        """Apply RZ rotation gate"""
        RZ = np.array([[np.exp(-1j * angle / 2), 0], 
                       [0, np.exp(1j * angle / 2)]])
        self._apply_single_qubit_gate(qubit, RZ)
    
    def _apply_single_qubit_gate(self, qubit: int, gate_matrix: np.ndarray):
        """
        Apply single-qubit gate using tensor product
        State update: |ψ⟩ → (I ⊗ ... ⊗ U ⊗ ... ⊗ I)|ψ⟩
        """
        # Build full gate matrix using tensor product
        full_matrix = 1
        for i in range(self.num_qubits):
            if i == qubit:
                if isinstance(full_matrix, int):
                    full_matrix = gate_matrix
                else:
                    full_matrix = np.kron(full_matrix, gate_matrix)
            else:
                identity = np.eye(2)
                if isinstance(full_matrix, int):
                    full_matrix = identity
                else:
                    full_matrix = np.kron(full_matrix, identity)
        
        # Apply gate
        self.state_vector = full_matrix @ self.state_vector
    
    # Two-qubit gate implementations
    
    def _apply_cnot(self, control: int, target: int):
        """Apply CNOT (Controlled-X) gate"""
        dim = 2 ** self.num_qubits
        new_state = np.copy(self.state_vector)
        
        # CNOT: if control=1, flip target
        for i in range(dim):
            # Check if control qubit is 1
            if (i >> (self.num_qubits - 1 - control)) & 1:
                # Flip target qubit
                flipped = i ^ (1 << (self.num_qubits - 1 - target))
                new_state[flipped] = self.state_vector[i]
                new_state[i] = self.state_vector[i]
        
        self.state_vector = new_state
    
    def _apply_cz(self, control: int, target: int):
        """Apply CZ (Controlled-Z) gate"""
        dim = 2 ** self.num_qubits
        
        # CZ: if both control and target are 1, apply phase
        for i in range(dim):
            control_bit = (i >> (self.num_qubits - 1 - control)) & 1
            target_bit = (i >> (self.num_qubits - 1 - target)) & 1
            if control_bit and target_bit:
                self.state_vector[i] *= -1
    
    def _apply_swap(self, qubit1: int, qubit2: int):
        """Apply SWAP gate"""
        dim = 2 ** self.num_qubits
        new_state = np.copy(self.state_vector)
        
        for i in range(dim):
            # Extract bits
            bit1 = (i >> (self.num_qubits - 1 - qubit1)) & 1
            bit2 = (i >> (self.num_qubits - 1 - qubit2)) & 1
            
            if bit1 != bit2:
                # Swap bits
                swapped = i ^ (1 << (self.num_qubits - 1 - qubit1))
                swapped = swapped ^ (1 << (self.num_qubits - 1 - qubit2))
                new_state[swapped] = self.state_vector[i]
        
        self.state_vector = new_state
    
    def _apply_toffoli(self, control1: int, control2: int, target: int):
        """Apply Toffoli (CCX) gate"""
        dim = 2 ** self.num_qubits
        new_state = np.copy(self.state_vector)
        
        for i in range(dim):
            c1_bit = (i >> (self.num_qubits - 1 - control1)) & 1
            c2_bit = (i >> (self.num_qubits - 1 - control2)) & 1
            
            # If both controls are 1, flip target
            if c1_bit and c2_bit:
                flipped = i ^ (1 << (self.num_qubits - 1 - target))
                new_state[flipped] = self.state_vector[i]
                new_state[i] = self.state_vector[i]
        
        self.state_vector = new_state
    
    # Measurement and analysis methods
    
    def _calculate_entropy(self) -> float:
        """
        Calculate von Neumann entropy of the state
        S = -Σ pᵢ log₂(pᵢ)
        Measures superposition: 0 (pure) to n (max superposition)
        """
        probabilities = np.abs(self.state_vector) ** 2
        # Filter out zero probabilities
        probabilities = probabilities[probabilities > 1e-10]
        return entropy(probabilities, base=2)
    
    def _normalize_entropy(self, entropy_value: float) -> float:
        """
        Normalize entropy to [0, 1] range
        Max entropy = log₂(2^n) = n qubits
        """
        max_entropy = self.num_qubits
        if max_entropy == 0:
            return 0.0
        return min(entropy_value / max_entropy, 1.0)
    
    def _calculate_entanglement(self) -> float:
        """
        Calculate entanglement using Schmidt rank or mutual information
        
        For simplicity, we use a heuristic based on:
        - Purity of reduced density matrices
        - Number of non-zero Schmidt coefficients
        """
        if self.num_qubits < 2:
            return 0.0
        
        # Calculate purity of first qubit's reduced density matrix
        # Entanglement present if purity < 1
        purity = self._calculate_reduced_purity(0)
        
        # Convert purity to entanglement score
        # Pure state (purity=1) → no entanglement (score=0)
        # Maximally mixed (purity=0.5 for single qubit) → max entanglement (score=1)
        entanglement_score = 1.0 - purity
        
        return min(entanglement_score, 1.0)
    
    def _calculate_reduced_purity(self, qubit: int) -> float:
        """
        Calculate purity of reduced density matrix for a qubit
        Tr(ρ²) where ρ is reduced density matrix
        """
        # Reshape state vector into matrix for partial trace
        dim = 2 ** self.num_qubits
        dim_half = 2 ** (self.num_qubits - 1)
        
        # Create density matrix ρ = |ψ⟩⟨ψ|
        density_matrix = np.outer(self.state_vector, np.conj(self.state_vector))
        
        # Partial trace (simplified for first qubit)
        reduced_density = np.zeros((2, 2), dtype=complex)
        
        for i in range(2):
            for j in range(2):
                for k in range(dim_half):
                    idx1 = i * dim_half + k
                    idx2 = j * dim_half + k
                    reduced_density[i, j] += density_matrix[idx1, idx2]
        
        # Calculate purity Tr(ρ²)
        rho_squared = reduced_density @ reduced_density
        purity = np.real(np.trace(rho_squared))
        
        return max(0.0, min(1.0, purity))
    
    def _calculate_purity(self) -> float:
        """Calculate purity of full state: Tr(|ψ⟩⟨ψ|²)"""
        # For pure states, purity = 1
        return np.real(np.sum(np.abs(self.state_vector) ** 4))
    
    def _heuristic_analysis(self, unified_ast: UnifiedAST) -> Dict[str, float]:
        """
        Fallback heuristic analysis for circuits too large to simulate
        Uses gate counting and patterns
        """
        gates = unified_ast.gates
        total_gates = len(gates)
        
        # Superposition heuristic
        superposition_gates = {GateType.H, GateType.RX, GateType.RY}
        sup_count = sum(1 for g in gates if g.gate_type in superposition_gates)
        superposition_score = min(sup_count / max(unified_ast.total_qubits, 1), 1.0)
        
        # Entanglement heuristic
        entangling_gates = {GateType.CNOT, GateType.CX, GateType.CZ, 
                           GateType.SWAP, GateType.TOFFOLI}
        ent_count = sum(1 for g in gates if g.gate_type in entangling_gates)
        entanglement_score = min(ent_count / max(total_gates, 1), 1.0)
        
        # Boost if multiple qubits involved
        if unified_ast.total_qubits > 2:
            entanglement_score = min(entanglement_score * 1.5, 1.0)
        
        return {
            'superposition_score': round(superposition_score, 3),
            'entanglement_score': round(entanglement_score, 3),
            'state_entropy': 0.0,  # Can't calculate without simulation
            'max_entanglement_achieved': round(entanglement_score, 3),
            'final_state_purity': 0.0,
            'note': 'Heuristic analysis (circuit too large to simulate)'
        }


# Example usage
if __name__ == "__main__":
    from models.unified_ast import UnifiedAST, QuantumRegisterNode
    
    # Test case: Bell state (maximally entangled)
    bell_ast = UnifiedAST(
        source_language='qiskit',
        quantum_registers=[QuantumRegisterNode(name='q', size=2, line_number=1)],
        gates=[
            QuantumGateNode(gate_type=GateType.H, qubits=[0], line_number=2),
            QuantumGateNode(gate_type=GateType.CNOT, qubits=[1], 
                          control_qubits=[0], is_controlled=True, line_number=3)
        ],
        total_qubits=2,
        total_gates=2
    )
    
    simulator = QuantumStateSimulator()
    result = simulator.simulate(bell_ast)
    
    print("Bell State Analysis:")
    print(f"Superposition Score: {result['superposition_score']}")
    print(f"Entanglement Score: {result['entanglement_score']}")
    print(f"State Entropy: {result['state_entropy']}")
    print(f"Expected: High superposition and entanglement (~1.0)")