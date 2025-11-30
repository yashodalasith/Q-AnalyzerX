"""
Unified Abstract Syntax Tree Data Models
Language-agnostic representation of quantum-classical code
"""
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
from enum import Enum

class NodeType(str, Enum):
    """AST Node Types"""
    PROGRAM = "program"
    IMPORT = "import"
    FUNCTION = "function"
    CLASS = "class"
    QUANTUM_CIRCUIT = "quantum_circuit"
    QUANTUM_REGISTER = "quantum_register"
    CLASSICAL_REGISTER = "classical_register"
    QUANTUM_GATE = "quantum_gate"
    MEASUREMENT = "measurement"
    LOOP = "loop"
    CONDITIONAL = "conditional"
    VARIABLE = "variable"
    EXPRESSION = "expression"

class GateType(str, Enum):
    """Quantum Gate Types"""
    # Single-qubit gates
    H = "hadamard"
    X = "pauli_x"
    Y = "pauli_y"
    Z = "pauli_z"
    S = "s_gate"
    T = "t_gate"
    RX = "rotation_x"
    RY = "rotation_y"
    RZ = "rotation_z"
    
    # Two-qubit gates (entangling)
    CNOT = "cnot"
    CX = "cx"
    CZ = "cz"
    SWAP = "swap"
    
    # Multi-qubit gates
    TOFFOLI = "toffoli"
    FREDKIN = "fredkin"
    
    # Other
    MEASURE = "measurement"
    BARRIER = "barrier"
    RESET = "reset"
    CUSTOM = "custom"

class ASTNode(BaseModel):
    """Base AST Node"""
    node_type: NodeType
    name: Optional[str] = None
    line_number: Optional[int] = None
    children: List['ASTNode'] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)

class QuantumRegisterNode(BaseModel):
    """Quantum Register Declaration"""
    name: str
    size: int
    line_number: Optional[int] = None

class ClassicalRegisterNode(BaseModel):
    """Classical Register Declaration"""
    name: str
    size: int
    line_number: Optional[int] = None

class QuantumGateNode(BaseModel):
    """Quantum Gate Operation"""
    gate_type: GateType
    qubits: List[int]  # Qubit indices
    parameters: List[float] = Field(default_factory=list)  # For parameterized gates
    line_number: Optional[int] = None
    is_controlled: bool = False
    control_qubits: List[int] = Field(default_factory=list)

class MeasurementNode(BaseModel):
    """Measurement Operation"""
    quantum_register: str
    classical_register: str
    qubit_indices: List[int]
    classical_indices: List[int]
    line_number: Optional[int] = None

class ControlFlowNode(BaseModel):
    """Loop or Conditional Statement"""
    flow_type: str  # "for", "while", "if", "else"
    condition: Optional[str] = None
    body: List[ASTNode] = Field(default_factory=list)
    line_number: Optional[int] = None

class UnifiedAST(BaseModel):
    """
    Unified Abstract Syntax Tree
    Language-agnostic representation of quantum-classical programs
    """
    source_language: str
    quantum_registers: List[QuantumRegisterNode] = Field(default_factory=list)
    classical_registers: List[ClassicalRegisterNode] = Field(default_factory=list)
    gates: List[QuantumGateNode] = Field(default_factory=list)
    measurements: List[MeasurementNode] = Field(default_factory=list)
    control_flows: List[ControlFlowNode] = Field(default_factory=list)
    imports: List[str] = Field(default_factory=list)
    functions: List[ASTNode] = Field(default_factory=list)
    root: Optional[ASTNode] = None
    
    # Metadata
    total_qubits: int = 0
    total_classical_bits: int = 0
    total_gates: int = 0
    
    def get_gate_types(self) -> Set[GateType]:
        """Get all unique gate types used"""
        return {gate.gate_type for gate in self.gates}
    
    def get_entangling_gates(self) -> List[QuantumGateNode]:
        """Get all entangling (two-qubit) gates"""
        entangling = {GateType.CNOT, GateType.CX, GateType.CZ, 
                     GateType.SWAP, GateType.TOFFOLI, GateType.FREDKIN}
        return [gate for gate in self.gates if gate.gate_type in entangling]
    
    def get_single_qubit_gates(self) -> List[QuantumGateNode]:
        """Get all single-qubit gates"""
        single_qubit = {GateType.H, GateType.X, GateType.Y, GateType.Z,
                       GateType.S, GateType.T, GateType.RX, GateType.RY, GateType.RZ}
        return [gate for gate in self.gates if gate.gate_type in single_qubit]
    
    def calculate_circuit_depth(self) -> int:
        """Calculate circuit depth (naive - sequential gates)"""
        # Simplified: assumes sequential execution
        # Real depth calculation needs gate scheduling
        return len(self.gates)
    
    def has_superposition(self) -> bool:
        """Check if circuit creates superposition"""
        superposition_gates = {GateType.H, GateType.RX, GateType.RY}
        return any(gate.gate_type in superposition_gates for gate in self.gates)
    
    def has_entanglement(self) -> bool:
        """Check if circuit creates entanglement"""
        return len(self.get_entangling_gates()) > 0

# Allow forward references
ASTNode.model_rebuild()