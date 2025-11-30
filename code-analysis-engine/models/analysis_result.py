"""
Analysis Result Models
Output format for Decision Engine
"""
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, Dict, Any

class ProblemType(str, Enum):
    """Type of computational problem"""
    SEARCH = "search"
    OPTIMIZATION = "optimization"
    SIMULATION = "simulation"
    MACHINE_LEARNING = "machine_learning"
    FACTORIZATION = "factorization"
    CRYPTOGRAPHY = "cryptography"
    SAMPLING = "sampling"
    CLASSICAL = "classical"
    UNKNOWN = "unknown"

class TimeComplexity(str, Enum):
    """Algorithm time complexity"""
    CONSTANT = "O(1)"
    LOGARITHMIC = "O(log n)"
    LINEAR = "O(n)"
    LINEARITHMIC = "O(n log n)"
    QUADRATIC = "O(n^2)"
    CUBIC = "O(n^3)"
    EXPONENTIAL = "O(2^n)"
    FACTORIAL = "O(n!)"
    POLYNOMIAL = "O(n^k)"
    QUANTUM_ADVANTAGE = "O(sqrt(n))"  # e.g., Grover
    UNKNOWN = "unknown"

class ClassicalComplexity(BaseModel):
    """Classical code complexity metrics"""
    cyclomatic_complexity: int = Field(..., description="McCabe complexity")
    cognitive_complexity: int = Field(default=0, description="Cognitive complexity")
    time_complexity: TimeComplexity = Field(..., description="Estimated time complexity")
    space_complexity: str = Field(default="O(1)", description="Space complexity")
    loop_count: int = Field(default=0, description="Number of loops")
    conditional_count: int = Field(default=0, description="Number of conditionals")
    function_count: int = Field(default=0, description="Number of functions")
    max_nesting_depth: int = Field(default=0, description="Maximum nesting depth")
    lines_of_code: int = Field(default=0, description="Total lines of code")

class QuantumComplexity(BaseModel):
    """Quantum-specific complexity metrics"""
    qubits_required: int = Field(..., ge=0, description="Number of qubits needed")
    circuit_depth: int = Field(..., ge=0, description="Depth of quantum circuit")
    gate_count: int = Field(..., ge=0, description="Total number of gates")
    single_qubit_gates: int = Field(default=0, description="Single-qubit gate count")
    two_qubit_gates: int = Field(default=0, description="Two-qubit gate count")
    cx_gate_count: int = Field(default=0, description="CNOT/CX gate count")
    cx_gate_ratio: float = Field(..., ge=0.0, le=1.0, description="Ratio of entangling gates")
    measurement_count: int = Field(default=0, description="Number of measurements")
    
    # Quantum characteristics
    superposition_score: float = Field(..., ge=0.0, le=1.0, description="Superposition potential")
    entanglement_score: float = Field(..., ge=0.0, le=1.0, description="Entanglement potential")
    has_superposition: bool = Field(default=False, description="Uses superposition")
    has_entanglement: bool = Field(default=False, description="Uses entanglement")
    
    # Resource estimation
    quantum_volume: Optional[float] = Field(default=None, description="Estimated quantum volume")
    estimated_runtime_ms: Optional[float] = Field(default=None, description="Estimated runtime")

class CodeAnalysisResult(BaseModel):
    """
    Complete code analysis result
    Output format for Decision Engine
    """
    # Language detection
    detected_language: str = Field(..., description="Detected programming language")
    language_confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    
    # Problem classification
    problem_type: ProblemType = Field(..., description="Type of computational problem")
    problem_size: int = Field(..., ge=1, description="Size of the problem")
    
    # Complexity metrics
    classical_metrics: Optional[ClassicalComplexity] = None
    quantum_metrics: Optional[QuantumComplexity] = None
    
    # Decision engine requirements (combined)
    qubits_required: int = Field(..., ge=0, description="Number of qubits (0 for classical)")
    circuit_depth: int = Field(..., ge=0, description="Circuit depth (0 for classical)")
    gate_count: int = Field(..., ge=0, description="Total gates (0 for classical)")
    cx_gate_ratio: float = Field(..., ge=0.0, le=1.0, description="Entangling gate ratio")
    superposition_score: float = Field(..., ge=0.0, le=1.0, description="Superposition score")
    entanglement_score: float = Field(..., ge=0.0, le=1.0, description="Entanglement score")
    time_complexity: TimeComplexity = Field(..., description="Algorithm time complexity")
    memory_requirement_mb: float = Field(..., ge=0, description="Memory requirement in MB")
    
    # Additional metadata
    is_quantum_eligible: bool = Field(default=False, description="Eligible for quantum execution")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall confidence")
    analysis_notes: str = Field(default="", description="Additional notes")
    
    # Algorithm detection (will be added later)
    detected_algorithms: list = Field(default_factory=list, description="Detected quantum algorithms")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detected_language": "qiskit",
                "language_confidence": 0.95,
                "problem_type": "search",
                "problem_size": 4,
                "qubits_required": 4,
                "circuit_depth": 12,
                "gate_count": 15,
                "cx_gate_ratio": 0.33,
                "superposition_score": 1.0,
                "entanglement_score": 0.75,
                "time_complexity": "O(sqrt(n))",
                "memory_requirement_mb": 0.5,
                "is_quantum_eligible": True,
                "confidence_score": 0.9
            }
        }