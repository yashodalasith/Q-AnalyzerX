"""
Algorithm Validators
Validates quantum algorithms for textbook correctness
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.unified_ast import UnifiedAST, GateType, QuantumGateNode
from modules.language_detector import LanguageDetector, SupportedLanguage
from modules.ast_builder import ASTBuilder

@dataclass
class ValidationResult:
    """Result of algorithm validation"""
    is_valid: bool
    algorithm: str
    confidence: float
    violations: List[str]
    requirements_met: Dict[str, bool]
    metadata: Dict

class AlgorithmValidator:
    """Base class for algorithm validators"""
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.ast_builder = ASTBuilder()
    
    def parse_code(self, code: str) -> Optional[UnifiedAST]:
        """Parse code to unified AST"""
        try:
            detected_lang = self.language_detector.detect(code)
            if detected_lang == SupportedLanguage.UNKNOWN:
                return None
            
            return self.ast_builder.build(code, detected_lang)
        except Exception:
            return None
    
    def validate(self, code: str) -> ValidationResult:
        """Validate algorithm - to be implemented by subclasses"""
        raise NotImplementedError

class BernsteinVaziraniValidator(AlgorithmValidator):
    """
    Validates Bernstein-Vazirani algorithm
    
    Requirements:
    1. Ancilla qubit in |-> state (X + H)
    2. H on all input qubits (superposition)
    3. Oracle: CX only where secret bit = 1
    4. H on input qubits after oracle
    5. Measurement on input qubits only
    """
    
    def validate(self, code: str) -> ValidationResult:
        ast = self.parse_code(code)
        if not ast:
            return ValidationResult(
                is_valid=False,
                algorithm='bernstein_vazirani',
                confidence=0.0,
                violations=['Failed to parse code'],
                requirements_met={},
                metadata={}
            )
        
        violations = []
        requirements = {}
        
        n_qubits = ast.total_qubits
        
        # Requirement 1: Ancilla in |-> state
        ancilla_idx = n_qubits - 1
        has_ancilla_x = any(
            g.gate_type == GateType.X and ancilla_idx in g.qubits
            for g in ast.gates
        )
        has_ancilla_h = any(
            g.gate_type == GateType.H and ancilla_idx in g.qubits
            for g in ast.gates
        )
        
        ancilla_prepared = has_ancilla_x and has_ancilla_h
        requirements['ancilla_prepared'] = ancilla_prepared
        if not ancilla_prepared:
            violations.append("Ancilla not prepared in |-> state")
        
        # Requirement 2: H on all input qubits
        input_qubits = set(range(n_qubits - 1))
        h_gates_on_input = {
            q for g in ast.gates
            if g.gate_type == GateType.H
            for q in g.qubits
            if q in input_qubits
        }
        
        all_input_superposition = len(h_gates_on_input) == len(input_qubits)
        requirements['input_superposition'] = all_input_superposition
        if not all_input_superposition:
            violations.append("Not all input qubits in superposition")
        
        # Requirement 3: Oracle structure (CX with ancilla as target)
        oracle_gates = [
            g for g in ast.gates
            if g.gate_type in {GateType.CX, GateType.CNOT}
            and ancilla_idx in g.qubits
        ]
        
        has_oracle = len(oracle_gates) > 0
        requirements['has_oracle'] = has_oracle
        if not has_oracle:
            violations.append("No oracle detected")
        
        # Requirement 4: H after oracle
        # This is hard to verify precisely without ordering, but we check presence
        h_count_input = sum(
            1 for g in ast.gates
            if g.gate_type == GateType.H
            and any(q in input_qubits for q in g.qubits)
        )
        
        # Should have H before and after oracle (2x coverage minimum)
        has_post_oracle_h = h_count_input >= len(input_qubits) * 2
        requirements['post_oracle_hadamard'] = has_post_oracle_h
        if not has_post_oracle_h:
            violations.append("Missing Hadamard gates after oracle")
        
        # Requirement 5: Measurements on input qubits
        has_measurements = ast.measurements and len(ast.measurements) > 0
        requirements['has_measurements'] = has_measurements
        if not has_measurements:
            violations.append("No measurements detected")
        
        # Calculate confidence
        met_count = sum(requirements.values())
        total_req = len(requirements)
        confidence = met_count / total_req if total_req > 0 else 0.0
        
        is_valid = confidence >= 0.8  # At least 80% requirements met
        
        return ValidationResult(
            is_valid=is_valid,
            algorithm='bernstein_vazirani',
            confidence=confidence,
            violations=violations,
            requirements_met=requirements,
            metadata={
                'qubits': n_qubits,
                'total_gates': len(ast.gates),
                'oracle_gates': len(oracle_gates)
            }
        )

class DeutschJozsaValidator(AlgorithmValidator):
    """
    Validates Deutsch-Jozsa algorithm
    
    Requirements:
    1. Ancilla in |-> state
    2. H on all qubits
    3. Oracle (varies by function)
    4. H on input qubits after oracle
    5. Measurement on input qubits
    """
    
    def validate(self, code: str) -> ValidationResult:
        ast = self.parse_code(code)
        if not ast:
            return ValidationResult(
                is_valid=False,
                algorithm='deutsch_jozsa',
                confidence=0.0,
                violations=['Failed to parse code'],
                requirements_met={},
                metadata={}
            )
        
        violations = []
        requirements = {}
        
        n_qubits = ast.total_qubits
        ancilla_idx = n_qubits - 1
        input_qubits = set(range(n_qubits - 1))
        
        # Similar structure to BV
        # Requirement 1: Ancilla preparation
        has_ancilla_x = any(
            g.gate_type == GateType.X and ancilla_idx in g.qubits
            for g in ast.gates
        )
        has_ancilla_h = any(
            g.gate_type == GateType.H and ancilla_idx in g.qubits
            for g in ast.gates
        )
        
        ancilla_prepared = has_ancilla_x and has_ancilla_h
        requirements['ancilla_prepared'] = ancilla_prepared
        if not ancilla_prepared:
            violations.append("Ancilla not prepared")
        
        # Requirement 2: Input superposition
        h_on_input = sum(
            1 for g in ast.gates
            if g.gate_type == GateType.H
            and any(q in input_qubits for q in g.qubits)
        )
        
        has_input_superposition = h_on_input >= len(input_qubits)
        requirements['input_superposition'] = has_input_superposition
        if not has_input_superposition:
            violations.append("Input qubits not in superposition")
        
        # Requirement 3: Oracle exists
        has_oracle = any(
            g.is_controlled for g in ast.gates
        )
        requirements['has_oracle'] = has_oracle
        if not has_oracle:
            violations.append("No oracle detected")
        
        # Requirement 4: Post-oracle Hadamards
        has_post_h = h_on_input >= len(input_qubits) * 2
        requirements['post_oracle_hadamard'] = has_post_h
        if not has_post_h:
            violations.append("Missing post-oracle Hadamards")
        
        # Requirement 5: Measurements
        has_measurements = ast.measurements and len(ast.measurements) > 0
        requirements['has_measurements'] = has_measurements
        if not has_measurements:
            violations.append("No measurements")
        
        confidence = sum(requirements.values()) / len(requirements)
        is_valid = confidence >= 0.8
        
        return ValidationResult(
            is_valid=is_valid,
            algorithm='deutsch_jozsa',
            confidence=confidence,
            violations=violations,
            requirements_met=requirements,
            metadata={'qubits': n_qubits}
        )

class GroverValidator(AlgorithmValidator):
    """
    Validates Grover's Search algorithm
    
    Requirements:
    1. Initial superposition (H on all qubits)
    2. Oracle (marks target state)
    3. Diffusion operator (inversion about average)
    4. Correct structure: H-X-MCZ-X-H
    5. Measurements
    """
    
    def validate(self, code: str) -> ValidationResult:
        ast = self.parse_code(code)
        if not ast:
            return ValidationResult(
                is_valid=False,
                algorithm='grover',
                confidence=0.0,
                violations=['Failed to parse code'],
                requirements_met={},
                metadata={}
            )
        
        violations = []
        requirements = {}
        
        n_qubits = ast.total_qubits
        
        # Requirement 1: Initial superposition
        h_gates = [g for g in ast.gates if g.gate_type == GateType.H]
        has_initial_h = len(h_gates) >= n_qubits
        requirements['initial_superposition'] = has_initial_h
        if not has_initial_h:
            violations.append("Missing initial superposition")
        
        # Requirement 2: Oracle (controlled gates)
        oracle_gates = [g for g in ast.gates if g.is_controlled]
        has_oracle = len(oracle_gates) > 0
        requirements['has_oracle'] = has_oracle
        if not has_oracle:
            violations.append("No oracle detected")
        
        # Requirement 3: Diffusion operator components
        # Should have: H gates, X gates, multi-controlled Z, X gates, H gates
        x_gates = [g for g in ast.gates if g.gate_type == GateType.X]
        has_x_gates = len(x_gates) >= n_qubits
        requirements['has_x_gates'] = has_x_gates
        if not has_x_gates:
            violations.append("Missing X gates in diffusion")
        
        # Multi-controlled gates (Toffoli, MCZ, etc.)
        multi_controlled = [
            g for g in ast.gates
            if g.is_controlled and len(g.control_qubits) >= 2
        ]
        has_multi_controlled = len(multi_controlled) > 0
        requirements['has_diffusion_core'] = has_multi_controlled
        if not has_multi_controlled:
            violations.append("Missing multi-controlled gate in diffusion")
        
        # Requirement 4: Multiple H layers (before and after diffusion)
        has_multiple_h_layers = len(h_gates) >= n_qubits * 2
        requirements['has_multiple_h_layers'] = has_multiple_h_layers
        if not has_multiple_h_layers:
            violations.append("Missing multiple Hadamard layers")
        
        # Requirement 5: Measurements
        has_measurements = ast.measurements and len(ast.measurements) > 0
        requirements['has_measurements'] = has_measurements
        if not has_measurements:
            violations.append("No measurements")
        
        confidence = sum(requirements.values()) / len(requirements)
        is_valid = confidence >= 0.8
        
        return ValidationResult(
            is_valid=is_valid,
            algorithm='grover',
            confidence=confidence,
            violations=violations,
            requirements_met=requirements,
            metadata={
                'qubits': n_qubits,
                'h_gates': len(h_gates),
                'oracle_gates': len(oracle_gates)
            }
        )

class QFTValidator(AlgorithmValidator):
    """
    Validates Quantum Fourier Transform
    
    Requirements:
    1. H gates on all qubits
    2. Controlled phase rotations (CP gates)
    3. Increasing rotation angles (powers of 2)
    4. Optional: SWAP gates at end
    """
    
    def validate(self, code: str) -> ValidationResult:
        ast = self.parse_code(code)
        if not ast:
            return ValidationResult(
                is_valid=False,
                algorithm='qft',
                confidence=0.0,
                violations=['Failed to parse code'],
                requirements_met={},
                metadata={}
            )
        
        violations = []
        requirements = {}
        
        n_qubits = ast.total_qubits
        
        # Requirement 1: H gates
        h_gates = [g for g in ast.gates if g.gate_type == GateType.H]
        has_h_gates = len(h_gates) >= n_qubits
        requirements['has_hadamards'] = has_h_gates
        if not has_h_gates:
            violations.append("Missing Hadamard gates")
        
        # Requirement 2: Controlled phase rotations
        cp_gates = [
            g for g in ast.gates
            if g.gate_type in {GateType.CP, GateType.CRZ}
            or (g.is_controlled and g.parameters)
        ]
        has_cp_gates = len(cp_gates) > 0
        requirements['has_phase_rotations'] = has_cp_gates
        if not has_cp_gates:
            violations.append("Missing controlled phase rotations")
        
        # Requirement 3: Sufficient rotations for QFT
        # QFT needs n(n-1)/2 controlled rotations
        expected_cp = (n_qubits * (n_qubits - 1)) // 2
        has_sufficient_rotations = len(cp_gates) >= expected_cp * 0.5  # At least half
        requirements['sufficient_rotations'] = has_sufficient_rotations
        if not has_sufficient_rotations:
            violations.append("Insufficient phase rotations for QFT")
        
        # Requirement 4: SWAP gates (optional but common)
        swap_gates = [g for g in ast.gates if g.gate_type == GateType.SWAP]
        has_swaps = len(swap_gates) > 0
        requirements['has_swaps'] = has_swaps
        # Note: Not a violation if missing, as it's optional
        
        confidence = sum(requirements.values()) / len(requirements)
        is_valid = confidence >= 0.75  # Slightly lower threshold
        
        return ValidationResult(
            is_valid=is_valid,
            algorithm='qft',
            confidence=confidence,
            violations=violations,
            requirements_met=requirements,
            metadata={
                'qubits': n_qubits,
                'h_gates': len(h_gates),
                'cp_gates': len(cp_gates),
                'swap_gates': len(swap_gates)
            }
        )

class SimonValidator(AlgorithmValidator):
    """
    Validates Simon's algorithm
    
    Requirements:
    1. Two registers (input and output)
    2. H on input register
    3. Oracle (copies pattern)
    4. H on input register after oracle
    5. Measurement on input register
    """
    
    def validate(self, code: str) -> ValidationResult:
        ast = self.parse_code(code)
        if not ast:
            return ValidationResult(
                is_valid=False,
                algorithm='simon',
                confidence=0.0,
                violations=['Failed to parse code'],
                requirements_met={},
                metadata={}
            )
        
        violations = []
        requirements = {}
        
        n_qubits = ast.total_qubits
        
        # Simon needs at least 2 qubits (1 input, 1 output minimum)
        if n_qubits < 2:
            return ValidationResult(
                is_valid=False,
                algorithm='simon',
                confidence=0.0,
                violations=['Insufficient qubits for Simon algorithm'],
                requirements_met={},
                metadata={'qubits': n_qubits}
            )
        
        # Assume n/2 split for input/output registers
        input_size = n_qubits // 2
        input_qubits = set(range(input_size))
        
        # Requirement 1: H on input qubits
        h_on_input = sum(
            1 for g in ast.gates
            if g.gate_type == GateType.H
            and any(q in input_qubits for q in g.qubits)
        )
        
        has_input_h = h_on_input >= input_size
        requirements['input_hadamards'] = has_input_h
        if not has_input_h:
            violations.append("Missing Hadamards on input register")
        
        # Requirement 2: Oracle (CX between registers)
        cx_gates = [
            g for g in ast.gates
            if g.gate_type in {GateType.CX, GateType.CNOT}
        ]
        has_oracle = len(cx_gates) >= input_size
        requirements['has_oracle'] = has_oracle
        if not has_oracle:
            violations.append("Oracle missing or incomplete")
        
        # Requirement 3: Post-oracle H
        has_post_h = h_on_input >= input_size * 2
        requirements['post_oracle_hadamards'] = has_post_h
        if not has_post_h:
            violations.append("Missing post-oracle Hadamards")
        
        # Requirement 4: Measurements
        has_measurements = ast.measurements and len(ast.measurements) > 0
        requirements['has_measurements'] = has_measurements
        if not has_measurements:
            violations.append("No measurements")
        
        confidence = sum(requirements.values()) / len(requirements)
        is_valid = confidence >= 0.75
        
        return ValidationResult(
            is_valid=is_valid,
            algorithm='simon',
            confidence=confidence,
            violations=violations,
            requirements_met=requirements,
            metadata={'qubits': n_qubits, 'input_size': input_size}
        )
    
class QPEValidator(AlgorithmValidator):
    """
    Validates Quantum Phase Estimation (QPE)

    Requirements:
    1. Hadamards on counting register
    2. Controlled unitary operations
    3. Inverse QFT (QFT†)
    4. Measurement
    """

    def validate(self, code: str) -> ValidationResult:
        ast = self.parse_code(code)
        if not ast:
            return ValidationResult(False, "qpe", 0.0, ["Parse failed"], {}, {})

        violations = []
        requirements = {}

        n_qubits = ast.total_qubits

        # 1. Hadamards
        h_gates = [g for g in ast.gates if g.gate_type == GateType.H]
        has_hadamards = len(h_gates) >= n_qubits // 2
        requirements["hadamards"] = has_hadamards
        if not has_hadamards:
            violations.append("Missing Hadamards on counting register")

        # 2. Controlled-U
        controlled_ops = [g for g in ast.gates if g.is_controlled]
        has_controlled_u = len(controlled_ops) > 0
        requirements["controlled_unitary"] = has_controlled_u
        if not has_controlled_u:
            violations.append("No controlled unitary detected")

        # 3. Inverse QFT
        has_qft = any(
            g.gate_type in {GateType.CP, GateType.CRZ}
            for g in ast.gates
        )
        requirements["inverse_qft"] = has_qft
        if not has_qft:
            violations.append("No inverse QFT structure detected")

        # 4. Measurement
        has_measurements = bool(ast.measurements)
        requirements["measurement"] = has_measurements
        if not has_measurements:
            violations.append("Missing measurement")

        confidence = sum(requirements.values()) / len(requirements)

        return ValidationResult(
            is_valid=confidence >= 0.75,
            algorithm="qpe",
            confidence=confidence,
            violations=violations,
            requirements_met=requirements,
            metadata={"qubits": n_qubits}
        )

class ShorValidator(AlgorithmValidator):
    """
    Validates Shor's Algorithm

    Requirements:
    1. QPE structure
    2. Modular exponentiation oracle
    3. Inverse QFT
    """

    def validate(self, code: str) -> ValidationResult:
        ast = self.parse_code(code)
        if not ast:
            return ValidationResult(False, "shor", 0.0, ["Parse failed"], {}, {})

        violations = []
        requirements = {}

        # 1. Controlled modular exponentiation
        controlled_ops = [g for g in ast.gates if g.is_controlled]
        has_controlled = len(controlled_ops) > 2
        requirements["controlled_modexp"] = has_controlled
        if not has_controlled:
            violations.append("Missing controlled modular exponentiation")

        # 2. Inverse QFT
        has_qft = any(
            g.gate_type in {GateType.CP, GateType.CRZ}
            for g in ast.gates
        )
        requirements["inverse_qft"] = has_qft
        if not has_qft:
            violations.append("Missing inverse QFT")

        # 3. Measurement
        has_measurements = bool(ast.measurements)
        requirements["measurement"] = has_measurements
        if not has_measurements:
            violations.append("Missing measurement")

        confidence = sum(requirements.values()) / len(requirements)

        return ValidationResult(
            is_valid=confidence >= 0.7,
            algorithm="shor",
            confidence=confidence,
            violations=violations,
            requirements_met=requirements,
            metadata={"controlled_ops": len(controlled_ops)}
        )

class QAOAValidator(AlgorithmValidator):
    """
    Validates QAOA (structure-only)

    Requirements:
    1. Parameterized gates
    2. Repeated layers
    3. Mixer + cost structure
    """

    def validate(self, code: str) -> ValidationResult:
        ast = self.parse_code(code)
        if not ast:
            return ValidationResult(False, "qaoa", 0.0, ["Parse failed"], {}, {})

        violations = []
        requirements = {}

        # 1. Parameterized gates
        param_gates = [g for g in ast.gates if g.parameters]
        has_params = len(param_gates) > 0
        requirements["parameterized_gates"] = has_params
        if not has_params:
            violations.append("No parameterized gates detected")

        # 2. Repeated structure
        repeated = len(ast.gates) > ast.total_qubits * 4
        requirements["layered_structure"] = repeated
        if not repeated:
            violations.append("Insufficient layered structure")

        # 3. Mixer (RX / H)
        has_mixer = any(
            g.gate_type in {GateType.RX, GateType.H}
            for g in ast.gates
        )
        requirements["mixer"] = has_mixer
        if not has_mixer:
            violations.append("No mixer detected")

        confidence = sum(requirements.values()) / len(requirements)

        return ValidationResult(
            is_valid=confidence >= 0.6,
            algorithm="qaoa",
            confidence=confidence,
            violations=violations,
            requirements_met=requirements,
            metadata={"param_gates": len(param_gates)}
        )

class AlgorithmValidatorFactory:
    """Factory for creating algorithm validators"""
    
    VALIDATORS = {
        "bernstein_vazirani": BernsteinVaziraniValidator,
        "deutsch_jozsa": DeutschJozsaValidator,
        "grover": GroverValidator,
        "qft": QFTValidator,
        "simon": SimonValidator,
        "qpe": QPEValidator,
        "shor": ShorValidator,
        "qaoa": QAOAValidator,
    }
    
    @classmethod
    def get_validator(cls, algorithm: str) -> Optional[AlgorithmValidator]:
        """Get validator for algorithm"""
        validator_class = cls.VALIDATORS.get(algorithm)
        if validator_class:
            return validator_class()
        return None
    
    @classmethod
    def validate_all_algorithms(cls, code: str) -> Dict[str, ValidationResult]:
        """Run all validators on code"""
        results = {}
        for algorithm, validator_class in cls.VALIDATORS.items():
            validator = validator_class()
            results[algorithm] = validator.validate(code)
        return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Test validation
    test_bv_code = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(4, 3)

# Initialize ancilla
qc.x(3)
qc.h(3)

# Superposition
for i in range(3):
    qc.h(i)

# Oracle
qc.cx(0, 3)
qc.cx(1, 3)

# Hadamard
for i in range(3):
    qc.h(i)

# Measurement
qc.measure(range(3), range(3))
    """
    
    print("=" * 80)
    print("ALGORITHM VALIDATOR TEST")
    print("=" * 80)
    
    # Test Bernstein-Vazirani
    bv_validator = BernsteinVaziraniValidator()
    result = bv_validator.validate(test_bv_code)
    
    print(f"\nAlgorithm: {result.algorithm}")
    print(f"Valid: {result.is_valid}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"\nRequirements:")
    for req, met in result.requirements_met.items():
        status = "✅" if met else "❌"
        print(f"  {status} {req}")
    
    if result.violations:
        print(f"\nViolations:")
        for violation in result.violations:
            print(f"  - {violation}")
    
    print("\n" + "=" * 80)
    
    # Test all validators
    print("\nTesting all validators on same code:")
    all_results = AlgorithmValidatorFactory.validate_all_algorithms(test_bv_code)
    
    print(f"\n{'Algorithm':<20} {'Valid':<10} {'Confidence':<15}")
    print("-" * 50)
    for algo, res in sorted(all_results.items()):
        valid_str = "✅ Yes" if res.is_valid else "❌ No"
        print(f"{algo:<20} {valid_str:<10} {res.confidence:<15.2%}")
    
    print("=" * 80)