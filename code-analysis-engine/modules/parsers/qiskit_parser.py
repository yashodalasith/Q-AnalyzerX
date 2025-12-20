"""
Qiskit Parser - Parses Qiskit quantum circuits (AST-based, robust)
"""
import re
import ast
from typing import Dict, Any, List

from .base_parser import BaseParser
from models.unified_ast import (
    QuantumRegisterNode,
    ClassicalRegisterNode,
    QuantumGateNode,
    MeasurementNode,
    GateType
)


class QiskitParser(BaseParser):
    """Parser for Qiskit code"""

    def __init__(self):
        super().__init__()

        # Comprehensive gate mapping
        self.gate_mapping = {
            # Single qubit
            'h': GateType.H,
            'x': GateType.X,
            'y': GateType.Y,
            'z': GateType.Z,
            's': GateType.S,
            't': GateType.T,
            'rx': GateType.RX,
            'ry': GateType.RY,
            'rz': GateType.RZ,

            # Two+ qubit
            'cx': GateType.CX,
            'cnot': GateType.CNOT,
            'cz': GateType.CZ,
            'swap': GateType.SWAP,
            'ccx': GateType.TOFFOLI,
            'toffoli': GateType.TOFFOLI,

            # Controlled / parameterized
            'ch': GateType.CH,
            'cy': GateType.CY,
            'cp': GateType.CP,
            'crx': GateType.CRX,
            'cry': GateType.CRY,
            'crz': GateType.CRZ,
            'cswap': GateType.CSWAP,
            'cu': GateType.CU,
            'cu1': GateType.CU1,
            'cu3': GateType.CU3,

            # Other ops
            'barrier': GateType.BARRIER,
            'reset': GateType.RESET,
            'measure': GateType.MEASURE
        }

    # ------------------------------------------------------------------ #
    # Main entry
    # ------------------------------------------------------------------ #

    def parse(self, code: str) -> Dict[str, Any]:
        self.code = code
        self.lines = code.splitlines()

        return {
            'imports': self.extract_imports(),
            'registers': self.extract_registers(),
            'gates': self.extract_quantum_operations(),
            'measurements': self.extract_measurements(),
            'functions': self.extract_functions(code),
            'metadata': {
                'lines_of_code': self.count_lines(code),
                'loop_count': self.count_loops(code),
                'conditional_count': self.count_conditionals(code),
                'nesting_depth': self.calculate_nesting_depth(code)
            }
        }

    # ------------------------------------------------------------------ #
    # Imports
    # ------------------------------------------------------------------ #

    def extract_imports(self) -> List[str]:
        imports = []
        patterns = [
            r'import\s+qiskit',
            r'from\s+qiskit\s+import\s+.+',
            r'from\s+qiskit\.\w+\s+import\s+.+'
        ]
        for line in self.lines:
            if any(re.search(p, line) for p in patterns):
                imports.append(line.strip())
        return imports

    # ------------------------------------------------------------------ #
    # Registers (regex is sufficient here)
    # ------------------------------------------------------------------ #

    def extract_registers(self) -> Dict[str, Any]:
        quantum_regs = []
        classical_regs = []

        qreg_pattern = r'QuantumRegister\s*\(\s*(\d+)(?:\s*,\s*[\'"](\w+)[\'"])?'
        creg_pattern = r'ClassicalRegister\s*\(\s*(\d+)(?:\s*,\s*[\'"](\w+)[\'"])?'
        circuit_pattern = r'QuantumCircuit\s*\(\s*(\d+)(?:\s*,\s*(\d+))?'

        for i, line in enumerate(self.lines):
            if m := re.search(qreg_pattern, line):
                quantum_regs.append(
                    QuantumRegisterNode(
                        name=m.group(2) or f'q{len(quantum_regs)}',
                        size=int(m.group(1)),
                        line_number=i + 1
                    )
                )

            if m := re.search(creg_pattern, line):
                classical_regs.append(
                    ClassicalRegisterNode(
                        name=m.group(2) or f'c{len(classical_regs)}',
                        size=int(m.group(1)),
                        line_number=i + 1
                    )
                )

            if m := re.search(circuit_pattern, line):
                quantum_regs.append(
                    QuantumRegisterNode(
                        name='q',
                        size=int(m.group(1)),
                        line_number=i + 1
                    )
                )

                if m.group(2):
                    classical_regs.append(
                        ClassicalRegisterNode(
                            name='c',
                            size=int(m.group(2)),
                            line_number=i + 1
                        )
                    )


        return {'quantum': quantum_regs, 'classical': classical_regs}

    # ------------------------------------------------------------------ #
    # AST utilities
    # ------------------------------------------------------------------ #

    def _is_qubit_like(self, node: ast.AST) -> bool:
        return isinstance(node, (ast.Constant, ast.Name, ast.Subscript, ast.List, ast.Tuple, ast.Call))

    # ------------------------------------------------------------------ #
    # Quantum gates (AST-based)
    # ------------------------------------------------------------------ #

    def extract_quantum_operations(self) -> List[QuantumGateNode]:
        gates = []

        try:
            tree = ast.parse(self.code)
        except SyntaxError:
            return gates

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue

            gate_name = node.func.attr.lower()
            if gate_name not in self.gate_mapping:
                continue

            qubit_args = [a for a in node.args if self._is_qubit_like(a)]
            if not qubit_args:
                continue

            gate_type = self.gate_mapping[gate_name]
            qubits = list(range(len(qubit_args)))

            controlled_gates = {
                'ch', 'cy', 'cp', 'crx', 'cry', 'crz',
                'cswap', 'cu', 'cu1', 'cu3'
            }

            is_controlled = gate_name in controlled_gates
            control_qubits = qubits[:-1] if is_controlled else []
            target_qubits = [qubits[-1]] if is_controlled else qubits

            gates.append(
                QuantumGateNode(
                    gate_type=gate_type,
                    qubits=target_qubits,
                    control_qubits=control_qubits,
                    is_controlled=is_controlled,
                    line_number=node.lineno
                )
            )

        return gates

    # ------------------------------------------------------------------ #
    # Measurements (AST-based)
    # ------------------------------------------------------------------ #

    def extract_measurements(self) -> List[MeasurementNode]:
        measurements = []

        try:
            tree = ast.parse(self.code)
        except SyntaxError:
            return measurements

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue

            name = node.func.attr

            if name == 'measure':
                measurements.append(
                    MeasurementNode(
                        quantum_register='q',
                        classical_register='c',
                        qubit_indices=[],
                        classical_indices=[],
                        line_number=node.lineno
                    )
                )

            elif name == 'measure_all':
                measurements.append(
                    MeasurementNode(
                        quantum_register='ALL',
                        classical_register='ALL',
                        qubit_indices=[],
                        classical_indices=[],
                        line_number=node.lineno
                    )
                )

        return measurements