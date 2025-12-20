"""
Cirq Parser - Accurate static parser for Google Cirq circuits
"""
import re
from typing import Dict, Any, List
from .base_parser import BaseParser
from models.unified_ast import (
    QuantumRegisterNode, QuantumGateNode, MeasurementNode, GateType
)

class CirqParser(BaseParser):
    """Parser for Cirq code"""

    def __init__(self):
        super().__init__()
        self.gate_mapping = {
            'H': GateType.H,
            'X': GateType.X,
            'Y': GateType.Y,
            'Z': GateType.Z,
            'S': GateType.S,
            'T': GateType.T,
            'RX': GateType.RX,
            'RY': GateType.RY,
            'RZ': GateType.RZ,
            'CNOT': GateType.CNOT,
            'CX': GateType.CX,
            'CZ': GateType.CZ,
            'SWAP': GateType.SWAP,
            'CCX': GateType.TOFFOLI,
            'TOFFOLI': GateType.TOFFOLI,
        }

    # ------------------------------------------------------------------ #
    # PARSE
    # ------------------------------------------------------------------ #
    def parse(self, code: str) -> Dict[str, Any]:
        self.code = code
        self.lines = code.split('\n')

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
    # IMPORTS
    # ------------------------------------------------------------------ #
    def extract_imports(self) -> List[str]:
        return [
            line.strip()
            for line in self.lines
            if line.strip().startswith(('import cirq', 'from cirq'))
        ]

    # ------------------------------------------------------------------ #
    # REGISTERS (CRITICAL FOR FEATURES)
    # ------------------------------------------------------------------ #
    def extract_registers(self) -> Dict[str, Any]:
        quantum_regs = []
        qubit_names = set()

        patterns = [
            r'cirq\.LineQubit\.range\s*\(\s*(\d+)\s*\)',
            r'cirq\.LineQubit\s*\(\s*(\d+)\s*\)',
            r'cirq\.NamedQubit\s*\(\s*[\'"](\w+)[\'"]\s*\)',
            r'cirq\.GridQubit\s*\(\s*\d+\s*,\s*\d+\s*\)',
        ]

        for i, line in enumerate(self.lines):
            # LineQubit.range(n)
            match = re.search(patterns[0], line)
            if match:
                quantum_regs.append(
                    QuantumRegisterNode(
                        name='qubits',
                        size=int(match.group(1)),
                        line_number=i + 1
                    )
                )
                continue

            # Single LineQubit / NamedQubit / GridQubit
            for p in patterns[1:]:
                matches = re.findall(p, line)
                for _ in matches:
                    qubit_names.add(f"q{len(qubit_names)}")

        if not quantum_regs and qubit_names:
            quantum_regs.append(
                QuantumRegisterNode(
                    name='qubits',
                    size=len(qubit_names),
                    line_number=0
                )
            )

        return {'quantum': quantum_regs, 'classical': []}

    # ------------------------------------------------------------------ #
    # GATES
    # ------------------------------------------------------------------ #
    def extract_quantum_operations(self) -> List[QuantumGateNode]:
        gates = []

        # Matches:
        # cirq.H(q)
        # cirq.H.on(q)
        # cirq.CNOT(q1, q2)
        # cirq.X(q) ** 0.5
        gate_pattern = r'cirq\.([A-Za-z0-9_]+)(?:\.on)?\s*\(([^)]*)\)'

        for i, line in enumerate(self.lines):
            for match in re.finditer(gate_pattern, line):
                raw_name = match.group(1).upper()

                if raw_name not in self.gate_mapping:
                    continue

                args = match.group(2)
                qubit_count = len([a for a in args.split(',') if a.strip()])

                gate_type = self.gate_mapping[raw_name]
                is_controlled = gate_type in {
                    GateType.CNOT, GateType.CX, GateType.CZ, GateType.TOFFOLI
                }

                gates.append(
                    QuantumGateNode(
                        gate_type=gate_type,
                        qubits=list(range(max(1, qubit_count))),
                        is_controlled=is_controlled,
                        line_number=i + 1
                    )
                )

        return gates

    # ------------------------------------------------------------------ #
    # MEASUREMENTS
    # ------------------------------------------------------------------ #
    def extract_measurements(self) -> List[MeasurementNode]:
        measurements = []

        # cirq.measure(q0, q1, key="m")
        pattern = r'cirq\.measure\s*\(([^)]*)\)'

        for i, line in enumerate(self.lines):
            match = re.search(pattern, line)
            if not match:
                continue

            args = match.group(1)
            qubits = [a.strip() for a in args.split(',') if 'key' not in a]

            measurements.append(
                MeasurementNode(
                    quantum_register='qubits',
                    classical_register='measurements',
                    qubit_indices=list(range(len(qubits))),
                    classical_indices=list(range(len(qubits))),
                    line_number=i + 1
                )
            )

        return measurements