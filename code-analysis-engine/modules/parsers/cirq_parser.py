"""
Cirq Parser - Parses Google Cirq quantum circuits
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
            'CNOT': GateType.CNOT,
            'CX': GateType.CX,
            'CZ': GateType.CZ,
            'SWAP': GateType.SWAP,
            'TOFFOLI': GateType.TOFFOLI,
            'measure': GateType.MEASURE
        }
    
    def parse(self, code: str) -> Dict[str, Any]:
        """Parse Cirq code"""
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
    
    def extract_imports(self) -> List[str]:
        """Extract Cirq imports"""
        imports = []
        for line in self.lines:
            if 'import cirq' in line or 'from cirq' in line:
                imports.append(line.strip())
        return imports
    
    def extract_registers(self) -> Dict[str, Any]:
        """Extract qubit declarations"""
        quantum_regs = []
        
        # Patterns: cirq.LineQubit.range(n) or cirq.GridQubit(row, col)
        line_qubit_pattern = r'cirq\.LineQubit\.range\s*\(\s*(\d+)\s*\)'
        grid_qubit_pattern = r'cirq\.GridQubit'
        
        total_qubits = 0
        for i, line in enumerate(self.lines):
            line_match = re.search(line_qubit_pattern, line)
            if line_match:
                n_qubits = int(line_match.group(1))
                total_qubits += n_qubits
                quantum_regs.append(
                    QuantumRegisterNode(name='qubits', size=n_qubits, line_number=i+1)
                )
            
            if re.search(grid_qubit_pattern, line):
                # Count grid qubits
                matches = re.findall(r'GridQubit\s*\(\s*\d+\s*,\s*\d+\s*\)', line)
                total_qubits += len(matches)
        
        if total_qubits > 0 and not quantum_regs:
            quantum_regs.append(
                QuantumRegisterNode(name='qubits', size=total_qubits, line_number=0)
            )
        
        return {'quantum': quantum_regs, 'classical': []}
    
    def extract_quantum_operations(self) -> List[QuantumGateNode]:
        """Extract quantum gates from Cirq circuits"""
        gates = []
        
        # Pattern: cirq.GATE(qubit) or cirq.GATE.on(qubit)
        gate_pattern = r'cirq\.(\w+)(?:\.on)?\s*\(\s*([^)]+)\s*\)'
        
        for i, line in enumerate(self.lines):
            matches = re.finditer(gate_pattern, line)
            for match in matches:
                gate_name = match.group(1).upper()
                
                if gate_name in self.gate_mapping:
                    # Simplified qubit extraction
                    qubits_arg = match.group(2)
                    # Count qubits (simplified)
                    qubit_count = qubits_arg.count('qubit') or 1
                    
                    gate_type = self.gate_mapping[gate_name]
                    is_controlled = gate_type in {GateType.CNOT, GateType.CX, GateType.CZ}
                    
                    gates.append(QuantumGateNode(
                        gate_type=gate_type,
                        qubits=list(range(qubit_count)),
                        is_controlled=is_controlled,
                        line_number=i+1
                    ))
        
        return gates
    
    def extract_measurements(self) -> List[MeasurementNode]:
        """Extract measurement operations"""
        measurements = []
        
        # Pattern: cirq.measure(qubits)
        measure_pattern = r'cirq\.measure\s*\('
        
        for i, line in enumerate(self.lines):
            if re.search(measure_pattern, line):
                measurements.append(MeasurementNode(
                    quantum_register='qubits',
                    classical_register='measurements',
                    qubit_indices=[],
                    classical_indices=[],
                    line_number=i+1
                ))
        
        return measurements