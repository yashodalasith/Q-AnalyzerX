"""
Python and Q# Parsers
"""
import re
import ast
from typing import Dict, Any, List
from .base_parser import BaseParser
from models.unified_ast import QuantumRegisterNode, QuantumGateNode, GateType

class PythonParser(BaseParser):
    """Parser for plain Python code (non-quantum)"""
    
    def parse(self, code: str) -> Dict[str, Any]:
        """Parse Python code"""
        self.code = code
        self.lines = code.split('\n')
        
        return {
            'imports': self.extract_imports(),
            'registers': {'quantum': [], 'classical': []},
            'gates': [],
            'measurements': [],
            'functions': self.extract_functions(code),
            'metadata': {
                'lines_of_code': self.count_lines(code),
                'loop_count': self.count_loops(code),
                'conditional_count': self.count_conditionals(code),
                'nesting_depth': self.calculate_nesting_depth(code)
            }
        }
    
    def extract_imports(self) -> List[str]:
        """Extract Python imports"""
        imports = []
        for line in self.lines:
            if line.strip().startswith(('import ', 'from ')):
                imports.append(line.strip())
        return imports
    
    def extract_quantum_operations(self) -> List[QuantumGateNode]:
        """No quantum operations in plain Python"""
        return []
    
    def extract_registers(self) -> Dict[str, Any]:
        """No quantum registers in plain Python"""
        return {'quantum': [], 'classical': []}


class QSharpParser(BaseParser):
    """Parser for Q# (Microsoft Quantum Development Kit)"""
    
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
            'SWAP': GateType.SWAP,
            'Measure': GateType.MEASURE
        }
    
    def parse(self, code: str) -> Dict[str, Any]:
        """Parse Q# code"""
        self.code = code
        self.lines = code.split('\n')
        
        return {
            'imports': self.extract_imports(),
            'registers': self.extract_registers(),
            'gates': self.extract_quantum_operations(),
            'measurements': [],
            'functions': self.extract_qsharp_operations(),
            'metadata': {
                'lines_of_code': self.count_lines(code),
                'loop_count': self.count_loops(code),
                'conditional_count': self.count_conditionals(code),
                'nesting_depth': self.calculate_nesting_depth(code)
            }
        }
    
    def extract_imports(self) -> List[str]:
        """Extract Q# using statements"""
        imports = []
        for line in self.lines:
            if 'using' in line or 'open' in line:
                imports.append(line.strip())
        return imports
    
    def extract_registers(self) -> Dict[str, Any]:
        """Extract Q# qubit allocations"""
        quantum_regs = []
        
        # Pattern: using (qubits = Qubit[n])
        qubit_pattern = r'using\s*\(\s*\w+\s*=\s*Qubit\[(\d+)\]'
        
        for i, line in enumerate(self.lines):
            match = re.search(qubit_pattern, line)
            if match:
                n_qubits = int(match.group(1))
                quantum_regs.append(
                    QuantumRegisterNode(name='qubits', size=n_qubits, line_number=i+1)
                )
        
        return {'quantum': quantum_regs, 'classical': []}
    
    def extract_quantum_operations(self) -> List[QuantumGateNode]:
        """Extract Q# quantum operations"""
        gates = []
        
        # Pattern: GATE(qubit);
        gate_pattern = r'(\w+)\s*\(\s*\w+(?:\[\d+\])?\s*\)'
        
        for i, line in enumerate(self.lines):
            matches = re.finditer(gate_pattern, line)
            for match in matches:
                gate_name = match.group(1)
                
                if gate_name in self.gate_mapping:
                    gate_type = self.gate_mapping[gate_name]
                    is_controlled = gate_type in {GateType.CNOT, GateType.CX}
                    
                    gates.append(QuantumGateNode(
                        gate_type=gate_type,
                        qubits=[0],  # Simplified
                        is_controlled=is_controlled,
                        line_number=i+1
                    ))
        
        return gates
    
    def extract_qsharp_operations(self) -> List[Dict]:
        """Extract Q# operation definitions"""
        operations = []
        op_pattern = r'operation\s+(\w+)\s*\('
        
        for match in re.finditer(op_pattern, self.code):
            operations.append({'name': match.group(1), 'type': 'operation'})
        
        return operations