"""
Qiskit Parser - Parses Qiskit quantum circuits
"""
import re
import ast
from typing import Dict, Any, List
from .base_parser import BaseParser
from models.unified_ast import (
    QuantumRegisterNode, ClassicalRegisterNode, 
    QuantumGateNode, MeasurementNode, GateType
)

class QiskitParser(BaseParser):
    """Parser for Qiskit code"""
    
    def __init__(self):
        super().__init__()
        # Map Qiskit gate names to GateType enum
        self.gate_mapping = {
            'h': GateType.H,
            'x': GateType.X,
            'y': GateType.Y,
            'z': GateType.Z,
            's': GateType.S,
            't': GateType.T,
            'rx': GateType.RX,
            'ry': GateType.RY,
            'rz': GateType.RZ,
            'cx': GateType.CX,
            'cnot': GateType.CNOT,
            'cz': GateType.CZ,
            'swap': GateType.SWAP,
            'ccx': GateType.TOFFOLI,
            'toffoli': GateType.TOFFOLI,
            'measure': GateType.MEASURE,
            'barrier': GateType.BARRIER,
            'reset': GateType.RESET
        }
    
    def parse(self, code: str) -> Dict[str, Any]:
        """Parse Qiskit code"""
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
        """Extract Qiskit imports"""
        imports = []
        import_patterns = [
            r'from\s+qiskit\s+import\s+(.+)',
            r'import\s+qiskit',
            r'from\s+qiskit\.(\w+)\s+import\s+(.+)'
        ]
        
        for line in self.lines:
            for pattern in import_patterns:
                match = re.search(pattern, line)
                if match:
                    imports.append(line.strip())
                    break
        
        return imports
    
    def extract_registers(self) -> Dict[str, Any]:
        """Extract quantum and classical register declarations"""
        quantum_regs = []
        classical_regs = []
        
        # Pattern: QuantumRegister(size, 'name') or QuantumCircuit(n_qubits, n_bits)
        qreg_pattern = r'QuantumRegister\s*\(\s*(\d+)(?:\s*,\s*[\'"](\w+)[\'"])?\s*\)'
        creg_pattern = r'ClassicalRegister\s*\(\s*(\d+)(?:\s*,\s*[\'"](\w+)[\'"])?\s*\)'
        circuit_pattern = r'QuantumCircuit\s*\(\s*(\d+)(?:\s*,\s*(\d+))?\s*\)'
        
        for i, line in enumerate(self.lines):
            # Quantum registers
            qreg_match = re.search(qreg_pattern, line)
            if qreg_match:
                size = int(qreg_match.group(1))
                name = qreg_match.group(2) or f'q{len(quantum_regs)}'
                quantum_regs.append(
                    QuantumRegisterNode(name=name, size=size, line_number=i+1)
                )
            
            # Classical registers
            creg_match = re.search(creg_pattern, line)
            if creg_match:
                size = int(creg_match.group(1))
                name = creg_match.group(2) or f'c{len(classical_regs)}'
                classical_regs.append(
                    ClassicalRegisterNode(name=name, size=size, line_number=i+1)
                )
            
            # QuantumCircuit shorthand
            circuit_match = re.search(circuit_pattern, line)
            if circuit_match:
                n_qubits = int(circuit_match.group(1))
                n_bits = int(circuit_match.group(2)) if circuit_match.group(2) else 0
                
                if n_qubits > 0:
                    quantum_regs.append(
                        QuantumRegisterNode(name='q', size=n_qubits, line_number=i+1)
                    )
                if n_bits > 0:
                    classical_regs.append(
                        ClassicalRegisterNode(name='c', size=n_bits, line_number=i+1)
                    )
        
        return {
            'quantum': quantum_regs,
            'classical': classical_regs
        }
    
    def extract_quantum_operations(self) -> List[QuantumGateNode]:
        """Extract quantum gate operations"""
        gates = []
        
        # Pattern: qc.gate(qubit_index) or qc.gate(control, target)
        gate_pattern = r'\.(\w+)\s*\(\s*([\d,\s]+)\s*\)'
        
        for i, line in enumerate(self.lines):
            matches = re.finditer(gate_pattern, line)
            for match in matches:
                gate_name = match.group(1).lower()
                
                if gate_name in self.gate_mapping:
                    # Parse qubit indices
                    qubit_str = match.group(2)
                    qubits = [int(q.strip()) for q in qubit_str.split(',')]
                    
                    gate_type = self.gate_mapping[gate_name]
                    
                    # Determine if it's a controlled gate
                    is_controlled = gate_type in {
                        GateType.CX, GateType.CNOT, GateType.CZ, GateType.TOFFOLI
                    }
                    
                    control_qubits = []
                    target_qubits = qubits
                    
                    if is_controlled and len(qubits) > 1:
                        control_qubits = qubits[:-1]
                        target_qubits = [qubits[-1]]
                    
                    gates.append(QuantumGateNode(
                        gate_type=gate_type,
                        qubits=target_qubits,
                        control_qubits=control_qubits,
                        is_controlled=is_controlled,
                        line_number=i+1
                    ))
        
        return gates
    
    def extract_measurements(self) -> List[MeasurementNode]:
        """Extract measurement operations"""
        measurements = []
        
        # Pattern: qc.measure(qreg, creg) or qc.measure([0,1], [0,1])
        measure_pattern = r'\.measure\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)'
        
        for i, line in enumerate(self.lines):
            match = re.search(measure_pattern, line)
            if match:
                quantum_arg = match.group(1)
                classical_arg = match.group(2)
                
                # Try to parse as indices
                try:
                    q_indices = self._parse_indices(quantum_arg)
                    c_indices = self._parse_indices(classical_arg)
                    
                    measurements.append(MeasurementNode(
                        quantum_register='q',
                        classical_register='c',
                        qubit_indices=q_indices,
                        classical_indices=c_indices,
                        line_number=i+1
                    ))
                except:
                    # Fallback for named registers
                    measurements.append(MeasurementNode(
                        quantum_register=quantum_arg,
                        classical_register=classical_arg,
                        qubit_indices=[],
                        classical_indices=[],
                        line_number=i+1
                    ))
        
        return measurements
    
    def _parse_indices(self, arg: str) -> List[int]:
        """Parse qubit/bit indices from string"""
        # Remove brackets and split
        arg = arg.strip('[]')
        if ',' in arg:
            return [int(x.strip()) for x in arg.split(',')]
        else:
            return [int(arg)]