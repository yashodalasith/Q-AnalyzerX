"""
OpenQASM Parser - Parses OpenQASM 2.0/3.0 quantum assembly
"""
import re
from typing import Dict, Any, List
from .base_parser import BaseParser
from models.unified_ast import (
    QuantumRegisterNode, ClassicalRegisterNode,
    QuantumGateNode, MeasurementNode, GateType
)

class OpenQASMParser(BaseParser):
    """Parser for OpenQASM code"""
    
    def __init__(self):
        super().__init__()
        self.gate_mapping = {
             # Standard 1-qubit gates
            'h': GateType.H,
            'x': GateType.X,
            'y': GateType.Y,
            'z': GateType.Z,
            's': GateType.S,
            't': GateType.T,

            # Parameterized single-qubit gates
            'rx': GateType.RX,
            'ry': GateType.RY,
            'rz': GateType.RZ,
            'u1': GateType.RZ,     # equivalent to phase rotation
            'u2': GateType.CUSTOM, # general-purpose
            'u3': GateType.CUSTOM,

            # 2-qubit gates
            'cx': GateType.CX,
            'cz': GateType.CZ,
            'swap': GateType.SWAP,
            'cp': GateType.CZ,     # controlled-phase
            'cu1': GateType.CZ,    # controlled-phase (U1)

            # Multi-qubit
            'ccx': GateType.TOFFOLI,
            'ccu': GateType.FREDKIN,  # optional - custom
        }
    
    def parse(self, code: str) -> Dict[str, Any]:
        """Parse OpenQASM code"""
        self.code = code
        self.lines = [line.strip() for line in code.split('\n') if line.strip()]
        
        return {
            'imports': self.extract_imports(),
            'registers': self.extract_registers(),
            'gates': self.extract_quantum_operations(),
            'measurements': self.extract_measurements(),
            'functions': [],  # OpenQASM has gate definitions
            'metadata': {
                'lines_of_code': len(self.lines),
                'loop_count': 0,  # OpenQASM 2.0 doesn't have loops
                'conditional_count': 0,
                'nesting_depth': 0
            }
        }
    
    def extract_imports(self) -> List[str]:
        """Extract include statements"""
        imports = []
        for line in self.lines:
            if line.startswith('include'):
                imports.append(line)
        return imports
    
    def extract_registers(self) -> Dict[str, Any]:
        """Extract qreg and creg declarations"""
        quantum_regs = []
        classical_regs = []
        
        # Pattern: qreg name[size]; or creg name[size];
        qreg_pattern = r'qreg\s+(\w+)\[(\d+)\]'
        creg_pattern = r'creg\s+(\w+)\[(\d+)\]'
        
        for i, line in enumerate(self.lines):
            qreg_match = re.search(qreg_pattern, line)
            if qreg_match:
                name = qreg_match.group(1)
                size = int(qreg_match.group(2))
                quantum_regs.append(
                    QuantumRegisterNode(name=name, size=size, line_number=i+1)
                )
            
            creg_match = re.search(creg_pattern, line)
            if creg_match:
                name = creg_match.group(1)
                size = int(creg_match.group(2))
                classical_regs.append(
                    ClassicalRegisterNode(name=name, size=size, line_number=i+1)
                )
        
        return {'quantum': quantum_regs, 'classical': classical_regs}
    
    def extract_quantum_operations(self) -> List[QuantumGateNode]:
        """Extract gate operations"""
        gates = []
        
        # Pattern: gate qreg[index]; or gate qreg1[i], qreg2[j];
        gate_pattern = r'(\w+)(\([^)]*\))?\s+([\w\[\],\s]+);'
        
        for i, line in enumerate(self.lines):
            # Skip declarations and directives
            if line.startswith(('OPENQASM', 'include', 'qreg', 'creg', 'measure', 'barrier')):
                continue
            
            match = re.search(gate_pattern, line)
            if match:
                gate_name = match.group(1).lower()
                
                if gate_name in self.gate_mapping:
                    gate_type = self.gate_mapping.get(gate_name, GateType.CUSTOM)

                    # Parse parameters
                    param_str = match.group(2)
                    parameters = self._parse_parameters(param_str)

                    # Extract qubit indices
                    qubits_str = match.group(3)
                    qubit_indices = self._extract_qubit_indices(qubits_str)

                    is_controlled = gate_name.startswith("c") and len(qubit_indices) > 1

                    control_qubits = []
                    target_qubits = qubit_indices

                    if is_controlled:
                        control_qubits = qubit_indices[:-1]
                        target_qubits = [qubit_indices[-1]]
                    
                    gates.append(QuantumGateNode(
                        gate_type=gate_type,
                        qubits=target_qubits,
                        control_qubits=control_qubits,
                        is_controlled=is_controlled,
                        parameters=parameters,
                        line_number=i + 1
                    ))

        return gates
    
    def extract_measurements(self) -> List[MeasurementNode]:
        """Extract measure operations"""
        measurements = []
        
        # Pattern: measure qreg[i] -> creg[j];
        measure_pattern = r'measure\s+(\w+)\[(\d+)\]\s*->\s*(\w+)\[(\d+)\]'
        
        for i, line in enumerate(self.lines):
            match = re.search(measure_pattern, line)
            if match:
                qreg_name = match.group(1)
                q_index = int(match.group(2))
                creg_name = match.group(3)
                c_index = int(match.group(4))
                
                measurements.append(MeasurementNode(
                    quantum_register=qreg_name,
                    classical_register=creg_name,
                    qubit_indices=[q_index],
                    classical_indices=[c_index],
                    line_number=i+1
                ))
        
        return measurements
    
    def _extract_qubit_indices(self, qubits_str: str) -> List[int]:
        """
        Extract qubit indices from strings like:
            q[0]
            q[0], q[1]
            q[0:3]
            q[1:5:2]
            q[0], q[2:5]
        """
        indices = []
        matches = re.findall(r'\[([0-9:]+)\]', qubits_str)

        for m in matches:
            if ':' in m:
                # Slice case
                parts = [int(x) if x else None for x in m.split(':')]
                start, stop, step = (parts + [None] * 3)[:3]  # pad missing values

                # Generate the expanded slice
                indices.extend(range(start or 0, stop, step or 1))
            else:
                # Single index
                indices.append(int(m))

        return indices
    
    def _parse_parameters(self, param_str: str) -> List[float]:
        """Parse parameters like (pi/2), (theta), (3.14)"""
        if not param_str:
            return []

        inside = param_str.strip("()")

        # Replace pi with numeric constant
        inside = inside.replace("pi", "3.141592653589793")

        # Support lists like (a, b, c)
        params = []
        for expr in inside.split(','):
            expr = expr.strip()
            try:
                params.append(float(eval(expr)))
            except:
                # Keep symbolic parameters
                params.append(expr)

        return params