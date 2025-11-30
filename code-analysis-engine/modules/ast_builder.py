"""
AST Builder - Constructs unified AST from parsed code
"""
from typing import Dict, Any
from models.unified_ast import UnifiedAST
from modules.language_detector import SupportedLanguage
from modules.parsers import (
    QiskitParser, CirqParser, OpenQASMParser,
    PythonParser, QSharpParser
)

class ASTBuilder:
    """Builds unified AST from parsed code"""
    
    def __init__(self):
        self.parsers = {
            SupportedLanguage.QISKIT: QiskitParser(),
            SupportedLanguage.CIRQ: CirqParser(),
            SupportedLanguage.OPENQASM: OpenQASMParser(),
            SupportedLanguage.PYTHON: PythonParser(),
            SupportedLanguage.QSHARP: QSharpParser()
        }
    
    def build(self, code: str, language: SupportedLanguage) -> UnifiedAST:
        """
        Build unified AST from source code
        
        Args:
            code: Source code string
            language: Detected programming language
            
        Returns:
            UnifiedAST object
        """
        if language not in self.parsers:
            raise ValueError(f"Unsupported language: {language}")
        
        # Parse code with appropriate parser
        parser = self.parsers[language]
        parsed_data = parser.parse(code)
        
        # Extract register information
        quantum_regs = parsed_data['registers'].get('quantum', [])
        classical_regs = parsed_data['registers'].get('classical', [])
        
        # Calculate total qubits and bits
        total_qubits = sum(reg.size for reg in quantum_regs)
        total_bits = sum(reg.size for reg in classical_regs)
        
        # Extract gates and measurements
        gates = parsed_data.get('gates', [])
        measurements = parsed_data.get('measurements', [])
        
        # Build unified AST
        unified_ast = UnifiedAST(
            source_language=language.value,
            quantum_registers=quantum_regs,
            classical_registers=classical_regs,
            gates=gates,
            measurements=measurements,
            imports=parsed_data.get('imports', []),
            functions=parsed_data.get('functions', []),
            total_qubits=total_qubits,
            total_classical_bits=total_bits,
            total_gates=len(gates)
        )
        
        return unified_ast
    
    def get_metadata(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from parsed data"""
        metadata = parsed_data.get('metadata', {})
        return {
            'lines_of_code': metadata.get('lines_of_code', 0),
            'loop_count': metadata.get('loop_count', 0),
            'conditional_count': metadata.get('conditional_count', 0),
            'nesting_depth': metadata.get('nesting_depth', 0),
            'function_count': len(parsed_data.get('functions', []))
        }