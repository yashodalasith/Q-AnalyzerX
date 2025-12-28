"""
Language Detector for Quantum Code Analysis Engine
Identifies programming language of submitted code
"""
import re
from enum import Enum
from typing import Optional, Dict

class SupportedLanguage(str, Enum):
    PYTHON = "python"
    QISKIT = "qiskit"
    CIRQ = "cirq"
    QSHARP = "qsharp"
    OPENQASM = "openqasm"
    UNKNOWN = "unknown"


class LanguageDetector:
    """
    Extremely accurate language detector for quantum languages.
    Uses multi-stage scoring:
        1. Keyword signature scoring
        2. Syntax/token heuristics
    """

    def __init__(self):
        # Strong signatures unique to each language
        self.signatures = {
            # ==============================================================
            # QISKIT
            # ==============================================================    
            SupportedLanguage.QISKIT: [
                # Imports
                r'\bfrom\s+qiskit\s+import\b',
                r'\bimport\s+qiskit\b',

                # Core classes
                r'\bQuantumCircuit\b',
                r'\bQuantumRegister\b',
                r'\bClassicalRegister\b',

                # Qiskit functions
                r'\bAer\b',
                r'\bIBMQ\b',
                r'\bexecute\s*\(',
                r'\bassemble\s*\(',
                r'\btranspile\s*\(',

                # Qiskit simulator names
                r'\bqasm_simulator\b',
                r'\bstatevector_simulator\b',

                # Qiskit-style circuit usage
                r'\bqc\.[a-zA-Z_]+\(',
                r'\bqc\.h\s*\(',
                r'\bqc\.cx\s*\(',
                r'\bqc\.measure\s*\(',
                r'\bqc\.barrier\s*\(',

                # Qiskit visualization
                r'qiskit\.visualization',
                r'\bplot_bloch_vector\b'
            ],

            # ==============================================================
            # CIRQ
            # ==============================================================    
            SupportedLanguage.CIRQ: [
                # Imports
                r'\bimport\s+cirq\b',
                r'\bfrom\s+cirq\s+import\b',

                # Qubit types (very strong)
                r'\bcirq\.LineQubit\.range\s*\(',
                r'\bcirq\.LineQubit\s*\(',
                r'\bcirq\.GridQubit\s*\(',
                r'\bcirq\.NamedQubit\s*\(',

                # Circuit construction
                r'\bcirq\.Circuit\s*\(',
                r'\bcirq\.Moment\s*\(',

                # Gates (Cirq-style)
                r'\bcirq\.(H|X|Y|Z)\s*\(',
                r'\bcirq\.(CX|CZ|CNOT)\s*\(',

                # Exponentiated gates (EXTREMELY strong)
                r'\bcirq\.[A-Za-z]+\s*\(.*\)\s*\*\*\s*[0-9.]+',

                # Measurement (with optional key)
                r'\bcirq\.measure\s*\(',

                # Simulation / execution
                r'\bcirq\.Simulator\s*\(',
                r'\bcirq\.DensityMatrixSimulator\s*\(',
                r'\bsim\.run\s*\(',
                r'\bsim\.simulate\s*\(',

                # Results
                r'\bresult\.histogram\s*\(',
                r'\bresult\.measurements\b',

                # Parameters
                r'\bcirq\.Symbol\s*\(',
                r'\bcirq\.ParamResolver\s*\(',
            ],

            # ==============================================================
            # Q#
            # ==============================================================    
            SupportedLanguage.QSHARP: [
                # Q# structure
                r'\bnamespace\s+[A-Za-z_]\w*',
                r'\boperation\s+[A-Za-z_]\w*\s*\(',
                r'\bfunction\s+[A-Za-z_]\w*\s*\(',

                # Q# keywords
                r'\bAdjoint\s+auto\b',
                r'\bControlled\s+auto\b',
                r'\bwithin\s*\{',
                r'\bapply\s*\{',

                # Q# types
                r'\bQubit\b',
                r'\bQubit\[\d*\]',   # arrays

                # Microsoft namespace
                r'Microsoft\.Quantum'
            ],

            # ==============================================================
            # OPENQASM
            # ==============================================================    
            SupportedLanguage.OPENQASM: [
                # Headers
                r'\bOPENQASM\s+\d+\.\d+\b',
                r'include\s+"qelib1\.inc"',

                # Registers
                r'\bqreg\s+[A-Za-z_]\w*\[\d+\]\s*;',
                r'\bcreg\s+[A-Za-z_]\w*\[\d+\]\s*;',

                # Gates (case-sensitive)
                r'\bCX\b',
                r'\bH\b',
                r'\bU\(',

                # Commands
                r'\bmeasure\s+\w+\s*->\s*\w+;',
                r'\bbarrier\s+\w+;',
                r'^gate\s+[A-Za-z_]\w*',
                r'^opaque\s+[A-Za-z_]\w*'
            ],

            # ==============================================================
            # PYTHON (Plain)
            # ==============================================================    
            SupportedLanguage.PYTHON: [
                # Definitions
                r'\bdef\s+[A-Za-z_]\w*\s*\(',
                r'\bclass\s+[A-Za-z_]\w*\s*\:',

                # Imports
                r'\bimport\s+[A-Za-z_]\w*',
                r'\bfrom\s+[A-Za-z_]\w*\s+import\b',

                # Python function/method usage
                r'[A-Za-z_]\w*\s*=\s*[\[\{\"\']',   # common assignments
                r'\breturn\b',
                r'\bfor\s+[A-Za-z_]\w*\s+in\b',
                r'\bif\s+.*:',

                # Common Python-only tokens
                r'\bself\b',
                r'\blambda\b',
                r'\bwith\s+open\b',
                r'\bprint\s*\('
            ]
        }

    # ---------------------------------------------------------------
    # Main detection
    # ---------------------------------------------------------------
    def detect(self, code: str) -> Dict[str, any]:
        # 0️⃣ Empty code check
        if not code or not code.strip():
            return self._error("Empty code provided")

        sig_score = self._score_signatures(code)
        tok_score = self._token_heuristics(code)

        def score(lang):
            return sig_score.get(lang, 0) * 0.70 + tok_score.get(lang, 0) * 0.30

        # ---------------------------------------------------------------
        # 1️⃣ OPENQASM (non-Python, must win immediately)
        # ---------------------------------------------------------------
        if score(SupportedLanguage.OPENQASM) >= 0.30:
            return {
                "language": SupportedLanguage.OPENQASM,
                "confidence": round(score(SupportedLanguage.OPENQASM), 3),
                "is_supported": True,
                "details": "Detected OpenQASM"
            }

        # ---------------------------------------------------------------
        # 2️⃣ Q# (non-Python, must win immediately)
        # ---------------------------------------------------------------
        if score(SupportedLanguage.QSHARP) >= 0.30:
            return {
                "language": SupportedLanguage.QSHARP,
                "confidence": round(score(SupportedLanguage.QSHARP), 3),
                "is_supported": True,
                "details": "Detected Q#"
            }

        # ---------------------------------------------------------------
        # 3️⃣ CIRQ (Python-based, but dominates Python)
        # ---------------------------------------------------------------
        if score(SupportedLanguage.CIRQ) >= 0.30:
            return {
                "language": SupportedLanguage.CIRQ,
                "confidence": round(score(SupportedLanguage.CIRQ), 3),
                "is_supported": True,
                "details": "Detected Cirq (Python quantum framework)"
            }

        # ---------------------------------------------------------------
        # 4️⃣ QISKIT (Python-based, but dominates Python)
        # ---------------------------------------------------------------
        if score(SupportedLanguage.QISKIT) >= 0.30:
            return {
                "language": SupportedLanguage.QISKIT,
                "confidence": round(score(SupportedLanguage.QISKIT), 3),
                "is_supported": True,
                "details": "Detected Qiskit (Python quantum framework)"
            }

        # ---------------------------------------------------------------
        # 5️⃣ PYTHON (fallback only)
        # ---------------------------------------------------------------
        if self._is_python(code):
            return {
                "language": SupportedLanguage.PYTHON,
                "confidence": 0.50,
                "is_supported": True,
                "details": "Detected plain Python (no quantum framework)"
            }

        # ---------------------------------------------------------------
        # 6️⃣ UNKNOWN
        # ---------------------------------------------------------------
        return {
            "language": SupportedLanguage.UNKNOWN,
            "confidence": 0.0,
            "is_supported": False,
            "details": "Unable to confidently determine language"
        }


    # ---------------------------------------------------------------
    # Signature scoring
    # ---------------------------------------------------------------
    def _score_signatures(self, code: str):
        scores = {}
        for lang, patterns in self.signatures.items():
            matches = sum(
                1 for p in patterns
                if re.search(p, code, flags=re.MULTILINE | re.IGNORECASE)
            )
            # Normalize score between 0–1
            scores[lang] = min(matches * 0.20, 1.0)
        return scores


    # ---------------------------------------------------------------
    # Token / syntax heuristics for small snippets
    # ---------------------------------------------------------------
    def _token_heuristics(self, code: str):
        lower = code.lower()

        return {
            SupportedLanguage.OPENQASM: (
                ("qreg" in lower) +
                ("creg" in lower) +
                ("measure" in lower) +
                ("openqasm" in lower)
            ) / 4,

            SupportedLanguage.QISKIT: (
                ("qc." in lower) +
                ("quantumcircuit" in lower)
            ) / 2,

            SupportedLanguage.CIRQ: (
                ("cirq." in lower) +
                ("linequbit" in lower) +
                ("gridqubit" in lower) +
                ("namedqubit" in lower) +
                ("**" in lower and "cirq." in lower) +
                ("simulator(" in lower)
            ) / 6,

            SupportedLanguage.QSHARP: (
                ("operation" in lower) +
                ("namespace" in lower) +
                ("microsoft.quantum" in lower)
            ) / 3,

            SupportedLanguage.PYTHON: (
                ("def " in lower) +
                ("import" in lower)
            ) / 2,
        }

    # ---------------------------------------------------------------
    # Error response
    # ---------------------------------------------------------------
    def _error(self, message: str):
        return {
            "language": SupportedLanguage.UNKNOWN,
            "confidence": 0.0,
            "is_supported": False,
            "details": message
        }
    
    def _is_python(self, code: str) -> bool:
        """Check if code is valid Python"""
        python_indicators = [
            r'def\s+\w+\s*\(',
            r'class\s+\w+',
            r'import\s+\w+',
            r'from\s+\w+\s+import',
            r'if\s+__name__\s*==\s*["\']__main__["\']'
        ]
        
        matches = sum(1 for pattern in python_indicators 
                     if re.search(pattern, code))
        
        return matches >= 2

# Example usage
if __name__ == "__main__":
    detector = LanguageDetector()
    
    # Test with Qiskit code
    qiskit_code = """
from qiskit import QuantumCircuit, QuantumRegister
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
    """
    
    result = detector.detect(qiskit_code)
    print(f"Language: {result['language']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Details: {result['details']}")