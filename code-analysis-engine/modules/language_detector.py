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
        1. File extension scoring
        2. Keyword signature scoring
        3. Syntax/token heuristics
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

                # Core types
                r'\bcirq\.Circuit\b',
                r'\bcirq\.LineQubit\b',
                r'\bcirq\.GridQubit\b',

                # Gates
                r'\bcirq\.H\b',
                r'\bcirq\.X\b',
                r'\bcirq\.Y\b',
                r'\bcirq\.Z\b',
                r'\bcirq\.CX\b',
                r'\bcirq\.CNOT\b',

                # Measurement
                r'\bcirq\.measure\b',

                # Cirq moments / ops
                r'\bcirq\.Moment\b',
                r'\bcirq\.GateOperation\b'
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


        # File extensions help a lot
        self.extensions = {
            # Strong identity
            ".qs": SupportedLanguage.QSHARP,
            ".qsx": SupportedLanguage.QSHARP,
            ".qsharp": SupportedLanguage.QSHARP,

            ".qasm": SupportedLanguage.OPENQASM,

            # Weak indicator (Python ecosystem)
            ".py": SupportedLanguage.PYTHON,

            # Optional artificial extensions
            ".cirq": SupportedLanguage.CIRQ,
            ".qiskit": SupportedLanguage.QISKIT,
        }


    # ---------------------------------------------------------------
    # Main detection
    # ---------------------------------------------------------------
    def detect(self, code: str, filename: Optional[str] = None) -> Dict[str, any]:
        # 0️⃣ Empty code check
        if not code or not code.strip():
            return self._error("Empty code provided")

        # 1️⃣ Filename extension check
        if filename:
            if "." not in filename:
                return self._error("Filename has no extension; cannot determine language")
        else:
            return self._error("Filename is required for language detection")

        # Extract extension
        ext = filename.split(".")[-1].lower()

        # Map extension → language
        ext_lang_map = {
            "py": SupportedLanguage.PYTHON,
            "qasm": SupportedLanguage.OPENQASM,
            "qs": SupportedLanguage.QSHARP,
            "qsx": SupportedLanguage.QSHARP,
            ".qsharp": SupportedLanguage.QSHARP,
            "cirq": SupportedLanguage.CIRQ,
            "qiskit": SupportedLanguage.QISKIT,
        }

        ext_lang = ext_lang_map.get(ext, None)

        # If extension is not known → error
        if ext_lang is None:
            return self._error(f"Unsupported file extension '.{ext}'")

        # 2️⃣ Signature score
        sig_score = self._score_signatures(code)

        # 3️⃣ Token score (for tiny snippets)
        tok_score = self._token_heuristics(code)

        # Final scores combined
        final_scores = {}

        for lang in SupportedLanguage:
            if lang == SupportedLanguage.UNKNOWN:
                continue

            if ext_lang == SupportedLanguage.PYTHON:
                # .py does NOT decide between Python/Qiskit/Cirq
                ext_weight = 0.10
            else:
                ext_weight = 0.50 if ext_lang == lang else 0.05

            final_scores[lang] = (
                (1 if ext_lang == lang else 0) * ext_weight +  # extension = strong signal
                sig_score.get(lang, 0) * 0.40 +
                tok_score.get(lang, 0) * 0.10
            )

        detected_lang = max(final_scores, key=final_scores.get)
        confidence = final_scores[detected_lang]

        # Threshold
        if confidence < 0.30:
            return {
                "language": SupportedLanguage.UNKNOWN,
                "confidence": confidence,
                "is_supported": False,
                "details": "Unable to confidently determine language"
            }

        return {
            "language": detected_lang,
            "confidence": round(confidence, 3),
            "is_supported": True,
            "details": "Detected using extension + signature + token heuristics"
        }

    # ---------------------------------------------------------------
    # Extension scoring
    # ---------------------------------------------------------------
    def _score_extension(self, filename: Optional[str]):
        if not filename:
            return ({}, None)

        for ext, lang in self.extensions.items():
            if filename.endswith(ext):
                return ({lang: 1.0}, lang)

        return ({}, None)

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
                ("linequbit" in lower)
            ) / 2,

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
    
    result = detector.detect(qiskit_code, "example.qiskit")
    print(f"Language: {result['language']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Details: {result['details']}")