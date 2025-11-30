"""
Language Detector for Quantum Code Analysis Engine
Identifies programming language of submitted code
"""
import re
from enum import Enum
from typing import Optional, Dict

class SupportedLanguage(str, Enum):
    """Supported quantum programming languages"""
    PYTHON = "python"
    QISKIT = "qiskit"
    QSHARP = "qsharp"
    CIRQ = "cirq"
    OPENQASM = "openqasm"
    UNKNOWN = "unknown"

class LanguageDetector:
    """Detects programming language from code content"""
    
    def __init__(self):
        # Define language signatures - unique imports/keywords
        self.signatures = {
            SupportedLanguage.QISKIT: [
                r'from\s+qiskit\s+import',
                r'import\s+qiskit',
                r'QuantumCircuit',
                r'QuantumRegister',
                r'ClassicalRegister'
            ],
            SupportedLanguage.CIRQ: [
                r'import\s+cirq',
                r'from\s+cirq\s+import',
                r'cirq\.Circuit',
                r'cirq\.LineQubit'
            ],
            SupportedLanguage.QSHARP: [
                r'namespace\s+\w+\s*{',
                r'operation\s+\w+\s*\(',
                r'using\s*\(',
                r'body\s*\(\.\.\.\)',
                r'Microsoft\.Quantum'
            ],
            SupportedLanguage.OPENQASM: [
                r'OPENQASM\s+\d+\.\d+',
                r'include\s+"qelib1\.inc"',
                r'qreg\s+\w+\[\d+\]',
                r'creg\s+\w+\[\d+\]',
                r'^gate\s+\w+'
            ]
        }
    
    def detect(self, code: str) -> Dict[str, any]:
        """
        Detect language from code content
        
        Args:
            code: Source code string
            
        Returns:
            Dict with language, confidence, and details
        """
        if not code or not code.strip():
            return {
                "language": SupportedLanguage.UNKNOWN,
                "confidence": 0.0,
                "is_supported": False,
                "details": "Empty code provided"
            }
        
        # Check each language signature
        scores = {}
        
        for lang, patterns in self.signatures.items():
            score = 0
            matched_patterns = []
            
            for pattern in patterns:
                if re.search(pattern, code, re.MULTILINE):
                    score += 1
                    matched_patterns.append(pattern)
            
            if score > 0:
                scores[lang] = {
                    "score": score,
                    "matched": matched_patterns
                }
        
        # Determine language
        if not scores:
            # Check if it's plain Python
            if self._is_python(code):
                return {
                    "language": SupportedLanguage.PYTHON,
                    "confidence": 0.7,
                    "is_supported": True,
                    "details": "Detected as Python (no quantum library detected)"
                }
            
            return {
                "language": SupportedLanguage.UNKNOWN,
                "confidence": 0.0,
                "is_supported": False,
                "details": "Could not identify language"
            }
        
        # Get language with highest score
        detected_lang = max(scores.items(), key=lambda x: x[1]["score"])
        lang_name = detected_lang[0]
        lang_data = detected_lang[1]
        
        # Calculate confidence (normalized)
        max_possible = len(self.signatures[lang_name])
        confidence = min(lang_data["score"] / max_possible, 1.0)
        
        return {
            "language": lang_name,
            "confidence": confidence,
            "is_supported": True,
            "details": f"Matched {lang_data['score']} patterns: {lang_data['matched'][:3]}"
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