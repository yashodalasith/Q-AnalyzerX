"""
Base Parser - Abstract class for all language parsers
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import ast
import re

class BaseParser(ABC):
    """Abstract base class for all code parsers"""
    
    def __init__(self):
        self.code = ""
        self.lines = []
    
    @abstractmethod
    def parse(self, code: str) -> Dict[str, Any]:
        """
        Parse code and return structured data
        
        Args:
            code: Source code string
            
        Returns:
            Dictionary with parsed elements
        """
        pass
    
    @abstractmethod
    def extract_imports(self) -> list:
        """Extract import statements"""
        pass
    
    @abstractmethod
    def extract_quantum_operations(self) -> list:
        """Extract quantum operations (gates, measurements)"""
        pass
    
    @abstractmethod
    def extract_registers(self) -> Dict[str, Any]:
        """Extract quantum and classical register declarations"""
        pass
    
    def count_lines(self, code: str) -> int:
        """Count non-empty, non-comment lines"""
        lines = [line.strip() for line in code.split('\n')]
        return len([line for line in lines if line and not line.startswith('#')])
    
    def extract_functions(self, code: str) -> list:
        """Extract function definitions (works for Python-like syntax)"""
        functions = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        'name': node.name,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args]
                    })
        except:
            # Fallback regex for non-Python languages
            pattern = r'(?:def|operation|function)\s+(\w+)\s*\('
            matches = re.finditer(pattern, code)
            for match in matches:
                functions.append({'name': match.group(1)})
        
        return functions
    
    def count_loops(self, code: str) -> int:
        """Count loop statements"""
        loop_keywords = ['for', 'while', 'repeat', 'loop']
        count = 0
        for line in code.split('\n'):
            line_lower = line.strip().lower()
            if any(line_lower.startswith(kw) for kw in loop_keywords):
                count += 1
        return count
    
    def count_conditionals(self, code: str) -> int:
        """Count conditional statements"""
        conditional_keywords = ['if', 'else', 'elif', 'switch', 'case']
        count = 0
        for line in code.split('\n'):
            line_lower = line.strip().lower()
            if any(line_lower.startswith(kw) for kw in conditional_keywords):
                count += 1
        return count
    
    def calculate_nesting_depth(self, code: str) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0
        current_depth = 0
        
        for char in code:
            if char in '{[(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in '}])':
                current_depth = max(0, current_depth - 1)
        
        # Also check indentation for Python-like languages
        for line in code.split('\n'):
            indent = len(line) - len(line.lstrip())
            depth = indent // 4  # Assuming 4-space indentation
            max_depth = max(max_depth, depth)
        
        return max_depth