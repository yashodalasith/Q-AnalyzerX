"""
Classical Complexity Analyzer
Calculates cyclomatic complexity, time complexity, space complexity
"""
import ast
import re
from typing import Dict, Any
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from models.analysis_result import ClassicalComplexity, TimeComplexity

class ComplexityAnalyzer:
    """Analyzes classical code complexity"""
    
    def __init__(self):
        pass
    
    def analyze(self, code: str, metadata: Dict[str, Any]) -> ClassicalComplexity:
        """
        Analyze classical complexity metrics
        
        Args:
            code: Source code string
            metadata: Metadata from parser
            
        Returns:
            ClassicalComplexity object
        """
        cyclomatic = self.calculate_cyclomatic_complexity(code)
        time_complexity = self.estimate_time_complexity(code, metadata)
        space_complexity = self.estimate_space_complexity(code)
        
        return ClassicalComplexity(
            cyclomatic_complexity=cyclomatic,
            time_complexity=time_complexity,
            space_complexity=space_complexity,
            loop_count=metadata.get('loop_count', 0),
            conditional_count=metadata.get('conditional_count', 0),
            function_count=metadata.get('function_count', 0),
            max_nesting_depth=metadata.get('nesting_depth', 0),
            lines_of_code=metadata.get('lines_of_code', 0)
        )
    
    def calculate_cyclomatic_complexity(self, code: str) -> int:
        """
        Calculate McCabe cyclomatic complexity
        Uses radon library for Python code
        """
        try:
            # Try Python AST parsing
            complexity_results = cc_visit(code)
            if complexity_results:
                # Return average complexity
                total = sum(item.complexity for item in complexity_results)
                return total // max(len(complexity_results), 1)
            return 1  # Base complexity
        except:
            # Fallback: manual calculation
            return self._calculate_complexity_manual(code)
    
    def _calculate_complexity_manual(self, code: str) -> int:
        """
        Manual cyclomatic complexity calculation
        Formula: M = E - N + 2P
        Simplified: count decision points + 1
        """
        decision_keywords = ['if', 'elif', 'else', 'for', 'while', 
                           'try', 'except', 'and', 'or', '?']
        
        complexity = 1  # Base complexity
        for line in code.split('\n'):
            line_lower = line.lower()
            for keyword in decision_keywords:
                complexity += line_lower.count(keyword)
        
        return min(complexity, 50)  # Cap at 50
    
    def estimate_time_complexity(self, code: str, metadata: Dict[str, Any]) -> TimeComplexity:
        """
        Estimate time complexity based on code structure
        Simple heuristic-based approach
        """
        loop_count = metadata.get('loop_count', 0)
        nesting_depth = metadata.get('nesting_depth', 0)
        
        # Check for specific patterns
        has_factorial = 'factorial' in code.lower() or '!' in code
        has_exponential = any(term in code.lower() for term in ['pow', '**', '^'])
        has_recursion = self._has_recursion(code)
        
        # Heuristic rules
        if has_factorial:
            return TimeComplexity.FACTORIAL
        
        if has_exponential and has_recursion:
            return TimeComplexity.EXPONENTIAL
        
        if nesting_depth >= 3 or loop_count >= 3:
            return TimeComplexity.CUBIC
        
        if nesting_depth >= 2 or loop_count >= 2:
            return TimeComplexity.QUADRATIC
        
        if loop_count == 1:
            # Check for sorting or binary search patterns
            if any(term in code.lower() for term in ['sort', 'sorted', 'quicksort', 'mergesort']):
                return TimeComplexity.LINEARITHMIC
            return TimeComplexity.LINEAR
        
        if loop_count == 0:
            if 'log' in code.lower() or 'binary' in code.lower():
                return TimeComplexity.LOGARITHMIC
            return TimeComplexity.CONSTANT
        
        return TimeComplexity.UNKNOWN
    
    def estimate_space_complexity(self, code: str) -> str:
        """
        Estimate space complexity
        Simple pattern-based estimation
        """
        # Check for data structure allocations
        has_list = any(term in code for term in ['[]', 'list(', 'List['])
        has_dict = any(term in code for term in ['{}', 'dict(', 'Dict['])
        has_set = 'set(' in code
        
        # Check for recursive calls (stack space)
        has_recursion = self._has_recursion(code)
        
        # Count nested structures
        nesting_level = code.count('[') + code.count('{')
        
        if has_recursion:
            return "O(n)"  # Stack space
        
        if nesting_level > 3 or (has_list and has_dict):
            return "O(n^2)"
        
        if has_list or has_dict or has_set:
            return "O(n)"
        
        return "O(1)"
    
    def _has_recursion(self, code: str) -> bool:
        """Detect recursive function calls"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    # Check if function calls itself
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            if isinstance(child.func, ast.Name) and child.func.id == func_name:
                                return True
        except:
            # Fallback regex
            pattern = r'def\s+(\w+)\s*\([^)]*\):.*\1\s*\('
            return bool(re.search(pattern, code, re.DOTALL))
        
        return False