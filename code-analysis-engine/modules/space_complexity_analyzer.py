"""
Accurate Space Complexity Analysis through Memory Allocation Tracking
"""
import ast
from typing import Dict, Set, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Variable:
    """Represents a variable and its memory characteristics"""
    name: str
    var_type: str  # 'scalar', 'list', 'dict', 'set', 'matrix', 'unknown'
    dimensions: List[str]  # e.g., ['n'], ['n', 'm'] for matrix
    scope: str  # 'local', 'global', 'parameter'
    line: int

@dataclass
class MemoryAllocation:
    """Represents a memory allocation"""
    size: str  # e.g., 'n', 'n^2', '1'
    var_name: str
    allocation_type: str  # 'heap', 'stack'
    line: int

class AccurateSpaceComplexityAnalyzer:
    """
    Analyzes space complexity through:
    1. Variable scope and lifetime tracking
    2. Data structure size analysis
    3. Recursive call stack depth
    4. Memory allocation pattern detection
    """
    
    def __init__(self):
        self.variables: Dict[str, Variable] = {}
        self.allocations: List[MemoryAllocation] = []
        self.max_recursion_depth = 0
        self.scope_stack = []
        self.complexity_cache = {}
    
    def analyze(self, code: str) -> str:
        """
        Analyze space complexity
        
        Returns: Space complexity string like 'O(1)', 'O(n)', 'O(n^2)'
        """
        try:
            tree = ast.parse(code)
            self._analyze_tree(tree)
            return self._calculate_total_complexity()
        except SyntaxError:
            return "O(1)"  # Default fallback
    
    def _analyze_tree(self, tree: ast.AST):
        """Traverse AST and track memory allocations"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._analyze_function(node)
            elif isinstance(node, ast.Assign):
                self._analyze_assignment(node)
            elif isinstance(node, ast.AugAssign):
                self._analyze_aug_assignment(node)
    
    def _analyze_function(self, func_node: ast.FunctionDef):
        """Analyze function for recursion and local variables"""
        func_name = func_node.name
        self.scope_stack.append(func_name)
        
        # Check for recursion
        recursion_depth = self._calculate_recursion_depth(func_node)
        if recursion_depth > 0:
            self.max_recursion_depth = max(self.max_recursion_depth, recursion_depth)
            # Recursive functions use stack space
            self.allocations.append(MemoryAllocation(
                size=self._estimate_recursion_complexity(func_node),
                var_name=f"{func_name}_stack",
                allocation_type='stack',
                line=func_node.lineno
            ))
        
        # Analyze function body
        for stmt in func_node.body:
            self._analyze_statement(stmt)
        
        self.scope_stack.pop()
    
    def _analyze_assignment(self, node: ast.Assign):
        """Analyze variable assignments for memory allocation"""
        if not node.targets:
            return
        
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            return
        
        var_name = target.id
        var_type, dimensions = self._infer_type_and_size(node.value)
        
        # Create variable entry
        self.variables[var_name] = Variable(
            name=var_name,
            var_type=var_type,
            dimensions=dimensions,
            scope='local' if self.scope_stack else 'global',
            line=node.lineno
        )
        
        # Calculate memory allocation
        if var_type in ['list', 'dict', 'set']:
            size = self._calculate_size(dimensions)
            self.allocations.append(MemoryAllocation(
                size=size,
                var_name=var_name,
                allocation_type='heap',
                line=node.lineno
            ))
    
    def _analyze_aug_assignment(self, node: ast.AugAssign):
        """Analyze augmented assignments (+=, etc.)"""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            # Check if it's list append or similar growth operation
            if isinstance(node.op, ast.Add) and var_name in self.variables:
                var = self.variables[var_name]
                if var.var_type == 'list':
                    # List is growing - might need O(n) space
                    pass  # Already tracked in initial allocation
    
    def _analyze_statement(self, stmt: ast.stmt):
        """Recursively analyze statements"""
        if isinstance(stmt, ast.Assign):
            self._analyze_assignment(stmt)
        elif isinstance(stmt, ast.AugAssign):
            self._analyze_aug_assignment(stmt)
        elif isinstance(stmt, (ast.For, ast.While)):
            for body_stmt in stmt.body:
                self._analyze_statement(body_stmt)
        elif isinstance(stmt, ast.If):
            for body_stmt in stmt.body:
                self._analyze_statement(body_stmt)
            for else_stmt in stmt.orelse:
                self._analyze_statement(else_stmt)
    
    def _infer_type_and_size(self, value_node: ast.AST) -> Tuple[str, List[str]]:
        """
        Infer variable type and dimensions from assignment value
        
        Returns: (type, dimensions)
        Examples:
            [0] * n -> ('list', ['n'])
            [[0] * m for _ in range(n)] -> ('matrix', ['n', 'm'])
            {} -> ('dict', [])
        """
        if isinstance(value_node, ast.List):
            # List literal [1, 2, 3] or list comprehension
            if isinstance(value_node, ast.ListComp):
                size = self._extract_comprehension_size(value_node)
                # Check if nested (matrix)
                if isinstance(value_node.elt, (ast.List, ast.ListComp)):
                    inner_size = self._extract_size_from_node(value_node.elt)
                    return ('matrix', [size, inner_size])
                return ('list', [size])
            else:
                # Fixed size list
                return ('list', [str(len(value_node.elts))])
        
        elif isinstance(value_node, ast.Dict):
            return ('dict', ['n'])  # Assume O(n) for dict
        
        elif isinstance(value_node, ast.Set):
            return ('set', ['n'])
        
        elif isinstance(value_node, ast.Call):
            func = value_node.func
            if isinstance(func, ast.Name):
                # Built-in constructors
                if func.id in ['list', 'dict', 'set']:
                    if value_node.args:
                        size = self._extract_size_from_node(value_node.args[0])
                        return (func.id, [size])
                    return (func.id, ['n'])
                
                # range() creates iterator, not list (O(1) space)
                elif func.id == 'range':
                    return ('scalar', [])
            
            elif isinstance(func, ast.Attribute):
                # Methods like str.split() -> list
                if func.attr in ['split', 'splitlines']:
                    return ('list', ['n'])
        
        elif isinstance(value_node, ast.BinOp):
            # [0] * n or similar multiplication
            if isinstance(value_node.op, ast.Mult):
                if isinstance(value_node.left, ast.List):
                    size = self._extract_size_from_node(value_node.right)
                    return ('list', [size])
                elif isinstance(value_node.right, ast.List):
                    size = self._extract_size_from_node(value_node.left)
                    return ('list', [size])
        
        # Default: scalar (O(1))
        return ('scalar', [])
    
    def _extract_size_from_node(self, node: ast.AST) -> str:
        """Extract size expression from AST node"""
        if isinstance(node, ast.Name):
            return node.id  # Variable name like 'n'
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'len':
                if node.args and isinstance(node.args[0], ast.Name):
                    return node.args[0].id
                return 'n'
            elif isinstance(node.func, ast.Name) and node.func.id == 'range':
                if node.args:
                    return self._extract_size_from_node(node.args[0])
                return 'n'
        return 'n'
    
    def _extract_comprehension_size(self, comp_node: ast.ListComp) -> str:
        """Extract size from list comprehension"""
        if comp_node.generators:
            gen = comp_node.generators[0]
            return self._extract_size_from_node(gen.iter)
        return 'n'
    
    def _calculate_recursion_depth(self, func_node: ast.FunctionDef) -> int:
        """Calculate maximum recursion depth"""
        func_name = func_node.name
        
        # Check if function calls itself
        has_recursion = False
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == func_name:
                    has_recursion = True
                    break
        
        if not has_recursion:
            return 0
        
        # Analyze recursion pattern
        # Simple heuristic: check if it's divide-and-conquer or linear
        depth = self._estimate_recursion_depth_pattern(func_node)
        return depth
    
    def _estimate_recursion_depth_pattern(self, func_node: ast.FunctionDef) -> int:
        """Estimate recursion depth based on pattern"""
        func_name = func_node.name
        
        # Count recursive calls
        recursive_calls = []
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == func_name:
                    recursive_calls.append(node)
        
        if len(recursive_calls) == 1:
            # Linear recursion (factorial, linked list traversal)
            return 1  # O(n) stack space
        elif len(recursive_calls) == 2:
            # Binary recursion (merge sort, binary tree)
            # Check if divide-and-conquer
            has_division = any(
                isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Div, ast.FloorDiv))
                for n in ast.walk(func_node)
            )
            if has_division:
                return 1  # O(log n) stack space for divide-and-conquer
            else:
                return 2  # O(n) for tree recursion
        else:
            return 2  # Multiple recursion paths
    
    def _estimate_recursion_complexity(self, func_node: ast.FunctionDef) -> str:
        """Estimate space complexity of recursion"""
        depth = self._estimate_recursion_depth_pattern(func_node)
        
        if depth == 0:
            return '1'
        elif depth == 1:
            # Check for divide-and-conquer
            has_division = any(
                isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Div, ast.FloorDiv))
                for n in ast.walk(func_node)
            )
            return 'log(n)' if has_division else 'n'
        else:
            return 'n'
    
    def _calculate_size(self, dimensions: List[str]) -> str:
        """Calculate total size from dimensions"""
        if not dimensions:
            return '1'
        
        # Handle constants
        total_size = []
        for dim in dimensions:
            if dim.isdigit():
                continue  # Constant dimensions
            else:
                total_size.append(dim)
        
        if not total_size:
            return '1'  # All dimensions are constants
        elif len(total_size) == 1:
            return total_size[0]  # O(n)
        else:
            # Multiple dimensions: O(n*m) or O(n^2)
            if all(d == total_size[0] for d in total_size):
                return f"{total_size[0]}^{len(total_size)}"  # O(n^2)
            else:
                return '*'.join(total_size)  # O(n*m)
    
    def _calculate_total_complexity(self) -> str:
        """Calculate overall space complexity"""
        if not self.allocations and self.max_recursion_depth == 0:
            return "O(1)"
        
        # Collect all space complexities
        complexities = []
        
        # Add recursion stack space
        if self.max_recursion_depth > 0:
            for alloc in self.allocations:
                if alloc.allocation_type == 'stack':
                    complexities.append(alloc.size)
        
        # Add heap allocations
        heap_allocs = [a for a in self.allocations if a.allocation_type == 'heap']
        
        # Find maximum heap allocation
        for alloc in heap_allocs:
            complexities.append(alloc.size)
        
        if not complexities:
            return "O(1)"
        
        # Return dominant complexity
        dominant = self._find_dominant_complexity(complexities)
        return f"O({dominant})"
    
    def _find_dominant_complexity(self, complexities: List[str]) -> str:
        """Find the dominant (maximum) complexity"""
        # Order of dominance
        order = {
            '1': 0,
            'log(n)': 1,
            'n': 2,
            'n*log(n)': 3,
            'n^2': 4,
            'n*m': 4,  # Treat as O(n^2) for comparison
            'n^3': 5,
            '2^n': 6,
            'n!': 7
        }
        
        max_complexity = '1'
        max_order = 0
        
        for comp in complexities:
            comp_order = order.get(comp, 2)  # Default to O(n)
            if comp_order > max_order:
                max_order = comp_order
                max_complexity = comp
        
        return max_complexity
    
    def get_detailed_report(self) -> Dict:
        """Get detailed memory allocation report"""
        return {
            'total_complexity': self._calculate_total_complexity(),
            'variables': {
                name: {
                    'type': var.var_type,
                    'dimensions': var.dimensions,
                    'scope': var.scope
                }
                for name, var in self.variables.items()
            },
            'allocations': [
                {
                    'size': alloc.size,
                    'variable': alloc.var_name,
                    'type': alloc.allocation_type,
                    'line': alloc.line
                }
                for alloc in self.allocations
            ],
            'max_recursion_depth': self.max_recursion_depth
        }


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_cases = [
        # O(1) - constant space
        """
x = 5
y = 10
z = x + y
        """,
        
        # O(n) - single array
        """
def process(n):
    arr = [0] * n
    return sum(arr)
        """,
        
        # O(n^2) - matrix
        """
def create_matrix(n):
    matrix = [[0] * n for _ in range(n)]
    return matrix
        """,
        
        # O(n) - recursion with linear depth
        """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
        """,
        
        # O(log n) - divide and conquer
        """
def binary_search(arr, x, low, high):
    if high >= low:
        mid = (high + low) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] > x:
            return binary_search(arr, x, low, mid - 1)
        else:
            return binary_search(arr, x, mid + 1, high)
    return -1
        """,
    ]
    
    analyzer = AccurateSpaceComplexityAnalyzer()
    
    for i, code in enumerate(test_cases, 1):
        analyzer = AccurateSpaceComplexityAnalyzer()  # Fresh instance
        result = analyzer.analyze(code)
        print(f"\nTest Case {i}:")
        print(f"Space Complexity: {result}")
        print(f"Details: {analyzer.get_detailed_report()}")