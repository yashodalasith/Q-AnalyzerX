# accurate_depth.py
from typing import Dict, List, Set, Optional, Any
from models.unified_ast import UnifiedAST, QuantumGateNode

class AccurateCircuitDepthCalculator:
    """
    Calculates exact circuit depth by building a dependency graph
    (per-qubit last-gate) and computing the critical path (longest path).
    """

    def __init__(self):
        # qubit -> last gate id seen on that qubit
        self.qubit_last_gate: Dict[int, int] = {}
        # gate_id -> list of dependent gate_ids (gates that must precede this gate)
        self.gate_dependencies: Dict[int, List[int]] = {}
        # gate_id -> computed depth (1-based)
        self.gate_depth: Dict[int, int] = {}
        # id -> gate object mapping for quick lookup
        self.id_to_gate: Dict[int, QuantumGateNode] = {}

    def calculate_depth(self, unified_ast: UnifiedAST) -> int:
        """
        Build dependency graph and compute depths for each gate.
        Returns the maximum depth (circuit depth).
        """
        gates = unified_ast.gates if unified_ast and unified_ast.gates else []
        if not gates:
            return 0

        # reset internal structures
        self.qubit_last_gate.clear()
        self.gate_dependencies.clear()
        self.gate_depth.clear()
        self.id_to_gate.clear()

        # Build dependency graph
        self._build_dependency_graph(gates)

        # Compute depth for each gate (memoized recursion)
        for gate in gates:
            self._calculate_gate_depth(gate)

        return max(self.gate_depth.values()) if self.gate_depth else 0

    def _build_dependency_graph(self, gates: List[QuantumGateNode]) -> None:
        """
        For each gate (in original sequence), compute dependencies as the union
        of last gates on all qubits it touches. Only after collecting deps,
        update the last-gate mapping for those qubits.
        """
        for gate in gates:
            gate_id = id(gate)
            self.id_to_gate[gate_id] = gate

            # collect all qubits involved (targets + controls)
            all_qubits = set((gate.qubits or []) + (gate.control_qubits or []))

            # gather dependencies (union of last gates for each qubit)
            deps: Set[int] = set()
            for q in all_qubits:
                prev = self.qubit_last_gate.get(q)
                if prev is not None:
                    deps.add(prev)

            # store as list (deterministic order: sort for stability)
            self.gate_dependencies[gate_id] = sorted(deps)

            # now update last gate seen on each involved qubit
            for q in all_qubits:
                self.qubit_last_gate[q] = gate_id

    def _calculate_gate_depth(self, gate: QuantumGateNode) -> int:
        """
        Depth of a gate = 1 if no dependencies, otherwise 1 + max(depth(deps)).
        Uses memoization in self.gate_depth.
        """
        gate_id = id(gate)
        if gate_id in self.gate_depth:
            return self.gate_depth[gate_id]

        deps = self.gate_dependencies.get(gate_id, [])
        if not deps:
            self.gate_depth[gate_id] = 1
            return 1

        max_dep = 0
        for dep_id in deps:
            dep_gate = self._find_gate_by_id(dep_id)
            if dep_gate is None:
                # defensive: if gate object not found, treat as depth 0 dependency
                continue
            dep_depth = self._calculate_gate_depth(dep_gate)
            if dep_depth > max_dep:
                max_dep = dep_depth

        depth = 1 + max_dep
        self.gate_depth[gate_id] = depth
        return depth

    def _find_gate_by_id(self, gate_id: int) -> Optional[QuantumGateNode]:
        """Lookup gate object from id using the id_to_gate map."""
        return self.id_to_gate.get(gate_id)

    def get_critical_path(self, unified_ast: UnifiedAST) -> List[QuantumGateNode]:
        """
        Return one critical path (longest dependent sequence of gates).
        If multiple endpoints have same max depth, choose the longest backtrack found.
        """
        gates = unified_ast.gates if unified_ast and unified_ast.gates else []
        if not gates:
            return []

        # Ensure depths are computed
        if not self.gate_depth:
            self.calculate_depth(unified_ast)

        if not self.gate_depth:
            return []

        max_depth = max(self.gate_depth.values())

        # collect candidates (gates with maximum depth)
        candidates = [g for g in gates if self.gate_depth.get(id(g)) == max_depth]
        best_path: List[QuantumGateNode] = []

        for cand in candidates:
            path = self._backtrack_path(cand, max_depth)
            if len(path) > len(best_path):
                best_path = path

        return best_path

    def _backtrack_path(self, gate: QuantumGateNode, target_depth: int) -> List[QuantumGateNode]:
        """
        Recursively backtrack from gate with depth target_depth to form a path
        of length target_depth. If multiple dependency choices exist, return
        the first path found that satisfies depth ordering.
        """
        if target_depth <= 1:
            return [gate]

        gate_id = id(gate)
        deps = self.gate_dependencies.get(gate_id, [])

        # Look for a dependency whose depth is target_depth - 1
        for dep_id in deps:
            dep_gate = self._find_gate_by_id(dep_id)
            if dep_gate and self.gate_depth.get(dep_id) == target_depth - 1:
                return self._backtrack_path(dep_gate, target_depth - 1) + [gate]

        # Fallback: if no exact predecessor found (shouldn't happen), return gate only
        return [gate]


# convenience wrapper
def calculate_accurate_depth(unified_ast: UnifiedAST) -> Dict[str, Any]:
    calculator = AccurateCircuitDepthCalculator()
    depth = calculator.calculate_depth(unified_ast)
    critical_path = calculator.get_critical_path(unified_ast)
    return {
        'circuit_depth': depth,
        'critical_path_length': len(critical_path),
        'critical_path': critical_path,
        'parallelism_ratio': len(unified_ast.gates) / max(depth, 1) if unified_ast.gates else 0.0
    }