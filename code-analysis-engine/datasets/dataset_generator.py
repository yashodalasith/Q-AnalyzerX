"""
Quantum Algorithm Dataset Generator
Generates variations of canonical quantum algorithms across multiple languages
Target: 100+ variations per algorithm per language
"""
import os
import json
import random
from typing import List, Dict, Tuple
from pathlib import Path

class QuantumAlgorithmDatasetGenerator:
    """
    Generates large-scale dataset of quantum algorithms with variations
    
    Algorithms covered:
    1. Grover's Search
    2. Shor's Factorization
    3. Quantum Fourier Transform (QFT)
    4. Variational Quantum Eigensolver (VQE)
    5. Quantum Approximate Optimization Algorithm (QAOA)
    6. Quantum Phase Estimation (QPE)
    7. Deutsch-Jozsa Algorithm
    8. Bernstein-Vazirani Algorithm
    9. Simon's Algorithm
    10. Amplitude Amplification
    
    Languages: Qiskit, Cirq, OpenQASM, Q#
    """
    
    def __init__(self, output_dir: str = "datasets/quantum_algorithms"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Variation parameters
        self.qubit_counts = [2, 3, 4, 5, 6, 8, 10, 12, 16]
        self.naming_styles = ['short', 'descriptive', 'verbose']
        self.comment_styles = ['none', 'minimal', 'extensive']
        
        self.dataset = []
    
    def generate_all_datasets(self, variations_per_algo: int = 100):
        """Generate complete dataset for all algorithms and languages"""
        
        SUPPORTED_ALGORITHMS = {
            "grover": [
                "qiskit",
                "cirq",
                "openqasm",
                "qsharp"
            ],

            "shor": [
                "qiskit"
            ],

            "qft": [
                "qiskit",
                "cirq",
                "openqasm",
                "qsharp"
            ],

            "vqe": [
                "qiskit"
            ],

            "qaoa": [
                "qiskit"
            ],

            "qpe": [
                "qiskit"
            ],

            "deutsch_jozsa": [
                "qiskit",
                "cirq",
                "openqasm",
                "qsharp"
            ],

            "bernstein_vazirani": [
                "qiskit",
                "cirq",
                "openqasm",
                "qsharp"
            ],

            "simon": [
                "qiskit"
            ],

            "amplitude_amplification": [
                "qiskit"
            ]
        }

        algorithms = list(SUPPORTED_ALGORITHMS.keys())
        print("=" * 80)
        print("QUANTUM ALGORITHM DATASET GENERATION")
        print("=" * 80)
        print(f"Target: {variations_per_algo} variations per supported algorithm-language pair")
        print(f"Algorithms: {len(algorithms)}")
        all_supported_languages = sorted(
            {lang for langs in SUPPORTED_ALGORITHMS.values() for lang in langs}
        )

        print(f"Languages (overall supported): {len(all_supported_languages)} → {all_supported_languages}")

        total_supported_pairs = sum(
            len(langs) for langs in SUPPORTED_ALGORITHMS.values()
        )
        total_codes = total_supported_pairs * variations_per_algo

        print(f"Total supported algorithm-language pairs: {total_supported_pairs}")
        print(f"Total codes to generate: {total_codes}")
        print("=" * 80)
        
        for algorithm in algorithms:
            print(f"\n📊 Generating {algorithm}...")
            
            for language in SUPPORTED_ALGORITHMS[algorithm]:
                generator_method = getattr(self, f"generate_{algorithm}_{language}", None)
                
                if generator_method:
                    codes = generator_method(variations_per_algo)
                    self._save_algorithm_dataset(algorithm, language, codes)
                    print(f"  ✅ {language}: {len(codes)} variations")
                else:
                    print(f"  ⚠️  {language}: Generator not implemented yet")
        
        # Save metadata
        self._save_metadata()
        print("\n" + "=" * 80)
        print(f"✅ Dataset generation complete!")
        print(f"📁 Location: {self.output_dir}")
        print(f"📊 Total samples: {len(self.dataset)}")
        print("=" * 80)
    
    # ========================================================================
    # GROVER'S ALGORITHM - All Languages
    # ========================================================================
    
    def generate_grover_qiskit(self, count: int) -> List[Tuple[str, Dict]]:
        """Generate Grover's algorithm variations in Qiskit"""
        codes = []
        
        for i in range(count):
            n_qubits = random.choice(self.qubit_counts[:6])  # Up to 8 qubits for Grover
            naming = random.choice(self.naming_styles)
            comments = random.choice(self.comment_styles)
            
            # Variations
            use_custom_oracle = random.choice([True, False])
            use_barrier = random.choice([True, False])
            measure_all = random.choice([True, False])
            use_initialize = random.choice([True, False])
            
            code = self._generate_grover_qiskit_code(
                n_qubits, naming, comments, use_custom_oracle, 
                use_barrier, measure_all, use_initialize
            )
            
            metadata = {
                'algorithm': 'grover',
                'language': 'qiskit',
                'qubits': n_qubits,
                'problem_type': 'search',
                'variation_id': i
            }
            
            codes.append((code, metadata))
        
        return codes
    
    def _generate_grover_qiskit_code(
        self, n_qubits: int, naming: str, comments: str,
        custom_oracle: bool, use_barrier: bool, measure_all: bool,
        use_initialize: bool
    ) -> str:
        """Generate actual Qiskit Grover code"""
        
        # Variable naming based on style
        if naming == 'short':
            qc_name = 'qc'
            qr_name = 'q'
            cr_name = 'c'
        elif naming == 'descriptive':
            qc_name = 'circuit'
            qr_name = 'qreg'
            cr_name = 'creg'
        else:  # verbose
            qc_name = 'grover_circuit'
            qr_name = 'quantum_register'
            cr_name = 'classical_register'
        
        # Comment style
        header_comment = ""
        if comments in ['minimal', 'extensive']:
            header_comment = f"# Grover's algorithm for {n_qubits} qubits\n"
        
        inline_comments = comments == 'extensive'
        
        # Build code
        lines = []
        
        if header_comment:
            lines.append(header_comment)
        
        lines.append("from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister")
        
        if use_initialize:
            lines.append("from qiskit.circuit.library import GroverOperator")
        
        lines.append("")
        
        # Create circuit
        lines.append(f"{qr_name} = QuantumRegister({n_qubits}, 'q')")
        lines.append(f"{cr_name} = ClassicalRegister({n_qubits}, 'c')")
        lines.append(f"{qc_name} = QuantumCircuit({qr_name}, {cr_name})")
        lines.append("")
        
        # Superposition
        if inline_comments:
            lines.append("# Create superposition")
        
        lines.append(f"for i in range({n_qubits}):")
        lines.append(f"    {qc_name}.h(i)")
        
        if use_barrier:
            lines.append(f"{qc_name}.barrier()")
        
        lines.append("")
        
        # Oracle (with variations)
        if inline_comments:
            lines.append("# Oracle: marks the target state")
        
        if custom_oracle:
            # Custom oracle variation
            target_state = random.randint(0, 2**n_qubits - 1)
            lines.append(f"# Marks state |{target_state:0{n_qubits}b}>")
            
            # Generate oracle based on target state
            for bit_idx in range(n_qubits):
                if not (target_state & (1 << bit_idx)):
                    lines.append(f"{qc_name}.x({bit_idx})")
            
            if n_qubits == 2:
                lines.append(f"{qc_name}.cz(0, 1)")
            elif n_qubits == 3:
                lines.append(f"{qc_name}.ccx(0, 1, 2)")
                lines.append(f"{qc_name}.z(2)")
                lines.append(f"{qc_name}.ccx(0, 1, 2)")
            else:
                lines.append(f"{qc_name}.mct(list(range({n_qubits-1})), {n_qubits-1})")
                lines.append(f"{qc_name}.z({n_qubits-1})")
            
            for bit_idx in range(n_qubits):
                if not (target_state & (1 << bit_idx)):
                    lines.append(f"{qc_name}.x({bit_idx})")
        else:
            # Simple oracle
            lines.append(f"{qc_name}.cz(0, 1)" if n_qubits == 2 else f"{qc_name}.mct(list(range({n_qubits-1})), {n_qubits-1})")
        
        if use_barrier:
            lines.append(f"{qc_name}.barrier()")
        
        lines.append("")
        
        # Diffusion operator
        if inline_comments:
            lines.append("# Diffusion operator")
        
        lines.append(f"for i in range({n_qubits}):")
        lines.append(f"    {qc_name}.h(i)")
        lines.append(f"for i in range({n_qubits}):")
        lines.append(f"    {qc_name}.x(i)")
        
        lines.append(f"{qc_name}.h({n_qubits-1})")
        
        if n_qubits == 2:
            lines.append(f"{qc_name}.cx(0, 1)")
        else:
            lines.append(f"{qc_name}.mct(list(range({n_qubits-1})), {n_qubits-1})")
        
        lines.append(f"{qc_name}.h({n_qubits-1})")
        
        lines.append(f"for i in range({n_qubits}):")
        lines.append(f"    {qc_name}.x(i)")
        lines.append(f"for i in range({n_qubits}):")
        lines.append(f"    {qc_name}.h(i)")
        
        if use_barrier:
            lines.append(f"{qc_name}.barrier()")
        
        lines.append("")
        
        # Measurement
        if inline_comments:
            lines.append("# Measurement")
        
        if measure_all:
            lines.append(f"{qc_name}.measure_all()")
        else:
            lines.append(f"{qc_name}.measure({qr_name}, {cr_name})")
        
        return "\n".join(lines)
    
    def generate_grover_cirq(self, count: int) -> List[Tuple[str, Dict]]:
        """Generate Grover's algorithm variations in Cirq"""
        codes = []
        
        for i in range(count):
            n_qubits = random.choice(self.qubit_counts[:6])
            use_line_qubits = random.choice([True, False])
            use_moment = random.choice([True, False])
            
            code = self._generate_grover_cirq_code(n_qubits, use_line_qubits, use_moment)
            
            metadata = {
                'algorithm': 'grover',
                'language': 'cirq',
                'qubits': n_qubits,
                'problem_type': 'search',
                'variation_id': i
            }
            
            codes.append((code, metadata))
        
        return codes
    
    def _generate_grover_cirq_code(
        self, n_qubits: int, use_line_qubits: bool, use_moment: bool
    ) -> str:
        """Generate Cirq Grover code"""
        
        lines = []
        lines.append("import cirq")
        lines.append("")
        
        # Qubit declaration
        if use_line_qubits:
            lines.append(f"qubits = cirq.LineQubit.range({n_qubits})")
        else:
            lines.append(f"qubits = [cirq.GridQubit(0, i) for i in range({n_qubits})]")
        
        lines.append("circuit = cirq.Circuit()")
        lines.append("")
        
        # Superposition
        if use_moment:
            lines.append("circuit.append([cirq.H(q) for q in qubits])")
        else:
            lines.append("for q in qubits:")
            lines.append("    circuit.append(cirq.H(q))")
        
        lines.append("")
        
        # Oracle
        lines.append("# Oracle")
        lines.append(f"circuit.append(cirq.CZ(qubits[0], qubits[1]))")
        lines.append("")
        
        # Diffusion
        lines.append("# Diffusion operator")
        lines.append("circuit.append([cirq.H(q) for q in qubits])")
        lines.append("circuit.append([cirq.X(q) for q in qubits])")
        lines.append(f"circuit.append(cirq.CZ(qubits[0], qubits[1]))")
        lines.append("circuit.append([cirq.X(q) for q in qubits])")
        lines.append("circuit.append([cirq.H(q) for q in qubits])")
        lines.append("")
        
        # Measurement
        lines.append("circuit.append(cirq.measure(*qubits, key='result'))")
        
        return "\n".join(lines)
    
    def generate_grover_openqasm(self, count: int) -> List[Tuple[str, Dict]]:
        """Generate Grover's algorithm variations in OpenQASM"""
        codes = []
        
        for i in range(count):
            n_qubits = random.choice(self.qubit_counts[:5])  # Smaller for QASM
            version = random.choice(['2.0', '3.0'])
            
            code = self._generate_grover_openqasm_code(n_qubits, version)
            
            metadata = {
                'algorithm': 'grover',
                'language': 'openqasm',
                'qubits': n_qubits,
                'problem_type': 'search',
                'variation_id': i
            }
            
            codes.append((code, metadata))
        
        return codes
    
    def _generate_grover_openqasm_code(self, n_qubits: int, version: str) -> str:
        """Generate OpenQASM Grover code"""
        
        lines = []
        lines.append(f"OPENQASM {version};")
        lines.append('include "qelib1.inc";')
        lines.append("")
        lines.append(f"qreg q[{n_qubits}];")
        lines.append(f"creg c[{n_qubits}];")
        lines.append("")
        
        # Superposition
        for i in range(n_qubits):
            lines.append(f"h q[{i}];")
        
        lines.append("")
        
        # Oracle
        lines.append("// Oracle")
        lines.append("cz q[0], q[1];")
        lines.append("")
        
        # Diffusion
        lines.append("// Diffusion")
        for i in range(n_qubits):
            lines.append(f"h q[{i}];")
        for i in range(n_qubits):
            lines.append(f"x q[{i}];")
        
        lines.append("h q[1];")
        lines.append("cx q[0], q[1];")
        lines.append("h q[1];")
        
        for i in range(n_qubits):
            lines.append(f"x q[{i}];")
        for i in range(n_qubits):
            lines.append(f"h q[{i}];")
        
        lines.append("")
        
        # Measurement
        for i in range(n_qubits):
            lines.append(f"measure q[{i}] -> c[{i}];")
        
        return "\n".join(lines)
    
    def generate_grover_qsharp(self, count: int) -> List[Tuple[str, Dict]]:
        """Generate Grover's algorithm variations in Q#"""
        codes = []
        
        for i in range(count):
            n_qubits = random.choice(self.qubit_counts[:6])
            
            code = self._generate_grover_qsharp_code(n_qubits)
            
            metadata = {
                'algorithm': 'grover',
                'language': 'qsharp',
                'qubits': n_qubits,
                'problem_type': 'search',
                'variation_id': i
            }
            
            codes.append((code, metadata))
        
        return codes
    
    def _generate_grover_qsharp_code(self, n_qubits: int) -> str:
        """Generate Q# Grover code"""
        
        lines = []
        lines.append("namespace GroverSearch {")
        lines.append("    open Microsoft.Quantum.Canon;")
        lines.append("    open Microsoft.Quantum.Intrinsic;")
        lines.append("")
        lines.append("    operation GroverSearch() : Result[] {")
        lines.append(f"        using (qubits = Qubit[{n_qubits}]) {{")
        lines.append("            ")
        lines.append("            // Superposition")
        lines.append("            ApplyToEach(H, qubits);")
        lines.append("            ")
        lines.append("            // Oracle")
        lines.append("            CZ(qubits[0], qubits[1]);")
        lines.append("            ")
        lines.append("            // Diffusion operator")
        lines.append("            ApplyToEach(H, qubits);")
        lines.append("            ApplyToEach(X, qubits);")
        lines.append("            ")
        lines.append(f"            Controlled Z([qubits[0]], qubits[1]);")
        lines.append("            ")
        lines.append("            ApplyToEach(X, qubits);")
        lines.append("            ApplyToEach(H, qubits);")
        lines.append("            ")
        lines.append("            return MultiM(qubits);")
        lines.append("        }")
        lines.append("    }")
        lines.append("}")
        
        return "\n".join(lines)
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _save_algorithm_dataset(self, algorithm: str, language: str, codes: List[Tuple[str, Dict]]):
        """Save generated codes to files"""
        
        algo_dir = self.output_dir / algorithm / language
        algo_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (code, metadata) in enumerate(codes):
            # Save code file
            code_file = algo_dir / f"{algorithm}_{language}_{i:04d}.txt"
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Add to dataset
            self.dataset.append({
                'file': str(code_file),
                'code': code,
                'metadata': metadata
            })
    
    def _save_metadata(self):
        """Save dataset metadata"""
        
        metadata_file = self.output_dir / "dataset_metadata.json"
        
        summary = {
            'total_samples': len(self.dataset),
            'algorithms': {},
            'languages': {},
            'samples': self.dataset
        }
        
        # Count by algorithm
        for sample in self.dataset:
            algo = sample['metadata']['algorithm']
            summary['algorithms'][algo] = summary['algorithms'].get(algo, 0) + 1
        
        # Count by language
        for sample in self.dataset:
            lang = sample['metadata']['language']
            summary['languages'][lang] = summary['languages'].get(lang, 0) + 1
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Metadata saved to: {metadata_file}")

    """
    Additional Quantum Algorithm Generators
    Extends the base dataset generator with QFT, VQE, QAOA, Shor, and more
    """

    # Add these methods to the QuantumAlgorithmDatasetGenerator class

    # ========================================================================
    # QUANTUM FOURIER TRANSFORM (QFT)
    # ========================================================================

    def generate_qft_qiskit(self, count: int) -> List[Tuple[str, Dict]]:
        """Generate QFT variations in Qiskit"""
        codes = []
        
        for i in range(count):
            n_qubits = random.choice([3, 4, 5, 6, 8])
            use_swaps = random.choice([True, False])
            inverse = random.choice([True, False])
            
            code = self._generate_qft_qiskit_code(n_qubits, use_swaps, inverse)
            
            metadata = {
                'algorithm': 'qft',
                'language': 'qiskit',
                'qubits': n_qubits,
                'problem_type': 'simulation',
                'variation_id': i
            }
            
            codes.append((code, metadata))
        
        return codes

    def _generate_qft_qiskit_code(self, n_qubits: int, use_swaps: bool, inverse: bool) -> str:
        """Generate QFT Qiskit code"""
        lines = []
        lines.append("from qiskit import QuantumCircuit")
        lines.append("import numpy as np")
        lines.append("")
        lines.append(f"qc = QuantumCircuit({n_qubits})")
        lines.append("")
        
        if inverse:
            lines.append("# Inverse QFT")
        else:
            lines.append("# Quantum Fourier Transform")
        
        lines.append("")
        
        if inverse:
            # Inverse QFT
            if use_swaps:
                lines.append("# Swap qubits")
                for i in range(n_qubits // 2):
                    lines.append(f"qc.swap({i}, {n_qubits - i - 1})")
                lines.append("")
            
            for j in range(n_qubits - 1, -1, -1):
                lines.append(f"# Qubit {j}")
                for k in range(j - 1, -1, -1):
                    power = j - k + 1
                    lines.append(f"qc.cp(-np.pi/{2**power}, {k}, {j})")
                lines.append(f"qc.h({j})")
                lines.append("")
        else:
            # Forward QFT
            for j in range(n_qubits):
                lines.append(f"# Qubit {j}")
                lines.append(f"qc.h({j})")
                for k in range(j + 1, n_qubits):
                    power = k - j + 1
                    lines.append(f"qc.cp(np.pi/{2**power}, {j}, {k})")
                lines.append("")
            
            if use_swaps:
                lines.append("# Swap qubits")
                for i in range(n_qubits // 2):
                    lines.append(f"qc.swap({i}, {n_qubits - i - 1})")
        
        lines.append("")
        lines.append("qc.measure_all()")
        
        return "\n".join(lines)

    # ========================================================================
    # VARIATIONAL QUANTUM EIGENSOLVER (VQE)
    # ========================================================================

    def generate_vqe_qiskit(self, count: int) -> List[Tuple[str, Dict]]:
        """Generate VQE variations in Qiskit"""
        codes = []
        
        for i in range(count):
            n_qubits = random.choice([2, 3, 4, 6])
            ansatz_type = random.choice(['ry', 'rx_ry', 'efficient_su2'])
            entangling_gate = random.choice(['cx', 'cz'])
            depth = random.choice([1, 2, 3])
            
            code = self._generate_vqe_qiskit_code(n_qubits, ansatz_type, entangling_gate, depth)
            
            metadata = {
                'algorithm': 'vqe',
                'language': 'qiskit',
                'qubits': n_qubits,
                'problem_type': 'optimization',
                'variation_id': i
            }
            
            codes.append((code, metadata))
        
        return codes

    def _generate_vqe_qiskit_code(self, n_qubits: int, ansatz_type: str, 
                                entangling_gate: str, depth: int) -> str:
        """Generate VQE Qiskit code"""
        lines = []
        lines.append("from qiskit import QuantumCircuit")
        lines.append("from qiskit.circuit import Parameter")
        lines.append("import numpy as np")
        lines.append("")
        lines.append(f"# VQE Ansatz for {n_qubits} qubits")
        lines.append(f"qc = QuantumCircuit({n_qubits})")
        lines.append("")
        
        param_count = 0
        
        for layer in range(depth):
            lines.append(f"# Layer {layer + 1}")
            
            # Rotation layer
            if ansatz_type == 'ry':
                for q in range(n_qubits):
                    lines.append(f"theta_{param_count} = Parameter('θ_{param_count}')")
                    lines.append(f"qc.ry(theta_{param_count}, {q})")
                    param_count += 1
            elif ansatz_type == 'rx_ry':
                for q in range(n_qubits):
                    lines.append(f"theta_{param_count} = Parameter('θ_{param_count}')")
                    lines.append(f"qc.rx(theta_{param_count}, {q})")
                    param_count += 1
                    lines.append(f"theta_{param_count} = Parameter('θ_{param_count}')")
                    lines.append(f"qc.ry(theta_{param_count}, {q})")
                    param_count += 1
            else:  # efficient_su2
                for q in range(n_qubits):
                    lines.append(f"theta_{param_count} = Parameter('θ_{param_count}')")
                    lines.append(f"qc.rz(theta_{param_count}, {q})")
                    param_count += 1
            
            lines.append("")
            
            # Entangling layer
            lines.append("# Entangling layer")
            for q in range(n_qubits - 1):
                if entangling_gate == 'cx':
                    lines.append(f"qc.cx({q}, {q + 1})")
                else:
                    lines.append(f"qc.cz({q}, {q + 1})")
            
            lines.append("")
        
        lines.append("# Measurement")
        lines.append("qc.measure_all()")
        
        return "\n".join(lines)

    # ========================================================================
    # QAOA (Quantum Approximate Optimization Algorithm)
    # ========================================================================

    def generate_qaoa_qiskit(self, count: int) -> List[Tuple[str, Dict]]:
        """Generate QAOA variations in Qiskit"""
        codes = []
        
        for i in range(count):
            n_qubits = random.choice([3, 4, 5, 6])
            p_layers = random.choice([1, 2, 3])
            problem_type = random.choice(['maxcut', 'portfolio', 'tsp'])
            
            code = self._generate_qaoa_qiskit_code(n_qubits, p_layers, problem_type)
            
            metadata = {
                'algorithm': 'qaoa',
                'language': 'qiskit',
                'qubits': n_qubits,
                'problem_type': 'optimization',
                'variation_id': i
            }
            
            codes.append((code, metadata))
        
        return codes

    def _generate_qaoa_qiskit_code(self, n_qubits: int, p_layers: int, problem_type: str) -> str:
        """Generate QAOA Qiskit code"""
        lines = []
        lines.append("from qiskit import QuantumCircuit")
        lines.append("from qiskit.circuit import Parameter")
        lines.append("import numpy as np")
        lines.append("")
        lines.append(f"# QAOA for {problem_type} problem")
        lines.append(f"qc = QuantumCircuit({n_qubits})")
        lines.append("")
        
        # Initial state (equal superposition)
        lines.append("# Initial superposition")
        lines.append(f"for i in range({n_qubits}):")
        lines.append("    qc.h(i)")
        lines.append("")
        
        for p in range(p_layers):
            lines.append(f"# QAOA layer {p + 1}")
            
            # Problem Hamiltonian (cost layer)
            lines.append("# Problem Hamiltonian")
            lines.append(f"gamma_{p} = Parameter('γ_{p}')")
            
            # Add ZZ interactions based on problem
            if problem_type == 'maxcut':
                for i in range(n_qubits - 1):
                    for j in range(i + 1, n_qubits):
                        lines.append(f"qc.rzz(gamma_{p}, {i}, {j})")
            else:
                for i in range(n_qubits - 1):
                    lines.append(f"qc.rzz(gamma_{p}, {i}, {i + 1})")
            
            lines.append("")
            
            # Mixing Hamiltonian
            lines.append("# Mixing Hamiltonian")
            lines.append(f"beta_{p} = Parameter('β_{p}')")
            lines.append(f"for i in range({n_qubits}):")
            lines.append(f"    qc.rx(beta_{p}, i)")
            lines.append("")
        
        lines.append("# Measurement")
        lines.append("qc.measure_all()")
        
        return "\n".join(lines)

    # ========================================================================
    # SHOR'S ALGORITHM
    # ========================================================================

    def generate_shor_qiskit(self, count: int) -> List[Tuple[str, Dict]]:
        """Generate Shor's algorithm variations in Qiskit"""
        codes = []
        
        for i in range(count):
            n_qubits = random.choice([5, 7, 9, 11])  # Needs more qubits
            number_to_factor = random.choice([15, 21, 35])
            
            code = self._generate_shor_qiskit_code(n_qubits, number_to_factor)
            
            metadata = {
                'algorithm': 'shor',
                'language': 'qiskit',
                'qubits': n_qubits,
                'problem_type': 'factorization',
                'variation_id': i
            }
            
            codes.append((code, metadata))
        
        return codes

    def _generate_shor_qiskit_code(self, n_qubits: int, N: int) -> str:
        """Generate Shor's Qiskit code"""
        lines = []
        lines.append("from qiskit import QuantumCircuit")
        lines.append("import numpy as np")
        lines.append("")
        lines.append(f"# Shor's algorithm for factoring {N}")
        
        counting_qubits = n_qubits // 2
        work_qubits = n_qubits - counting_qubits
        
        lines.append(f"qc = QuantumCircuit({n_qubits}, {counting_qubits})")
        lines.append("")
        
        # Initialize superposition on counting qubits
        lines.append("# Superposition on counting qubits")
        for i in range(counting_qubits):
            lines.append(f"qc.h({i})")
        lines.append("")
        
        # Modular exponentiation (simplified)
        lines.append("# Modular exponentiation (controlled)")
        for i in range(counting_qubits):
            lines.append(f"# Controlled U^(2^{i})")
            lines.append(f"qc.cx({i}, {counting_qubits})")
        
        lines.append("")
        
        # Inverse QFT on counting qubits
        lines.append("# Inverse QFT")
        for j in range(counting_qubits - 1, -1, -1):
            for k in range(j - 1, -1, -1):
                power = j - k + 1
                lines.append(f"qc.cp(-np.pi/{2**power}, {k}, {j})")
            lines.append(f"qc.h({j})")
        
        lines.append("")
        
        # Measurement
        lines.append("# Measure counting qubits")
        for i in range(counting_qubits):
            lines.append(f"qc.measure({i}, {i})")
        
        return "\n".join(lines)

    # ========================================================================
    # DEUTSCH-JOZSA ALGORITHM
    # ========================================================================

    def generate_deutsch_jozsa_qiskit(self, count: int) -> List[Tuple[str, Dict]]:
        """Generate Deutsch-Jozsa variations in Qiskit"""
        codes = []
        
        for i in range(count):
            n_qubits = random.choice([2, 3, 4, 5, 6])
            oracle_type = random.choice(['constant', 'balanced'])
            
            code = self._generate_deutsch_jozsa_qiskit_code(n_qubits, oracle_type)
            
            metadata = {
                'algorithm': 'deutsch_jozsa',
                'language': 'qiskit',
                'qubits': n_qubits + 1,  # +1 for ancilla
                'problem_type': 'search',
                'variation_id': i
            }
            
            codes.append((code, metadata))
        
        return codes

    def _generate_deutsch_jozsa_qiskit_code(self, n_qubits: int, oracle_type: str) -> str:
        """Generate Deutsch-Jozsa Qiskit code"""
        lines = []
        lines.append("from qiskit import QuantumCircuit")
        lines.append("")
        lines.append(f"# Deutsch-Jozsa for {oracle_type} function")
        lines.append(f"qc = QuantumCircuit({n_qubits + 1}, {n_qubits})")
        lines.append("")
        
        # Initialize ancilla to |1>
        lines.append("# Initialize ancilla")
        lines.append(f"qc.x({n_qubits})")
        lines.append("")
        
        # Hadamard on all qubits
        lines.append("# Superposition")
        lines.append(f"for i in range({n_qubits + 1}):")
        lines.append("    qc.h(i)")
        lines.append("")
        
        # Oracle
        lines.append(f"# Oracle ({oracle_type})")
        if oracle_type == 'constant':
            # Do nothing for constant-0, or flip ancilla for constant-1
            if random.choice([True, False]):
                lines.append(f"qc.x({n_qubits})  # Constant-1")
        else:  # balanced
            # Apply CX from half the qubits
            for i in range(n_qubits // 2):
                lines.append(f"qc.cx({i}, {n_qubits})")
        
        lines.append("")
        
        # Hadamard on input qubits
        lines.append("# Hadamard on input qubits")
        lines.append(f"for i in range({n_qubits}):")
        lines.append("    qc.h(i)")
        lines.append("")
        
        # Measurement
        lines.append("# Measurement")
        lines.append(f"for i in range({n_qubits}):")
        lines.append("    qc.measure(i, i)")
        
        return "\n".join(lines)

    # ========================================================================
    # AMPLITUDE AMPLIFICATION
    # ========================================================================

    def generate_amplitude_amplification_qiskit(self, count: int) -> List[Tuple[str, Dict]]:
        """Generate Amplitude Amplification variations in Qiskit"""
        codes = []
        
        for i in range(count):
            n_qubits = random.choice([2, 3, 4, 5])
            iterations = random.choice([1, 2, 3])
            
            code = self._generate_amplitude_amplification_qiskit_code(n_qubits, iterations)
            
            metadata = {
                'algorithm': 'amplitude_amplification',
                'language': 'qiskit',
                'qubits': n_qubits,
                'problem_type': 'search',
                'variation_id': i
            }
            
            codes.append((code, metadata))
        
        return codes

    def _generate_amplitude_amplification_qiskit_code(self, n_qubits: int, iterations: int) -> str:
        """Generate Amplitude Amplification Qiskit code"""
        # Very similar to Grover
        return self._generate_grover_qiskit_code(
            n_qubits, 'descriptive', 'minimal', True, True, True, False
        )
    
    def generate_qft_cirq(self, count: int):
        codes = []

        for idx in range(count):
            n = random.choice([3, 4, 5])

            code = self._generate_qft_cirq_code(n)

            metadata = {
                "algorithm": "qft",
                "language": "cirq",
                "qubits": n,
                "variation_id": idx
            }

            codes.append((code, metadata))

        return codes

    def _generate_qft_cirq_code(self, n: int) -> str:
        lines = [
            "import cirq",
            "",
            f"qubits = cirq.LineQubit.range({n})",
            "circuit = cirq.Circuit()",
            "",
            "# Quantum Fourier Transform"
        ]

        for i in range(n):
            lines.append(f"circuit.append(cirq.H(qubits[{i}]))")
            for j in range(i + 1, n):
                angle = 1 / (2 ** (j - i))
                lines.append(
                    f"circuit.append(cirq.CZ(qubits[{j}], qubits[{i}]) ** {angle})"
                )

        lines.append("")
        lines.append("print(circuit)")

        return "\n".join(lines)

    def generate_qpe_qiskit(self, count: int):
        codes = []

        for idx in range(count):
            n_count = random.choice([3, 4])
            total_qubits = n_count + 1

            code = self._generate_qpe_qiskit_code(n_count)

            metadata = {
                "algorithm": "qpe",
                "language": "qiskit",
                "qubits": total_qubits,
                "variation_id": idx
            }

            codes.append((code, metadata))

        return codes

    def _generate_qpe_qiskit_code(self, n_count: int) -> str:
        total_qubits = n_count + 1

        lines = [
            "from qiskit import QuantumCircuit",
            "from qiskit.circuit.library import QFT",
            "",
            f"qc = QuantumCircuit({total_qubits}, {n_count})",
            "",
            "# Initialize eigenstate |1>",
            f"qc.x({total_qubits - 1})",
            "",
            "# Hadamard on counting qubits",
            f"for i in range({n_count}):",
            "    qc.h(i)",
            "",
            "# Controlled-U operations"
        ]

        for i in range(n_count):
            power = 2 ** i
            lines.append(f"qc.cp({2 * 3.141592653589793 / power}, {i}, {total_qubits - 1})")

        lines += [
            "",
            "# Inverse QFT",
            f"qc.append(QFT({n_count}, inverse=True), range({n_count}))",
            "",
            "# Measurement",
            f"qc.measure(range({n_count}), range({n_count}))"
        ]

        return "\n".join(lines)
    
    def generate_bernstein_vazirani_qiskit(self, count: int):
        codes = []

        for idx in range(count):
            n = random.choice([3, 4, 5])
            secret = [random.choice([0, 1]) for _ in range(n)]

            code = self._generate_bernstein_vazirani_qiskit_code(n, secret)

            metadata = {
                "algorithm": "bernstein_vazirani",
                "language": "qiskit",
                "qubits": n + 1,
                "variation_id": idx
            }

            codes.append((code, metadata))

        return codes

    def _generate_bernstein_vazirani_qiskit_code(self, n: int, secret: list[int]) -> str:
        lines = [
            "from qiskit import QuantumCircuit",
            "",
            f"qc = QuantumCircuit({n + 1}, {n})",
            "",
            "# Initialize ancilla",
            f"qc.x({n})",
            f"qc.h({n})",
            "",
            "# Superposition",
            f"for i in range({n}):",
            "    qc.h(i)",
            "",
            "# Oracle"
        ]

        for i, bit in enumerate(secret):
            if bit == 1:
                lines.append(f"qc.cx({i}, {n})")

        lines += [
            "",
            "# Hadamard",
            f"for i in range({n}):",
            "    qc.h(i)",
            "",
            "# Measurement",
            f"qc.measure(range({n}), range({n}))"
        ]

        return "\n".join(lines)
    
    def generate_simon_qiskit(self, count: int):
        codes = []

        for idx in range(count):
            n = random.choice([3, 4])

            code = self._generate_simon_qiskit_code(n)

            metadata = {
                "algorithm": "simon",
                "language": "qiskit",
                "qubits": 2 * n,
                "variation_id": idx
            }

            codes.append((code, metadata))

        return codes

    def _generate_simon_qiskit_code(self, n: int) -> str:
        total_qubits = 2 * n

        lines = [
            "from qiskit import QuantumCircuit",
            "",
            f"qc = QuantumCircuit({total_qubits}, {n})",
            "",
            "# Superposition",
            f"for i in range({n}):",
            "    qc.h(i)",
            "",
            "# Oracle (simple Simon oracle)"
        ]

        for i in range(n):
            lines.append(f"qc.cx({i}, {i + n})")

        lines += [
            "",
            "# Hadamard on input register",
            f"for i in range({n}):",
            "    qc.h(i)",
            "",
            "# Measurement",
            f"qc.measure(range({n}), range({n}))"
        ]

        return "\n".join(lines)
    
    def generate_qft_openqasm(self, count: int):
        codes = []

        for idx in range(count):
            n = random.choice([3, 4, 5])
            version = random.choice(["2.0", "3.0"])

            code = self._generate_qft_openqasm_code(n, version)

            codes.append((
                code,
                {
                    "algorithm": "qft",
                    "language": "openqasm",
                    "qubits": n,
                    "qasm_version": version,
                    "variation_id": idx
                }
            ))

        return codes
    
    def _generate_qft_openqasm_code(self, n: int, version: str) -> str:
        lines = []

        if version == "2.0":
            lines.append("OPENQASM 2.0;")
            lines.append("include \"qelib1.inc\";")
        else:
            lines.append("OPENQASM 3.0;")
            lines.append("include \"stdgates.inc\";")

        lines.append("")
        lines.append(f"qreg q[{n}];")
        lines.append(f"creg c[{n}];")
        lines.append("")
        lines.append("// Quantum Fourier Transform")

        for i in range(n):
            lines.append(f"h q[{i}];")
            for j in range(i + 1, n):
                angle = 3.141592653589793 / (2 ** (j - i))
                lines.append(f"cp({angle}) q[{j}], q[{i}];")
            lines.append("")

        for i in range(n // 2):
            lines.append(f"swap q[{i}], q[{n - i - 1}];")

        lines.append("")
        for i in range(n):
            lines.append(f"measure q[{i}] -> c[{i}];")

        return "\n".join(lines)
    
    def generate_qft_qsharp(self, count: int):
        codes = []

        for idx in range(count):
            n = random.choice([3, 4, 5])
            code = self._generate_qft_qsharp_code(n)

            codes.append((
                code,
                {
                    "algorithm": "qft",
                    "language": "qsharp",
                    "qubits": n,
                    "variation_id": idx
                }
            ))

        return codes
    
    def _generate_qft_qsharp_code(self, n: int) -> str:
        lines = [
            "namespace Quantum.Algorithms {",
            "    open Microsoft.Quantum.Intrinsic;",
            "    open Microsoft.Quantum.Canon;",
            "",
            "    operation QFT(qubits : Qubit[]) : Unit {",
            "        let n = Length(qubits);",
            "        for (i in 0 .. n - 1) {",
            "            H(qubits[i]);",
            "            for (j in i + 1 .. n - 1) {",
            "                let angle = 1.0 / IntAsDouble(2 ^ (j - i));",
            "                Controlled R1([qubits[j]], (2.0 * PI() * angle, qubits[i]));",
            "            }",
            "        }",
            "",
            "        for (i in 0 .. n / 2 - 1) {",
            "            SWAP(qubits[i], qubits[n - i - 1]);",
            "        }",
            "    }",
            "",
            "}"
        ]

        return "\n".join(lines)
    
    def generate_deutsch_jozsa_cirq(self, count: int):
        codes = []

        for idx in range(count):
            n = random.choice([3, 4, 5])
            oracle_type = random.choice(["constant", "balanced"])

            code = self._generate_deutsch_jozsa_cirq_code(n, oracle_type)

            codes.append((
                code,
                {
                    "algorithm": "deutsch_jozsa",
                    "language": "cirq",
                    "qubits": n + 1,
                    "oracle_type": oracle_type,
                    "variation_id": idx
                }
            ))

        return codes
    
    def _generate_deutsch_jozsa_cirq_code(self, n: int, oracle_type: str) -> str:
        lines = [
            "import cirq",
            "",
            f"qubits = cirq.LineQubit.range({n + 1})",
            "circuit = cirq.Circuit()",
            "",
            "# Initialize ancilla",
            f"circuit.append(cirq.X(qubits[{n}]))",
            f"circuit.append(cirq.H(qubits[{n}]))",
            "",
            "# Superposition",
            f"for i in range({n}):",
            "    circuit.append(cirq.H(qubits[i]))",
            "",
            "# Oracle"
        ]

        if oracle_type == "balanced":
            for i in range(n):
                lines.append(f"circuit.append(cirq.CNOT(qubits[{i}], qubits[{n}]))")

        lines += [
            "",
            "# Interference",
            f"for i in range({n}):",
            "    circuit.append(cirq.H(qubits[i]))",
            "",
            "print(circuit)"
        ]

        return "\n".join(lines)
    
    def generate_deutsch_jozsa_openqasm(self, count: int):
        codes = []

        for idx in range(count):
            n = random.choice([3, 4, 5])
            oracle_type = random.choice(["constant", "balanced"])

            code = self._generate_deutsch_jozsa_openqasm_code(n, oracle_type)

            codes.append((
                code,
                {
                    "algorithm": "deutsch_jozsa",
                    "language": "openqasm",
                    "qubits": n + 1,
                    "oracle_type": oracle_type,
                    "variation_id": idx
                }
            ))

        return codes
    
    def _generate_deutsch_jozsa_openqasm_code(self, n: int, oracle_type: str) -> str:
        lines = [
            "OPENQASM 2.0;",
            "include \"qelib1.inc\";",
            "",
            f"qreg q[{n + 1}];",
            f"creg c[{n}];",
            "",
            f"x q[{n}];",
            f"h q[{n}];"
        ]

        for i in range(n):
            lines.append(f"h q[{i}];")

        if oracle_type == "balanced":
            for i in range(n):
                lines.append(f"cx q[{i}], q[{n}];")

        for i in range(n):
            lines.append(f"h q[{i}];")
            lines.append(f"measure q[{i}] -> c[{i}];")

        return "\n".join(lines)
    
    def generate_deutsch_jozsa_qsharp(self, count: int):
        codes = []

        for idx in range(count):
            n = random.choice([3, 4, 5])
            code = self._generate_deutsch_jozsa_qsharp_code(n)

            codes.append((
                code,
                {
                    "algorithm": "deutsch_jozsa",
                    "language": "qsharp",
                    "qubits": n + 1,
                    "variation_id": idx
                }
            ))

        return codes
    
    def _generate_deutsch_jozsa_qsharp_code(self, n: int) -> str:
        return f"""
    namespace Quantum.Algorithms {{
        open Microsoft.Quantum.Intrinsic;

        operation DeutschJozsa(qubits : Qubit[]) : Unit {{
            let n = Length(qubits) - 1;
            X(qubits[n]);
            H(qubits[n]);

            for (i in 0 .. n - 1) {{
                H(qubits[i]);
            }}

            for (i in 0 .. n - 1) {{
                CNOT(qubits[i], qubits[n]);
            }}

            for (i in 0 .. n - 1) {{
                H(qubits[i]);
            }}
        }}
    }}
    """.strip()

    def generate_bernstein_vazirani_cirq(self, count: int):
        codes = []

        for idx in range(count):
            n = random.choice([3, 4, 5])
            secret = [random.choice([0, 1]) for _ in range(n)]

            code = self._generate_bernstein_vazirani_cirq_code(n, secret)

            codes.append((
                code,
                {
                    "algorithm": "bernstein_vazirani",
                    "language": "cirq",
                    "qubits": n + 1,
                    "secret": secret,
                    "variation_id": idx
                }
            ))

        return codes
    
    def _generate_bernstein_vazirani_cirq_code(self, n: int, secret: list) -> str:
        lines = [
            "import cirq",
            "",
            f"qubits = cirq.LineQubit.range({n + 1})",
            "circuit = cirq.Circuit()",
            "",
            f"circuit.append(cirq.X(qubits[{n}]))",
            f"circuit.append(cirq.H(qubits[{n}]))"
        ]

        for i in range(n):
            lines.append(f"circuit.append(cirq.H(qubits[{i}]))")

        for i, bit in enumerate(secret):
            if bit:
                lines.append(f"circuit.append(cirq.CNOT(qubits[{i}], qubits[{n}]))")

        for i in range(n):
            lines.append(f"circuit.append(cirq.H(qubits[{i}]))")

        lines.append("print(circuit)")
        return "\n".join(lines)
    
    def generate_bernstein_vazirani_openqasm(self, count: int):
        codes = []

        for idx in range(count):
            n = random.choice([3, 4, 5])
            secret = [random.choice([0, 1]) for _ in range(n)]

            code = self._generate_bernstein_vazirani_openqasm_code(n, secret)

            codes.append((
                code,
                {
                    "algorithm": "bernstein_vazirani",
                    "language": "openqasm",
                    "qubits": n + 1,
                    "secret": secret,
                    "variation_id": idx
                }
            ))

        return codes
    
    def _generate_bernstein_vazirani_openqasm_code(self, n: int, secret: list) -> str:
        lines = [
            "OPENQASM 2.0;",
            "include \"qelib1.inc\";",
            "",
            f"qreg q[{n + 1}];",
            f"creg c[{n}];",
            "",
            f"x q[{n}];",
            f"h q[{n}];"
        ]

        for i in range(n):
            lines.append(f"h q[{i}];")

        for i, bit in enumerate(secret):
            if bit:
                lines.append(f"cx q[{i}], q[{n}];")

        for i in range(n):
            lines.append(f"h q[{i}];")
            lines.append(f"measure q[{i}] -> c[{i}];")

        return "\n".join(lines)
    
    def generate_bernstein_vazirani_qsharp(self, count: int):
        codes = []

        for idx in range(count):
            n = random.choice([3, 4, 5])
            code = self._generate_bernstein_vazirani_qsharp_code(n)

            codes.append((
                code,
                {
                    "algorithm": "bernstein_vazirani",
                    "language": "qsharp",
                    "qubits": n + 1,
                    "variation_id": idx
                }
            ))

        return codes
    
    def _generate_bernstein_vazirani_qsharp_code(self, n: int) -> str:
        return f"""
    namespace Quantum.Algorithms {{
        open Microsoft.Quantum.Intrinsic;

        operation BernsteinVazirani(qubits : Qubit[]) : Unit {{
            let n = Length(qubits) - 1;
            X(qubits[n]);
            H(qubits[n]);

            for (i in 0 .. n - 1) {{
                H(qubits[i]);
                CNOT(qubits[i], qubits[n]);
                H(qubits[i]);
            }}
        }}
    }}
    """.strip()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    generator = QuantumAlgorithmDatasetGenerator()
    
    # Generate 100 variations per algorithm per language
    generator.generate_all_datasets(variations_per_algo=100)
    
    print("\n🎉 Dataset generation complete!")
    print("📊 Next steps:")
    print("   1. Extract features from all generated codes")
    print("   2. Train Random Forest + Gradient Boosting models")
    print("   3. Evaluate model accuracy")