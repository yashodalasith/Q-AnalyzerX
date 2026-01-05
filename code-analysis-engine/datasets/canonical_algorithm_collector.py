"""
Canonical Quantum Algorithm Collector
Searches and collects textbook-correct implementations from trusted sources
"""
import os
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

@dataclass
class CanonicalImplementation:
    """Represents a verified canonical quantum algorithm implementation"""
    algorithm: str
    language: str
    code: str
    source_url: str
    source_type: str  # 'official_docs', 'textbook', 'tutorial', 'sample_repo'
    verified: bool
    hash: str
    qubit_count: int
    metadata: Dict

class TrustedSource:
    """Defines trusted sources for quantum algorithms"""
    
    QISKIT_SOURCES = {
        'textbook': 'https://qiskit.org/textbook/',
        'tutorials': 'https://qiskit.org/documentation/tutorials.html',
        'github': 'https://github.com/Qiskit/qiskit-tutorials'
    }
    
    QSHARP_SOURCES = {
        'samples': 'https://github.com/microsoft/Quantum',
        'docs': 'https://docs.microsoft.com/quantum/',
        'katas': 'https://github.com/microsoft/QuantumKatas'
    }
    
    CIRQ_SOURCES = {
        'docs': 'https://quantumai.google/cirq',
        'github': 'https://github.com/quantumlib/Cirq',
        'tutorials': 'https://quantumai.google/cirq/tutorials'
    }
    
    OPENQASM_SOURCES = {
        'github': 'https://github.com/Qiskit/openqasm',
        'examples': 'https://github.com/Qiskit/qiskit-terra/tree/main/qiskit/qasm'
    }

class AlgorithmSearchPatterns:
    """Search patterns for each algorithm type"""
    
    PATTERNS = {
        'grover': {
            'qiskit': [
                'GroverOperator',
                'grover',
                'oracle.*diffusion',
                'amplitude.*amplification.*search'
            ],
            'cirq': [
                'grover',
                'oracle.*diffusion'
            ],
            'qsharp': [
                'GroverSearch',
                'ReflectAboutMarked'
            ],
            'openqasm': [
                '// Grover',
                'oracle',
                'diffusion'
            ]
        },
        'deutsch_jozsa': {
            'qiskit': ['DeutschJozsa', 'deutsch_jozsa', 'dj_oracle'],
            'cirq': ['deutsch_jozsa', 'DeutschJozsa'],
            'qsharp': ['DeutschJozsaAlgorithm'],
            'openqasm': ['// Deutsch-Jozsa']
        },
        'bernstein_vazirani': {
            'qiskit': ['BernsteinVazirani', 'bernstein_vazirani', 'bv_oracle'],
            'cirq': ['bernstein_vazirani'],
            'qsharp': ['BernsteinVaziraniAlgorithm'],
            'openqasm': ['// Bernstein-Vazirani']
        },
        'simon': {
            'qiskit': ['SimonAlgorithm', 'simon', 'simon_oracle'],
            'cirq': ['simon'],
            'qsharp': ['SimonAlgorithm'],
            'openqasm': ['// Simon']
        },
        'shor': {
            'qiskit': ['Shor', 'shor', 'period_finding', 'modular_exponentiation'],
            'cirq': ['shor'],
            'qsharp': ['ShorAlgorithm'],
            'openqasm': []
        },
        'qft': {
            'qiskit': ['QFT', 'quantum_fourier_transform'],
            'cirq': ['quantum_fourier_transform', 'qft'],
            'qsharp': ['QFT', 'QuantumFourierTransform'],
            'openqasm': ['// QFT', '// Quantum Fourier']
        },
        'qpe': {
            'qiskit': ['QPE', 'PhaseEstimation', 'phase_estimation'],
            'cirq': ['phase_estimation'],
            'qsharp': ['QuantumPhaseEstimation'],
            'openqasm': ['// QPE']
        },
        'vqe': {
            'qiskit': ['VQE', 'VariationalQuantumEigensolver'],
            'cirq': [],
            'qsharp': [],
            'openqasm': []
        },
        'qaoa': {
            'qiskit': ['QAOA', 'QuantumApproximateOptimization'],
            'cirq': ['qaoa'],
            'qsharp': [],
            'openqasm': []
        }
    }

class CanonicalAlgorithmCollector:
    """
    Collects canonical quantum algorithm implementations from trusted sources
    """
    
    def __init__(self, output_dir: str = "datasets/canonical_algorithms"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.implementations: List[CanonicalImplementation] = []
        self.seen_hashes: Set[str] = set()
        
        # Rate limiting
        self.request_delay = 1.0  # seconds between requests
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _compute_hash(self, code: str) -> str:
        """Compute hash for deduplication"""
        # Normalize code before hashing
        normalized = ''.join(code.split())  # Remove all whitespace
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def _is_duplicate(self, code: str) -> bool:
        """Check if code is duplicate"""
        code_hash = self._compute_hash(code)
        if code_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(code_hash)
        return False
    
    def collect_from_github_file(
        self, 
        repo_url: str, 
        file_path: str,
        algorithm: str,
        language: str
    ) -> Optional[CanonicalImplementation]:
        """
        Collect implementation from a specific GitHub file
        
        Example:
            collector.collect_from_github_file(
                'https://github.com/Qiskit/qiskit-tutorials',
                'tutorials/algorithms/grover.py',
                'grover',
                'qiskit'
            )
        """
        self._rate_limit()
        
        # Convert to raw GitHub URL
        if 'github.com' in repo_url:
            raw_url = repo_url.replace('github.com', 'raw.githubusercontent.com')
            raw_url = raw_url.replace('/blob/', '/')
            full_url = urljoin(raw_url, file_path)
        else:
            full_url = urljoin(repo_url, file_path)
        
        try:
            response = requests.get(full_url, timeout=10)
            response.raise_for_status()
            code = response.text
            
            # Check for duplicates
            if self._is_duplicate(code):
                print(f"  ⚠️  Duplicate skipped: {file_path}")
                return None
            
            implementation = CanonicalImplementation(
                algorithm=algorithm,
                language=language,
                code=code,
                source_url=full_url,
                source_type='official_docs',
                verified=False,  # Will be validated separately
                hash=self._compute_hash(code),
                qubit_count=self._extract_qubit_count(code),
                metadata={
                    'file_path': file_path,
                    'repo': repo_url
                }
            )
            
            self.implementations.append(implementation)
            return implementation
            
        except Exception as e:
            print(f"  ❌ Error collecting {file_path}: {e}")
            return None
    
    def _extract_qubit_count(self, code: str) -> int:
        """Extract qubit count from code (heuristic)"""
        import re
        
        # Qiskit patterns
        qiskit_patterns = [
            r'QuantumCircuit\((\d+)',
            r'QuantumRegister\((\d+)',
            r'qubits\s*=\s*(\d+)'
        ]
        
        # Cirq patterns
        cirq_patterns = [
            r'LineQubit\.range\((\d+)',
            r'GridQubit\((\d+)',
            r'qubits\s*=.*range\((\d+)'
        ]
        
        # Q# patterns
        qsharp_patterns = [
            r'Qubit\[(\d+)\]',
            r'using\s*\(.*=\s*Qubit\[(\d+)\]'
        ]
        
        # OpenQASM patterns
        qasm_patterns = [
            r'qreg\s+\w+\[(\d+)\]'
        ]
        
        all_patterns = qiskit_patterns + cirq_patterns + qsharp_patterns + qasm_patterns
        
        for pattern in all_patterns:
            match = re.search(pattern, code)
            if match:
                return int(match.group(1))
        
        return 0  # Unknown
    
    def collect_qiskit_textbook_algorithms(self):
        """
        Collect from Qiskit Textbook
        Note: In practice, you'd use GitHub API or web scraping
        This is a template showing the structure
        """
        print("\n📚 Collecting from Qiskit Textbook...")
        
        # These are known canonical examples from Qiskit
        # In practice, you'd automate this discovery
        known_examples = {
            'grover': [
                'https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/grover.ipynb'
            ],
            'deutsch_jozsa': [
                'https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/deutsch-jozsa.ipynb'
            ],
            'bernstein_vazirani': [
                'https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/bernstein-vazirani.ipynb'
            ],
            'simon': [
                'https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/simon.ipynb'
            ],
            'shor': [
                'https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/shor.ipynb'
            ],
            'qft': [
                'https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/quantum-fourier-transform.ipynb'
            ],
            'qpe': [
                'https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/quantum-phase-estimation.ipynb'
            ]
        }
        
        # This is a placeholder - in practice you'd extract code from notebooks
        print("  ℹ️  Manual collection required for Jupyter notebooks")
        print("  ℹ️  Use nbconvert to extract code cells")
    
    def collect_microsoft_qsharp_samples(self):
        """Collect from Microsoft Q# samples"""
        print("\n🔷 Collecting from Microsoft Q# Samples...")
        
        # Known Q# sample paths
        base_repo = 'https://github.com/microsoft/Quantum'
        
        known_samples = {
            'deutsch_jozsa': 'samples/algorithms/deutsch-jozsa/DeutschJozsa.qs',
            'grover': 'samples/algorithms/database-search/DatabaseSearch.qs',
            'bernstein_vazirani': 'samples/algorithms/bernstein-vazirani/BernsteinVazirani.qs',
            'simon': 'samples/algorithms/simon/Simon.qs',
            'qft': 'samples/algorithms/qft/QFT.qs'
        }
        
        for algorithm, file_path in known_samples.items():
            print(f"  Collecting {algorithm}...")
            self.collect_from_github_file(base_repo, file_path, algorithm, 'qsharp')
    
    def save_canonical_dataset(self):
        """Save collected canonical implementations"""
        
        # Create directory structure
        for impl in self.implementations:
            algo_dir = self.output_dir / impl.algorithm / impl.language
            algo_dir.mkdir(parents=True, exist_ok=True)
            
            # Save code
            filename = f"canonical_{impl.algorithm}_{impl.language}_{impl.hash[:8]}.txt"
            code_file = algo_dir / filename
            
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(impl.code)
        
        # Save metadata
        metadata = {
            'total_implementations': len(self.implementations),
            'algorithms': {},
            'languages': {},
            'implementations': [asdict(impl) for impl in self.implementations]
        }
        
        # Count by algorithm
        for impl in self.implementations:
            metadata['algorithms'][impl.algorithm] = \
                metadata['algorithms'].get(impl.algorithm, 0) + 1
            metadata['languages'][impl.language] = \
                metadata['languages'].get(impl.language, 0) + 1
        
        metadata_file = self.output_dir / 'canonical_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Saved {len(self.implementations)} canonical implementations")
        print(f"📁 Location: {self.output_dir}")
    
    def generate_collection_report(self):
        """Generate report of collected algorithms"""
        print("\n" + "=" * 80)
        print("CANONICAL ALGORITHM COLLECTION REPORT")
        print("=" * 80)
        
        print(f"\nTotal implementations: {len(self.implementations)}")
        
        print("\nBy Algorithm:")
        algo_counts = {}
        for impl in self.implementations:
            algo_counts[impl.algorithm] = algo_counts.get(impl.algorithm, 0) + 1
        
        for algo, count in sorted(algo_counts.items()):
            print(f"  {algo}: {count}")
        
        print("\nBy Language:")
        lang_counts = {}
        for impl in self.implementations:
            lang_counts[impl.language] = lang_counts.get(impl.language, 0) + 1
        
        for lang, count in sorted(lang_counts.items()):
            print(f"  {lang}: {count}")
        
        print("\nBy Source Type:")
        source_counts = {}
        for impl in self.implementations:
            source_counts[impl.source_type] = \
                source_counts.get(impl.source_type, 0) + 1
        
        for source, count in sorted(source_counts.items()):
            print(f"  {source}: {count}")
        
        print("=" * 80)


# ============================================================================
# MANUAL ADDITION HELPERS
# ============================================================================

class ManualAlgorithmAdder:
    """Helper for manually adding canonical implementations"""
    
    def __init__(self, collector: CanonicalAlgorithmCollector):
        self.collector = collector
    
    def add_from_file(
        self,
        file_path: str,
        algorithm: str,
        language: str,
        source_url: str = "manual",
        source_type: str = "manual"
    ):
        """Add algorithm from local file"""
        
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        if self.collector._is_duplicate(code):
            print(f"  ⚠️  Duplicate: {file_path}")
            return
        
        implementation = CanonicalImplementation(
            algorithm=algorithm,
            language=language,
            code=code,
            source_url=source_url,
            source_type=source_type,
            verified=False,
            hash=self.collector._compute_hash(code),
            qubit_count=self.collector._extract_qubit_count(code),
            metadata={'file_path': file_path}
        )
        
        self.collector.implementations.append(implementation)
        print(f"  ✅ Added: {algorithm} ({language})")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    collector = CanonicalAlgorithmCollector()
    
    print("=" * 80)
    print("CANONICAL QUANTUM ALGORITHM COLLECTOR")
    print("=" * 80)
    print("\nℹ️  This collector searches trusted sources for canonical implementations")
    print("ℹ️  Rate limited to respect source servers")
    print()
    
    # Collect from various sources
    # Note: Many of these require manual curation due to Jupyter notebooks
    collector.collect_qiskit_textbook_algorithms()
    collector.collect_microsoft_qsharp_samples()
    
    # Manual addition example
    print("\n📝 Manual Addition")
    print("=" * 80)
    print("For Jupyter notebooks and complex sources, use:")
    print("  adder = ManualAlgorithmAdder(collector)")
    print("  adder.add_from_file('path/to/grover.py', 'grover', 'qiskit')")
    print()
    
    # Save dataset
    if collector.implementations:
        collector.save_canonical_dataset()
        collector.generate_collection_report()
    else:
        print("\n⚠️  No implementations collected")
        print("   This is expected - most canonical sources require manual extraction")
        print("   See instructions above for manual addition")
    
    print("\n✅ Collection complete!")
    print("=" * 80)