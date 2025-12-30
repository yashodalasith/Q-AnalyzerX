"""
Automated GitHub Dataset Generator
Scrapes real quantum code from GitHub for language classification

Target: 20,000+ samples across 5 languages
- Qiskit: 4,000
- Cirq: 4,000  
- Q#: 3,000
- OpenQASM: 3,000
- Python (non-quantum): 6,000

Usage:
    python github_dataset_generator.py --token YOUR_GITHUB_TOKEN
"""

import os
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import base64
import random
from dotenv import load_dotenv
load_dotenv()

class GitHubQuantumDatasetGenerator:
    """
    Automated dataset generator using GitHub API
    Scrapes real quantum code repositories
    """
    
    def __init__(self, github_token: Optional[str] = None, output_dir: str = "datasets/language_classification"):
        self.token = github_token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GitHub token not found. Set GITHUB_TOKEN in .env file.")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        self.base_url = 'https://api.github.com'
        
        # Rate limiting
        self.requests_count = 0
        self.max_requests_per_hour = 4500  # GitHub allows 5000/hour
        
        # Dataset
        self.dataset = []
        
        # Deduplication + checkpointing
        self.seen_files = set()
        self.checkpoint_file = self.output_dir / "checkpoint.json"

        # Resume from checkpoint if exists
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r") as f:
                state = json.load(f)
                self.dataset = state.get("dataset", [])
                self.seen_files = set(state.get("seen_files", []))
                print(f"🔁 Resumed from checkpoint ({len(self.dataset)} samples)")

        # Search queries for each language
        self.search_queries = {
            'qiskit': [
                'from qiskit import language:Python',
                'QuantumCircuit language:Python',
                'qiskit.circuit language:Python',
                'qiskit.providers language:Python'
            ],
            'cirq': [
                'import cirq language:Python',
                'cirq.Circuit language:Python',
                'cirq.LineQubit language:Python',
                'cirq.Simulator language:Python'
            ],
            'qsharp': [
                'extension:qs',
                'namespace Microsoft.Quantum',
                'operation language:Q#'
            ],
            'openqasm': [
                'extension:qasm',
                'OPENQASM 2.0',
                'OPENQASM 3.0',
                'qreg creg'
            ],
            'python': [
                'def main language:Python -qiskit -cirq -quantum',
                'import numpy language:Python -qiskit -cirq',
                'class language:Python -quantum'
            ]
        }
    
    def generate_dataset(self, samples_per_language: Dict[str, int]):
        """
        Generate complete dataset by scraping GitHub
        
        Args:
            samples_per_language: Target samples per language
                {'qiskit': 4000, 'cirq': 4000, ...}
        """
        
        print("=" * 80)
        print("GITHUB QUANTUM CODE DATASET GENERATOR")
        print("=" * 80)
        print(f"Target samples: {sum(samples_per_language.values()):,}")
        print()
        
        for language, target_count in samples_per_language.items():
            print(f"\n📊 Scraping {language} code (target: {target_count})...")
            
            collected = 0
            for query in self.search_queries[language]:
                if collected >= target_count:
                    break
                
                print(f"  Query: {query}")
                
                files = self._search_github_code(query, max_results=target_count - collected)
                
                for file_info in files:
                    if collected >= target_count:
                        break
                    
                    code = self._download_file(file_info)
                    
                    repo = file_info.get('repository', {}).get('full_name', 'unknown')
                    path = file_info.get('path', '')
                    unique_id = f"{repo}:{path}"

                    # Deduplication
                    if unique_id in self.seen_files:
                        continue

                    if code and self._is_valid_sample(code, language):
                        self.dataset.append({
                            'code': code,
                            'label': language,
                            'source': 'github',
                            'repo': repo,
                            'path': path,
                            'size': len(code)
                        })

                        self.seen_files.add(unique_id)
                        collected += 1

                        # Save checkpoint every 50 samples
                        if collected % 50 == 0:
                            self._save_checkpoint()
                            print(f"    💾 Checkpoint saved ({collected}/{target_count})")
                        
                        if collected % 100 == 0:
                            print(f"    Collected: {collected}/{target_count}")
                
                # Respect rate limits
                self._check_rate_limit()
                # Save progress after each query
                self._save_checkpoint()
            
            print(f"  ✅ Collected {collected} {language} samples")
        
        # Save dataset
        self._save_dataset()
        # Cleanup checkpoint after successful completion
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            print("🧹 Checkpoint cleared (dataset generation completed)")
        
        print("\n" + "=" * 80)
        print(f"✅ Dataset generation complete!")
        print(f"   Total samples: {len(self.dataset):,}")
        print(f"   Output: {self.output_dir}")
        print("=" * 80)
    
    def _search_github_code(self, query: str, max_results: int = 1000) -> List[Dict]:
        """Search GitHub code API"""
        
        results = []
        page = 1
        per_page = 100  # Max allowed by GitHub
        
        while len(results) < max_results and page <= 10:  # Max 10 pages
            url = f"{self.base_url}/search/code"
            params = {
                'q': query,
                'per_page': per_page,
                'page': page
            }
            
            try:
                response = requests.get(url, headers=self.headers, params=params)
                self.requests_count += 1
                
                if response.status_code == 200:
                    data = response.json()
                    items = data.get('items', [])
                    
                    if not items:
                        break
                    
                    results.extend(items)
                    page += 1
                    
                    time.sleep(2)  # Be nice to GitHub
                    
                elif response.status_code == 403:
                    reset_time = response.headers.get("X-RateLimit-Reset")
                    if reset_time:
                        wait_seconds = int(reset_time) - int(time.time()) + 5
                        wait_seconds = max(wait_seconds, 10)
                        print(f"⏳ Rate limit hit. Waiting {wait_seconds}s...")
                        time.sleep(wait_seconds)
                    else:
                        print("⏳ Rate limit hit. Waiting 60s...")
                        time.sleep(60)
            
                else:
                    print(f"    Error: {response.status_code}")
                    break
                    
            except Exception as e:
                print(f"    Error searching: {e}")
                break
        
        return results[:max_results]
    
    def _download_file(self, file_info: Dict) -> Optional[str]:
        """Download file contents from GitHub"""
        
        try:
            download_url = file_info.get('url')
            
            if not download_url:
                return None
            
            response = requests.get(download_url, headers=self.headers)
            self.requests_count += 1
            
            if response.status_code == 200:
                data = response.json()
                
                # Decode base64 content
                content = data.get('content', '')
                if content:
                    decoded = base64.b64decode(content).decode('utf-8', errors='ignore')
                    return decoded
            
            time.sleep(1)
            
        except Exception as e:
            print(f"    Error downloading: {e}")
        
        return None
    
    def _is_valid_sample(self, code: str, language: str) -> bool:
        """Validate code sample"""
        
        # Length checks
        if len(code) < 50 or len(code) > 10000:
            return False
        
        # Line count check
        lines = code.split('\n')
        if len(lines) < 5 or len(lines) > 500:
            return False
        
        # Language-specific validation
        code_lower = code.lower()
        
        if language == 'qiskit':
            return 'qiskit' in code_lower or 'quantumcircuit' in code_lower
        
        elif language == 'cirq':
            return 'cirq' in code_lower
        
        elif language == 'qsharp':
            return 'namespace' in code_lower or 'operation' in code_lower
        
        elif language == 'openqasm':
            return 'openqasm' in code_lower or ('qreg' in code_lower and 'creg' in code_lower)
        
        elif language == 'python':
            # Must NOT contain quantum keywords
            quantum_keywords = ['qiskit', 'cirq', 'quantum', 'qubit', 'qreg']
            return not any(kw in code_lower for kw in quantum_keywords)
        
        return True
    
    def _check_rate_limit(self):
        """Check and handle GitHub rate limits safely"""

        if self.requests_count < self.max_requests_per_hour:
            return

        url = f"{self.base_url}/rate_limit"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            data = response.json()
            core = data["resources"]["core"]

            remaining = core["remaining"]
            reset_time = core["reset"]

            if remaining == 0:
                wait_seconds = reset_time - int(time.time()) + 5
                wait_seconds = max(wait_seconds, 10)

                print(f"⏳ Rate limit hit. Waiting {wait_seconds}s until reset...")
                time.sleep(wait_seconds)

        # Reset counter after waiting
        self.requests_count = 0

    def _save_checkpoint(self):
        """Save progress so scraping can resume after crash"""
        with open(self.checkpoint_file, "w") as f:
            json.dump({
                "dataset": self.dataset,
                "seen_files": list(self.seen_files)
            }, f)
    
    def _save_dataset(self):
        """Save dataset to JSONL format"""
        
        # Shuffle dataset
        random.shuffle(self.dataset)
        
        # Save full dataset
        output_file = self.output_dir / 'language_dataset.jsonl'
        with open(output_file, 'w') as f:
            for sample in self.dataset:
                f.write(json.dumps(sample) + '\n')
        
        print(f"\n💾 Dataset saved to: {output_file}")
        
        # Split into train/val/test
        total = len(self.dataset)
        train_size = int(total * 0.8)
        val_size = int(total * 0.1)
        
        train_data = self.dataset[:train_size]
        val_data = self.dataset[train_size:train_size + val_size]
        test_data = self.dataset[train_size + val_size:]
        
        # Save splits
        for name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            split_file = self.output_dir / f'{name}.jsonl'
            with open(split_file, 'w') as f:
                for sample in data:
                    f.write(json.dumps(sample) + '\n')
        
        # Save metadata
        metadata = {
            'total_samples': total,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'languages': {},
            'generated_at': datetime.now().isoformat()
        }
        
        # Count per language
        for sample in self.dataset:
            lang = sample['label']
            metadata['languages'][lang] = metadata['languages'].get(lang, 0) + 1
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   Train: {len(train_data):,} samples")
        print(f"   Val: {len(val_data):,} samples")
        print(f"   Test: {len(test_data):,} samples")


# ============================================================================
# AUGMENTATION & NEGATIVE SAMPLING
# ============================================================================

class DataAugmenter:
    """Augment dataset with synthetic variations"""
    
    @staticmethod
    def augment_code(code: str, num_variations: int = 3) -> List[str]:
        """Generate code variations"""
        
        variations = [code]  # Include original
        
        for _ in range(num_variations):
            augmented = code
            
            # Random transformations
            if random.random() > 0.5:
                augmented = DataAugmenter._add_comments(augmented)
            
            if random.random() > 0.5:
                augmented = DataAugmenter._change_whitespace(augmented)
            
            if random.random() > 0.5:
                augmented = DataAugmenter._rename_variables(augmented)
            
            variations.append(augmented)
        
        return variations
    
    @staticmethod
    def _add_comments(code: str) -> str:
        """Add random comments"""
        comments = [
            "# TODO: Optimize this",
            "# FIXME: Check edge cases",
            "# NOTE: Important section",
            "# Code generated automatically"
        ]
        
        lines = code.split('\n')
        insert_pos = random.randint(0, len(lines))
        lines.insert(insert_pos, random.choice(comments))
        
        return '\n'.join(lines)
    
    @staticmethod
    def _change_whitespace(code: str) -> str:
        """Change whitespace formatting"""
        # Add/remove blank lines
        lines = code.split('\n')
        
        if random.random() > 0.5 and len(lines) > 2:
            # Remove a blank line
            lines = [l for l in lines if l.strip()]
        else:
            # Add a blank line
            insert_pos = random.randint(0, len(lines))
            lines.insert(insert_pos, '')
        
        return '\n'.join(lines)
    
    @staticmethod
    def _rename_variables(code: str) -> str:
        """Simple variable renaming"""
        # Replace common variable names
        replacements = {
            'qc': 'circuit',
            'circuit': 'qc',
            'qreg': 'q',
            'creg': 'c'
        }
        
        augmented = code
        for old, new in replacements.items():
            if old in code:
                augmented = augmented.replace(old, new)
                break
        
        return augmented


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate language classification dataset from GitHub')
    parser.add_argument('--token', help='GitHub personal access token (optional if using .env)')
    parser.add_argument('--output', default='datasets/language_classification', help='Output directory')
    
    args = parser.parse_args()
    
    # Target samples per language
    samples = {
        'qiskit': 4000,
        'cirq': 4000,
        'qsharp': 3000,
        'openqasm': 3000,
        'python': 6000
    }
    
    generator = GitHubQuantumDatasetGenerator(args.token, args.output)
    try:
        generator.generate_dataset(samples)
    except KeyboardInterrupt:
        generator._save_checkpoint()
        print("\n🛑 Interrupted. Progress saved to checkpoint.")
    
    print("\n🎉 Dataset ready for training!")
    print("Next step: python train_language_classifier.py")