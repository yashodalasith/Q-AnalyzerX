"""
Multi-Label Quantum Algorithm Classifier
Detects MULTIPLE algorithms in a single quantum circuit
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set, Tuple
import joblib
import json

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    hamming_loss, 
    accuracy_score, 
    f1_score,
    classification_report,
    multilabel_confusion_matrix
)

import sys
sys.path.append('..')

from models.unified_ast import UnifiedAST
from models.analysis_result import ProblemType

class MultiLabelAlgorithmClassifier:
    """
    Multi-label classifier for detecting multiple algorithms in quantum code
    
    Key differences from single-label:
    - Output is a SET of algorithms, not just one
    - Uses MultiOutputClassifier or one-vs-rest approach
    - Training data includes combinations of algorithms
    """
    
    def __init__(self, models_dir: str = "models/trained_multilabel"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Models
        self.random_forest = None
        self.gradient_boosting = None
        self.xgboost = None
        self.scaler = StandardScaler()
        self.mlb = MultiLabelBinarizer()
        
        # Feature names
        self.feature_names = []
        
        # Algorithm list (will be set during training)
        self.algorithms = []
        
        self.loaded = False
    
    def extract_features_from_ast(self, unified_ast: UnifiedAST, quantum_metrics) -> np.ndarray:
        """
        Extract feature vector from unified AST and quantum metrics
        SAME AS SINGLE-LABEL for consistency
        """
        features = []
        
        # Basic counts
        features.append(quantum_metrics.qubits_required)
        features.append(quantum_metrics.gate_count)
        features.append(quantum_metrics.single_qubit_gates)
        features.append(quantum_metrics.two_qubit_gates)
        features.append(quantum_metrics.cx_gate_count)
        
        # Circuit characteristics
        features.append(quantum_metrics.circuit_depth)
        features.append(quantum_metrics.cx_gate_ratio)
        features.append(quantum_metrics.superposition_score)
        features.append(quantum_metrics.entanglement_score)
        features.append(quantum_metrics.logical_circuit_volume or 0)
        
        # Boolean features
        features.append(int(quantum_metrics.has_superposition))
        features.append(int(quantum_metrics.has_entanglement))
        features.append(int(quantum_metrics.measurement_count > 0))
        
        # Gate type distribution
        from models.unified_ast import GateType
        
        for gate_type in [GateType.H, GateType.X, GateType.Y, GateType.Z,
                         GateType.S, GateType.T, GateType.RX, GateType.RY, GateType.RZ,
                         GateType.CNOT, GateType.CX, GateType.CZ, GateType.SWAP,
                         GateType.TOFFOLI]:
            count = sum(1 for g in unified_ast.gates if g.gate_type == gate_type)
            features.append(count)
        
        # Gate diversity
        gate_types = unified_ast.get_gate_types()
        features.append(len(gate_types))
        
        # Rotation gates
        rotation_gates = sum(1 for g in unified_ast.gates 
                           if g.gate_type in {GateType.RX, GateType.RY, GateType.RZ})
        features.append(rotation_gates)
        
        # Controlled gates
        controlled_gates = sum(1 for g in unified_ast.gates if g.is_controlled)
        features.append(controlled_gates)
        
        # Average gates per qubit
        avg_gates_per_qubit = quantum_metrics.gate_count / max(quantum_metrics.qubits_required, 1)
        features.append(avg_gates_per_qubit)
        
        # Measurement count
        features.append(quantum_metrics.measurement_count)
        
        # Ratio metrics
        single_qubit_ratio = quantum_metrics.single_qubit_gates / max(quantum_metrics.gate_count, 1)
        two_qubit_ratio = quantum_metrics.two_qubit_gates / max(quantum_metrics.gate_count, 1)
        features.append(single_qubit_ratio)
        features.append(two_qubit_ratio)
        
        # Depth-to-gate ratio
        depth_to_gate_ratio = quantum_metrics.circuit_depth / max(quantum_metrics.gate_count, 1)
        features.append(depth_to_gate_ratio)
        
        return np.array(features, dtype=float)
    
    def prepare_multilabel_dataset(self, dataset_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare multi-label dataset
        
        Expected format:
        - Each sample can have MULTIPLE algorithm labels
        - Labels stored as lists: ['grover', 'qft'] or ['bernstein_vazirani']
        """
        print("=" * 80)
        print("PREPARING MULTI-LABEL DATASET")
        print("=" * 80)
        
        # Load dataset metadata
        with open(Path(dataset_path) / 'dataset_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        X_list = []
        y_list = []
        
        # For this example, we'll use the validator to determine all present algorithms
        from datasets.algorithm_validators import AlgorithmValidatorFactory
        
        for i, sample in enumerate(metadata['samples']):
            if i % 50 == 0:
                print(f"Processing: {i}/{len(metadata['samples'])}")
            
            code = sample['code']
            
            # Detect ALL algorithms present
            validation_results = AlgorithmValidatorFactory.validate_all_algorithms(code)
            
            # Get algorithms with high confidence
            detected_algorithms = [
                algo for algo, result in validation_results.items()
                if result.confidence >= 0.7  # Threshold for detection
            ]
            
            # If no algorithms detected with high confidence, use original label
            if not detected_algorithms:
                detected_algorithms = [sample['metadata']['algorithm']]
            
            # Extract features
            from modules.language_detector import LanguageDetector, SupportedLanguage
            from modules.ast_builder import ASTBuilder
            from modules.quantum_analyzer import QuantumAnalyzer
            
            try:
                lang_detector = LanguageDetector()
                ast_builder = ASTBuilder()
                quantum_analyzer = QuantumAnalyzer()
                
                detected_lang = SupportedLanguage(sample['metadata']['language'])
                unified_ast = ast_builder.build(code, detected_lang)
                quantum_metrics = quantum_analyzer.analyze(unified_ast)
                
                features = self.extract_features_from_ast(unified_ast, quantum_metrics)
                
                X_list.append(features)
                y_list.append(detected_algorithms)
                
            except Exception as e:
                print(f"  Error processing sample {i}: {e}")
                continue
        
        X = np.array(X_list)
        
        # Fit MultiLabelBinarizer
        self.mlb.fit(y_list)
        y = self.mlb.transform(y_list)
        
        self.algorithms = list(self.mlb.classes_)
        
        # Define feature names
        self.feature_names = [
            'qubits', 'total_gates', 'single_qubit_gates', 'two_qubit_gates', 'cx_gates',
            'circuit_depth', 'cx_ratio', 'superposition_score', 'entanglement_score', 
            'logical_circuit_volume', 'has_superposition', 'has_entanglement', 'has_measurement',
            'h_gates', 'x_gates', 'y_gates', 'z_gates', 's_gates', 't_gates',
            'rx_gates', 'ry_gates', 'rz_gates', 'cnot_gates', 'cx_gates_2', 
            'cz_gates', 'swap_gates', 'toffoli_gates',
            'gate_diversity', 'rotation_gates', 'controlled_gates',
            'avg_gates_per_qubit', 'measurement_count',
            'single_qubit_ratio', 'two_qubit_ratio', 'depth_to_gate_ratio'
        ]
        
        print(f"\n✅ Multi-label dataset prepared!")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Algorithms: {len(self.algorithms)}")
        print(f"   Algorithm list: {self.algorithms}")
        
        # Print label statistics
        label_counts = y.sum(axis=0)
        print(f"\n📊 Label Distribution:")
        for i, algo in enumerate(self.algorithms):
            print(f"   {algo}: {int(label_counts[i])} samples")
        
        # Multi-label statistics
        samples_per_label = y.sum(axis=1)
        print(f"\n📊 Labels per Sample:")
        print(f"   Mean: {samples_per_label.mean():.2f}")
        print(f"   Max: {int(samples_per_label.max())}")
        print(f"   Samples with 1 label: {(samples_per_label == 1).sum()}")
        print(f"   Samples with 2+ labels: {(samples_per_label > 1).sum()}")
        
        print("=" * 80)
        
        return pd.DataFrame(X, columns=self.feature_names), y
    
    def train_models(self, X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2):
        """
        Train multi-label models
        """
        print("\n" + "=" * 80)
        print("TRAINING MULTI-LABEL MODELS")
        print("=" * 80)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store for evaluation
        self.X_test_scaled = X_test_scaled
        self.y_test = y_test
        
        # ===== RANDOM FOREST (Multi-Output) =====
        print("🌲 Training Multi-Label Random Forest...")
        base_rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.random_forest = MultiOutputClassifier(base_rf, n_jobs=-1)
        self.random_forest.fit(X_train_scaled, y_train)
        print("✅ Random Forest trained!")
        print()
        
        # ===== GRADIENT BOOSTING (Multi-Output) =====
        print("🚀 Training Multi-Label Gradient Boosting...")
        base_gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.gradient_boosting = MultiOutputClassifier(base_gb, n_jobs=-1)
        self.gradient_boosting.fit(X_train_scaled, y_train)
        print("✅ Gradient Boosting trained!")
        print()
        
        # ===== XGBOOST (Multi-Output) =====
        print("🔥 Training Multi-Label XGBoost...")
        base_xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        self.xgboost = MultiOutputClassifier(base_xgb, n_jobs=-1)
        self.xgboost.fit(X_train_scaled, y_train)
        print("✅ XGBoost trained!")
        
        print("=" * 80)
    
    def evaluate_models(self):
        """Evaluate multi-label models"""
        print("\n" + "=" * 80)
        print("MULTI-LABEL MODEL EVALUATION")
        print("=" * 80)
        
        # Predictions
        y_pred_rf = self.random_forest.predict(self.X_test_scaled)
        y_pred_gb = self.gradient_boosting.predict(self.X_test_scaled)
        y_pred_xgb = self.xgboost.predict(self.X_test_scaled)
        
        # Ensemble prediction (majority vote)
        y_pred_ensemble = ((y_pred_rf + y_pred_gb + y_pred_xgb) >= 2).astype(int)
        
        # Metrics
        metrics = {}
        
        for name, y_pred in [
            ('Random Forest', y_pred_rf),
            ('Gradient Boosting', y_pred_gb),
            ('XGBoost', y_pred_xgb),
            ('Ensemble', y_pred_ensemble)
        ]:
            print(f"\n{name}")
            print("-" * 80)
            
            # Hamming loss (fraction of wrong labels)
            h_loss = hamming_loss(self.y_test, y_pred)
            print(f"Hamming Loss: {h_loss:.4f}")
            
            # Exact match ratio (all labels must be correct)
            exact_match = accuracy_score(self.y_test, y_pred)
            print(f"Exact Match Ratio: {exact_match:.4f} ({exact_match*100:.2f}%)")
            
            # F1 scores
            f1_micro = f1_score(self.y_test, y_pred, average='micro')
            f1_macro = f1_score(self.y_test, y_pred, average='macro')
            f1_weighted = f1_score(self.y_test, y_pred, average='weighted')
            
            print(f"F1 (Micro): {f1_micro:.4f}")
            print(f"F1 (Macro): {f1_macro:.4f}")
            print(f"F1 (Weighted): {f1_weighted:.4f}")
            
            metrics[name] = {
                'hamming_loss': h_loss,
                'exact_match': exact_match,
                'f1_micro': f1_micro,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted
            }
        
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print(f"{'Model':<25} {'Exact Match':<15} {'F1 (Micro)':<15}")
        print("-" * 80)
        for name, m in metrics.items():
            print(f"{name:<25} {m['exact_match']:<15.4f} {m['f1_micro']:<15.4f}")
        
        print("=" * 80)
        
        return metrics
    
    def classify(
        self, 
        unified_ast: UnifiedAST, 
        quantum_metrics,
        threshold: float = 0.5
    ) -> Dict:
        """
        Classify quantum code - returns MULTIPLE algorithms
        
        Args:
            threshold: Probability threshold for considering an algorithm present
        
        Returns:
            {
                'algorithms': List[str],  # All detected algorithms
                'confidences': Dict[str, float],  # Confidence per algorithm
                'primary_algorithm': str,  # Most confident
                'problem_type': ProblemType,
                'method': 'ml_multilabel'
            }
        """
        if not self.loaded and not self.random_forest:
            return {
                'algorithms': [],
                'confidences': {},
                'primary_algorithm': 'unknown',
                'problem_type': ProblemType.UNKNOWN,
                'method': 'none'
            }
        
        try:
            # Extract features
            features = self.extract_features_from_ast(unified_ast, quantum_metrics)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get predictions from all models
            # Note: MultiOutputClassifier doesn't have predict_proba directly
            # We need to access individual estimators
            
            # Simplified: Use hard predictions and ensemble
            y_pred_rf = self.random_forest.predict(features_scaled)[0]
            y_pred_gb = self.gradient_boosting.predict(features_scaled)[0]
            y_pred_xgb = self.xgboost.predict(features_scaled)[0]
            
            # Ensemble (majority vote)
            y_pred_ensemble = ((y_pred_rf + y_pred_gb + y_pred_xgb) >= 2).astype(int)
            
            # Get detected algorithms
            detected_indices = np.where(y_pred_ensemble == 1)[0]
            detected_algorithms = [self.algorithms[i] for i in detected_indices]
            
            # Compute confidences (simplified - based on model agreement)
            confidences = {}
            for i, algo in enumerate(self.algorithms):
                vote_sum = y_pred_rf[i] + y_pred_gb[i] + y_pred_xgb[i]
                confidences[algo] = vote_sum / 3.0
            
            # Primary algorithm (highest confidence)
            if detected_algorithms:
                primary = max(detected_algorithms, key=lambda a: confidences[a])
            else:
                primary = 'unknown'
            
            # Determine problem type based on ALL detected algorithms
            problem_types = [
                self._map_algorithm_to_problem_type(algo)
                for algo in detected_algorithms
            ]
            
            # Use most specific problem type
            if ProblemType.FACTORIZATION in problem_types:
                problem_type = ProblemType.FACTORIZATION
            elif ProblemType.SEARCH in problem_types:
                problem_type = ProblemType.SEARCH
            elif ProblemType.OPTIMIZATION in problem_types:
                problem_type = ProblemType.OPTIMIZATION
            elif problem_types:
                problem_type = problem_types[0]
            else:
                problem_type = ProblemType.UNKNOWN
            
            return {
                'algorithms': detected_algorithms,
                'confidences': {a: confidences[a] for a in detected_algorithms},
                'primary_algorithm': primary,
                'problem_type': problem_type,
                'method': 'ml_multilabel',
                'all_confidences': confidences  # For debugging
            }
            
        except Exception as e:
            print(f"ML classification error: {e}")
            return {
                'algorithms': [],
                'confidences': {},
                'primary_algorithm': 'unknown',
                'problem_type': ProblemType.UNKNOWN,
                'method': 'error'
            }
    
    def _map_algorithm_to_problem_type(self, algorithm: str) -> ProblemType:
        """Map algorithm to problem type"""
        mapping = {
            'grover': ProblemType.SEARCH,
            'amplitude_amplification': ProblemType.SEARCH,
            'bernstein_vazirani': ProblemType.ORACLE_IDENTIFICATION,
            'deutsch_jozsa': ProblemType.PROPERTY_TESTING,
            'simon': ProblemType.HIDDEN_STRUCTURE,
            'shor': ProblemType.FACTORIZATION,
            'vqe': ProblemType.OPTIMIZATION,
            'qaoa': ProblemType.OPTIMIZATION,
            'qft': ProblemType.SIMULATION,
            'qpe': ProblemType.SIMULATION
        }
        return mapping.get(algorithm, ProblemType.UNKNOWN)
    
    def save_models(self):
        """Save trained models"""
        print("\n💾 Saving multi-label models...")
        
        joblib.dump(self.random_forest, self.models_dir / 'multilabel_rf.pkl')
        joblib.dump(self.gradient_boosting, self.models_dir / 'multilabel_gb.pkl')
        joblib.dump(self.xgboost, self.models_dir / 'multilabel_xgb.pkl')
        joblib.dump(self.scaler, self.models_dir / 'multilabel_scaler.pkl')
        joblib.dump(self.mlb, self.models_dir / 'multilabel_binarizer.pkl')
        
        with open(self.models_dir / 'multilabel_feature_names.json', 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        with open(self.models_dir / 'multilabel_algorithms.json', 'w') as f:
            json.dump(self.algorithms, f, indent=2)
        
        print(f"✅ Models saved to: {self.models_dir}")
    
    def load_models(self):
        """Load trained models"""
        self.random_forest = joblib.load(self.models_dir / 'multilabel_rf.pkl')
        self.gradient_boosting = joblib.load(self.models_dir / 'multilabel_gb.pkl')
        self.xgboost = joblib.load(self.models_dir / 'multilabel_xgb.pkl')
        self.scaler = joblib.load(self.models_dir / 'multilabel_scaler.pkl')
        self.mlb = joblib.load(self.models_dir / 'multilabel_binarizer.pkl')
        
        with open(self.models_dir / 'multilabel_feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        
        with open(self.models_dir / 'multilabel_algorithms.json', 'r') as f:
            self.algorithms = json.load(f)
        
        self.loaded = True
        print("✅ Multi-label models loaded successfully!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-LABEL QUANTUM ALGORITHM CLASSIFIER")
    print("=" * 80)
    print("\n⚠️  This is a template. You need to:")
    print("   1. Generate/collect multi-label dataset")
    print("   2. Run training pipeline")
    print("   3. Integrate with main API")
    print("\nKey difference: Detects MULTIPLE algorithms per circuit!")
    print("=" * 80)