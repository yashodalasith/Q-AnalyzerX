"""
ML Model Integration Module
Use trained Random Forest + Gradient Boosting models in main API

Add this to modules/ml_algorithm_classifier.py
"""
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List
from models.unified_ast import UnifiedAST
from models.analysis_result import ProblemType

class MLAlgorithmClassifier:
    """
    ML-based algorithm classifier using trained Random Forest + Gradient Boosting
    
    Fallback hierarchy:
    1. ML models (Random Forest + Gradient Boosting ensemble) - Highest accuracy
    2. Pattern matching (algorithm_detector.py) - Good accuracy
    3. Heuristics (keyword search) - Lowest accuracy
    """
    
    def __init__(self, models_dir: str = "models/trained"):
        self.models_dir = Path(models_dir)
        
        self.random_forest = None
        self.gradient_boosting = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = []
        
        self.loaded = False
        
        # Try to load models
        try:
            self.load_models()
        except Exception as e:
            print(f"⚠️  ML models not loaded: {e}")
            print("   Will use pattern matching fallback")
    
    def load_models(self):
        """Load trained models and artifacts"""
        
        self.random_forest = joblib.load(self.models_dir / 'random_forest.pkl')
        self.gradient_boosting = joblib.load(self.models_dir / 'gradient_boosting.pkl')
        self.scaler = joblib.load(self.models_dir / 'scaler.pkl')
        self.label_encoder = joblib.load(self.models_dir / 'label_encoder.pkl')
        
        with open(self.models_dir / 'feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        
        self.loaded = True
        print("✅ ML models loaded successfully!")
    
    def extract_features_from_ast(self, unified_ast: UnifiedAST, quantum_metrics) -> np.ndarray:
        """
        Extract feature vector from unified AST and quantum metrics
        
        Must match the features used during training!
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
        features.append(quantum_metrics.quantum_volume or 0)
        
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
    
    def classify(
        self, 
        unified_ast: UnifiedAST, 
        quantum_metrics,
        use_ensemble: bool = True
    ) -> Dict:
        """
        Classify quantum algorithm using ML models
        
        Returns:
            {
                'algorithm': str,
                'problem_type': ProblemType,
                'confidence': float,
                'method': 'ml' | 'fallback'
            }
        """
        
        if not self.loaded:
            return {
                'algorithm': 'unknown',
                'problem_type': ProblemType.UNKNOWN,
                'confidence': 0.0,
                'method': 'none'
            }
        
        try:
            # Extract features
            features = self.extract_features_from_ast(unified_ast, quantum_metrics)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            if use_ensemble:
                # Ensemble prediction (average probabilities)
                proba_rf = self.random_forest.predict_proba(features_scaled)[0]
                proba_gb = self.gradient_boosting.predict_proba(features_scaled)[0]
                
                proba_ensemble = (proba_rf + proba_gb) / 2
                algorithm = self.label_encoder.classes_[np.argmax(proba_ensemble)]
                confidence = float(np.max(proba_ensemble))
                
            else:
                # Use Random Forest only
                pred = self.random_forest.predict(features_scaled)[0]
                algorithm = self.label_encoder.inverse_transform([pred])[0]
                confidence = float(np.max(self.random_forest.predict_proba(features_scaled)[0]))
            
            # Map algorithm to problem type
            problem_type = self._map_algorithm_to_problem_type(algorithm)
            
            return {
                'algorithm': algorithm,
                'problem_type': problem_type,
                'confidence': confidence,
                'method': 'ml'
            }
            
        except Exception as e:
            print(f"ML classification error: {e}")
            return {
                'algorithm': 'unknown',
                'problem_type': ProblemType.UNKNOWN,
                'confidence': 0.0,
                'method': 'error'
            }
    
    def _map_algorithm_to_problem_type(self, algorithm: str) -> ProblemType:
        """Map detected algorithm to problem type"""
        
        mapping = {
            # Search
            'grover': ProblemType.SEARCH,
            'amplitude_amplification': ProblemType.SEARCH,

            # Oracle / structure learning
            'bernstein_vazirani': ProblemType.ORACLE_IDENTIFICATION,
            'deutsch_jozsa': ProblemType.PROPERTY_TESTING,
            'simon': ProblemType.HIDDEN_STRUCTURE,

            # Factorization / number theory
            'shor': ProblemType.FACTORIZATION,

            # Optimization
            'vqe': ProblemType.OPTIMIZATION,
            'qaoa': ProblemType.OPTIMIZATION,

            # Simulation
            'qft': ProblemType.SIMULATION,
            'qpe': ProblemType.SIMULATION,

            # Sampling
            'boson_sampling': ProblemType.SAMPLING,
        }
        
        return mapping.get(algorithm, ProblemType.UNKNOWN)