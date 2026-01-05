"""
ML Model Integration Module
Use trained Random Forest + Gradient Boosting + XG Boost models in main API

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
        self.xgboost = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = []
        self.ensemble_weights = {
            'random_forest': 0.35,
            'gradient_boosting': 0.30,
            'xgboost': 0.35
        }
        
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
        self.xgboost = joblib.load('xgboost.pkl')
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
            
            # Get predictions from all 3 models
            proba_rf = self.random_forest.predict_proba(features_scaled)[0]
            proba_gb = self.gradient_boosting.predict_proba(features_scaled)[0]
            proba_xgb = self.xgboost.predict_proba(features_scaled)[0] 
            
            # Individual predictions (for debugging/transparency)
            pred_rf = self.label_encoder.classes_[np.argmax(proba_rf)]
            pred_gb = self.label_encoder.classes_[np.argmax(proba_gb)]
            pred_xgb = self.label_encoder.classes_[np.argmax(proba_xgb)] 
            
            if use_ensemble:
                # Weighted ensemble (3 models)
                proba_ensemble = (
                    self.ensemble_weights['random_forest'] * proba_rf +
                    self.ensemble_weights['gradient_boosting'] * proba_gb +
                    self.ensemble_weights['xgboost'] * proba_xgb  
                )
                
                algorithm = self.label_encoder.classes_[np.argmax(proba_ensemble)]
                confidence = float(np.max(proba_ensemble))
                
            else:
                # Use XGBoost only (typically best single model)
                algorithm = pred_xgb
                confidence = float(np.max(proba_xgb))
            
            # Map algorithm to problem type
            problem_type = self._map_algorithm_to_problem_type(algorithm)
            
            return {
                'algorithm': algorithm,
                'problem_type': problem_type,
                'confidence': confidence,
                'method': 'ml',
                'model_predictions': {  
                    'random_forest': pred_rf,
                    'gradient_boosting': pred_gb,
                    'xgboost': pred_xgb,
                    'ensemble': algorithm
                }
            }
            
        except Exception as e:
            print(f"ML classification error: {e}")
            return {
                'algorithm': 'unknown',
                'problem_type': ProblemType.UNKNOWN,
                'confidence': 0.0,
                'method': 'error'
            }
    
    def classify_with_details(
        self, 
        unified_ast: UnifiedAST, 
        quantum_metrics
    ) -> Dict:
        """
        Classify with detailed probability breakdown from all models
        
        Useful for debugging and understanding model decisions
        """
        
        if not self.loaded:
            return {'error': 'Models not loaded'}
        
        try:
            features = self.extract_features_from_ast(unified_ast, quantum_metrics)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get probabilities from all models
            proba_rf = self.random_forest.predict_proba(features_scaled)[0]
            proba_gb = self.gradient_boosting.predict_proba(features_scaled)[0]
            proba_xgb = self.xgboost.predict_proba(features_scaled)[0]
            
            # Ensemble
            proba_ensemble = (
                self.ensemble_weights['random_forest'] * proba_rf +
                self.ensemble_weights['gradient_boosting'] * proba_gb +
                self.ensemble_weights['xgboost'] * proba_xgb
            )
            
            # Build detailed response
            classes = self.label_encoder.classes_
            
            details = {
                'random_forest': {
                    cls: float(proba_rf[i]) 
                    for i, cls in enumerate(classes)
                },
                'gradient_boosting': {
                    cls: float(proba_gb[i]) 
                    for i, cls in enumerate(classes)
                },
                'xgboost': {
                    cls: float(proba_xgb[i]) 
                    for i, cls in enumerate(classes)
                },
                'ensemble': {
                    cls: float(proba_ensemble[i]) 
                    for i, cls in enumerate(classes)
                },
                'final_prediction': classes[np.argmax(proba_ensemble)],
                'confidence': float(np.max(proba_ensemble)),
                'ensemble_weights': self.ensemble_weights
            }
            
            return details
            
        except Exception as e:
            return {'error': str(e)}
        
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        
        if not self.loaded:
            return {'status': 'not_loaded'}
        
        return {
            'status': 'loaded',
            'models': ['random_forest', 'gradient_boosting', 'xgboost'],
            'ensemble_weights': self.ensemble_weights,
            'feature_count': len(self.feature_names),
            'algorithms': list(self.label_encoder.classes_),
            'models_dir': str(self.models_dir)
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
    
# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Test the updated classifier with XGBoost
    """
    
    from models.unified_ast import UnifiedAST, QuantumRegisterNode, QuantumGateNode, GateType
    from models.analysis_result import QuantumComplexity
    
    # Initialize classifier
    classifier = MLAlgorithmClassifier()
    
    if not classifier.loaded:
        print("⚠️  Models not loaded. Run training first:")
        print("   python run_complete_pipeline.py")
        exit()
    
    # Print model info
    info = classifier.get_model_info()
    print("=" * 80)
    print("ML ALGORITHM CLASSIFIER INFO")
    print("=" * 80)
    print(f"Status: {info['status']}")
    print(f"Models: {', '.join(info['models'])}")
    print(f"Ensemble weights:")
    for model, weight in info['ensemble_weights'].items():
        print(f"   {model}: {weight:.2f}")
    print(f"Features: {info['feature_count']}")
    print(f"Algorithms: {', '.join(info['algorithms'])}")
    print("=" * 80)
    
    # Create sample Grover circuit
    grover_ast = UnifiedAST(
        source_language='qiskit',
        quantum_registers=[QuantumRegisterNode(name='q', size=3, line_number=1)],
        gates=[
            QuantumGateNode(gate_type=GateType.H, qubits=[0], line_number=2),
            QuantumGateNode(gate_type=GateType.H, qubits=[1], line_number=3),
            QuantumGateNode(gate_type=GateType.H, qubits=[2], line_number=4),
            QuantumGateNode(gate_type=GateType.CZ, qubits=[1], control_qubits=[0], 
                          is_controlled=True, line_number=5),
            QuantumGateNode(gate_type=GateType.H, qubits=[0], line_number=6),
            QuantumGateNode(gate_type=GateType.X, qubits=[0], line_number=7),
            QuantumGateNode(gate_type=GateType.H, qubits=[1], line_number=8),
            QuantumGateNode(gate_type=GateType.X, qubits=[1], line_number=9),
            QuantumGateNode(gate_type=GateType.CCX, qubits=[2], control_qubits=[0, 1],
                          is_controlled=True, line_number=10),
        ],
        total_qubits=3,
        total_gates=9
    )
    
    # Mock quantum metrics
    mock_metrics = QuantumComplexity(
        qubits_required=3,
        circuit_depth=8,
        gate_count=9,
        single_qubit_gates=6,
        two_qubit_gates=3,
        cx_gate_count=1,
        cx_gate_ratio=0.11,
        measurement_count=3,
        superposition_score=0.85,
        entanglement_score=0.72,
        has_superposition=True,
        has_entanglement=True,
        quantum_volume=64
    )
    
    # Test classification
    print("\n" + "=" * 80)
    print("TESTING 3-MODEL ENSEMBLE")
    print("=" * 80)
    
    result = classifier.classify(grover_ast, mock_metrics, use_ensemble=True)
    
    print(f"\n🎯 Final Prediction: {result['algorithm']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Problem Type: {result['problem_type']}")
    print(f"   Method: {result['method']}")
    
    print(f"\n📊 Individual Model Predictions:")
    print(f"   Random Forest: {result['model_predictions']['random_forest']}")
    print(f"   Gradient Boosting: {result['model_predictions']['gradient_boosting']}")
    print(f"   XGBoost: {result['model_predictions']['xgboost']}")
    
    # Check if models differ (good for ensemble!)
    preds = set([
        result['model_predictions']['random_forest'],
        result['model_predictions']['gradient_boosting'],
        result['model_predictions']['xgboost']
    ])
    
    if len(preds) > 1:
        print(f"\n✅ Models give DIFFERENT predictions ({len(preds)} unique)")
        print("   This means ensemble will be more robust!")
    else:
        print(f"\n⚠️  All models agree on: {preds.pop()}")
    
    # Get detailed probabilities
    print("\n" + "=" * 80)
    print("DETAILED PROBABILITY BREAKDOWN")
    print("=" * 80)
    
    details = classifier.classify_with_details(grover_ast, mock_metrics)
    
    if 'error' not in details:
        print("\nTop 3 predictions per model:")
        
        for model in ['random_forest', 'gradient_boosting', 'xgboost', 'ensemble']:
            probs = details[model]
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            
            print(f"\n{model.replace('_', ' ').title()}:")
            for algo, prob in sorted_probs:
                print(f"   {algo}: {prob:.4f} ({prob*100:.2f}%)")
    
    print("\n" + "=" * 80)