"""
Machine Learning Training Pipeline
Feature Extraction → Random Forest & Gradient Boosting → Evaluation
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Import your analysis modules
import sys
sys.path.append('..')
from modules.language_detector import LanguageDetector, SupportedLanguage
from modules.ast_builder import ASTBuilder
from modules.quantum_analyzer import QuantumAnalyzer
from modules.complexity_analyzer import ComplexityAnalyzer

class QuantumAlgorithmMLPipeline:
    """
    Complete ML pipeline for quantum algorithm classification
    
    Steps:
    1. Feature extraction from unified AST
    2. Dataset preparation
    3. Model training (Random Forest + Gradient Boosting)
    4. Model evaluation
    5. Model persistence
    """
    
    def __init__(self, dataset_dir: str = "datasets/quantum_algorithms"):
        self.dataset_dir = Path(dataset_dir)
        self.models_dir = Path("models/trained")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzers
        self.language_detector = LanguageDetector()
        self.ast_builder = ASTBuilder()
        self.quantum_analyzer = QuantumAnalyzer()
        self.complexity_analyzer = ComplexityAnalyzer()
        
        # Models
        self.random_forest = None
        self.gradient_boosting = None
        self.xgboost = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = []
    
    def extract_features_from_code(self, code: str, language: str) -> np.ndarray:
        """
        Extract feature vector from quantum code using unified AST
        
        Features extracted:
        - Qubit count
        - Gate count (total, single-qubit, two-qubit)
        - Circuit depth (accurate)
        - Gate type distribution (H, X, Y, Z, CNOT, CZ, etc.)
        - CX gate ratio
        - Superposition score (simulated)
        - Entanglement score (simulated)
        - Quantum volume
        - Has measurements
        - Gate diversity (unique gate types)
        - Parameterized gate count
        - Rotation gate count
        - Controlled gate count
        - Circuit width (qubits used)
        - Average gates per qubit
        """
        
        try:
            # Parse code and build AST
            detected_lang = SupportedLanguage(language)
            parser = self.ast_builder.parsers[detected_lang]
            parsed_data = parser.parse(code)
            unified_ast = self.ast_builder.build(code, detected_lang)
            
            # Get quantum metrics (now uses accurate analyzers)
            quantum_metrics = self.quantum_analyzer.analyze(unified_ast)
            
            # Extract features
            features = []
            
            # Basic counts
            features.append(quantum_metrics.qubits_required)
            features.append(quantum_metrics.gate_count)
            features.append(quantum_metrics.single_qubit_gates)
            features.append(quantum_metrics.two_qubit_gates)
            features.append(quantum_metrics.cx_gate_count)
            
            # Circuit characteristics
            features.append(quantum_metrics.circuit_depth)  # ACCURATE
            features.append(quantum_metrics.cx_gate_ratio)
            features.append(quantum_metrics.superposition_score)  # ACCURATE (simulated)
            features.append(quantum_metrics.entanglement_score)  # ACCURATE (simulated)
            features.append(quantum_metrics.logical_circuit_volume or 0)
            
            # Boolean features (converted to int)
            features.append(int(quantum_metrics.has_superposition))
            features.append(int(quantum_metrics.has_entanglement))
            features.append(int(quantum_metrics.measurement_count > 0))
            
            # Gate type distribution
            gate_types = unified_ast.get_gate_types()
            from models.unified_ast import GateType
            
            for gate_type in [GateType.H, GateType.X, GateType.Y, GateType.Z,
                             GateType.S, GateType.T, GateType.RX, GateType.RY, GateType.RZ,
                             GateType.CNOT, GateType.CX, GateType.CZ, GateType.SWAP,
                             GateType.TOFFOLI]:
                count = sum(1 for g in unified_ast.gates if g.gate_type == gate_type)
                features.append(count)
            
            # Gate diversity
            features.append(len(gate_types))
            
            # Rotation gates (parameterized)
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
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return zero vector on error
            return np.zeros(35, dtype=float)  # Adjust size based on feature count
    
    def prepare_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load dataset and extract features for all samples
        
        Returns:
            X: Feature matrix (samples × features)
            y: Labels (algorithm names)
        """
        
        print("=" * 80)
        print("PREPARING DATASET")
        print("=" * 80)
        
        # Load metadata
        metadata_file = self.dataset_dir / "dataset_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        samples = metadata['samples']
        print(f"Total samples: {len(samples)}")
        
        X_list = []
        y_list = []
        
        for i, sample in enumerate(samples):
            if i % 100 == 0:
                print(f"Processing: {i}/{len(samples)}")
            
            code = sample['code']
            language = sample['metadata']['language']
            algorithm = sample['metadata']['algorithm']
            
            # Extract features
            features = self.extract_features_from_code(code, language)
            
            X_list.append(features)
            y_list.append(algorithm)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Define feature names
        self.feature_names = [
            'qubits', 'total_gates', 'single_qubit_gates', 'two_qubit_gates', 'cx_gates',
            'circuit_depth', 'cx_ratio', 'superposition_score', 'entanglement_score', 'logical_circuit_volume',
            'has_superposition', 'has_entanglement', 'has_measurement',
            'h_gates', 'x_gates', 'y_gates', 'z_gates', 's_gates', 't_gates',
            'rx_gates', 'ry_gates', 'rz_gates', 'cnot_gates', 'cx_gates_2', 'cz_gates', 'swap_gates', 'toffoli_gates',
            'gate_diversity', 'rotation_gates', 'controlled_gates',
            'avg_gates_per_qubit', 'measurement_count',
            'single_qubit_ratio', 'two_qubit_ratio', 'depth_to_gate_ratio'
        ]
        
        print(f"\n✅ Feature extraction complete!")
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Algorithms: {len(np.unique(y))}")
        print("=" * 80)
        
        return pd.DataFrame(X, columns=self.feature_names), pd.Series(y, name='algorithm')
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """
        Train Random Forest, Gradient Boosting, AND XGBoost
        """
        
        print("\n" + "=" * 80)
        print("TRAINING MODELS")
        print("=" * 80)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print()
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # ===== RANDOM FOREST =====
        print("🌲 Training Random Forest...")
        self.random_forest = RandomForestClassifier(
            n_estimators=300,
            max_depth=25,              # Different from others
            min_samples_split=2,       # More aggressive splits
            min_samples_leaf=1,        # Allow pure leaves
            max_features='sqrt',       # Feature subsampling
            bootstrap=True,            # Bootstrap sampling
            random_state=100,          # DIFFERENT random state
            n_jobs=-1,
            verbose=0,
            class_weight='balanced'
        )
        
        self.random_forest.fit(self.X_train_scaled, self.y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.random_forest, self.X_train_scaled, self.y_train, 
            cv=5, scoring='accuracy'
        )
        
        print(f"✅ Random Forest trained!")
        print(f"   Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print()
        
        # ===== GRADIENT BOOSTING =====
        print("🚀 Training Gradient Boosting...")
        self.gradient_boosting = GradientBoostingClassifier(
            n_estimators=200,          # Fewer trees (different strategy)
            learning_rate=0.05,        # Slower learning (more conservative)
            max_depth=6,               # Different depth
            min_samples_split=10,      # More conservative splits
            min_samples_leaf=4,        # Larger leaves
            subsample=0.7,             # Different subsampling
            max_features='log2',       # Different feature selection
            random_state=200,          # DIFFERENT random state
            verbose=0
        )
        
        self.gradient_boosting.fit(self.X_train_scaled, self.y_train)
        
        # Cross-validation
        cv_scores_gb = cross_val_score(
            self.gradient_boosting, self.X_train_scaled, self.y_train,
            cv=5, scoring='accuracy'
        )
        
        print(f"✅ Gradient Boosting trained!")
        print(f"   Cross-validation accuracy: {cv_scores_gb.mean():.4f} (+/- {cv_scores_gb.std():.4f})")
        print("=" * 80)

        # ===== XGBOOST =====
        print("🚀 Training XGBoost...")
        self.xgboost = XGBClassifier(
            n_estimators=250,          # Different number
            max_depth=8,               # Different depth
            learning_rate=0.1,         # Different learning rate
            subsample=0.85,            # Different subsampling
            colsample_bytree=0.85,     # Column subsampling
            colsample_bylevel=0.7,     # Level-wise column subsampling
            gamma=0.5,                 # Min split loss
            min_child_weight=3,        # Different min child weight
            reg_alpha=0.05,            # L1 regularization
            reg_lambda=2.0,            # L2 regularization
            random_state=300,          # DIFFERENT random state
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0,
            n_jobs=-1
        )
        
        self.xgboost.fit(self.X_train_scaled, self.y_train)
        
        # Cross-validation
        cv_scores_xgb = cross_val_score(
            self.xgboost, self.X_train_scaled, self.y_train,
            cv=5, scoring='accuracy'
        )
        
        print(f"✅ XGBoost trained!")
        print(f"   CV accuracy: {cv_scores_xgb.mean():.4f} (+/- {cv_scores_xgb.std():.4f})")
        print("=" * 80)
    
    def evaluate_models(self):
        """
        Evaluate all THREE models (RF, GB, XGB) + ensemble
        """
        
        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)
        
        # Predictions
        y_pred_rf = self.random_forest.predict(self.X_test_scaled)
        y_pred_gb = self.gradient_boosting.predict(self.X_test_scaled)
        y_pred_xgb = self.xgboost.predict(self.X_test_scaled) 
        y_train_pred_rf = self.random_forest.predict(self.X_train_scaled)
        y_train_pred_gb = self.gradient_boosting.predict(self.X_train_scaled)
        y_train_pred_xgb = self.xgboost.predict(self.X_train_scaled)

        # Probabilities
        y_proba_rf = self.random_forest.predict_proba(self.X_test_scaled)
        y_proba_gb = self.gradient_boosting.predict_proba(self.X_test_scaled)
        y_proba_xgb = self.xgboost.predict_proba(self.X_test_scaled)
        y_train_proba_rf = self.random_forest.predict_proba(self.X_train_scaled)
        y_train_proba_gb = self.gradient_boosting.predict_proba(self.X_train_scaled)
        y_train_proba_xgb = self.xgboost.predict_proba(self.X_train_scaled)
        
        proba_ensemble = (y_proba_rf + y_proba_gb + y_proba_xgb) / 3
        y_pred_ensemble = np.argmax(proba_ensemble, axis=1)
        avg_train_proba = (y_train_proba_rf + y_train_proba_gb + y_train_proba_xgb) / 3
        y_train_pred_ensemble = np.argmax(avg_train_proba, axis=1)
        
        # Decode labels
        y_test_labels = self.label_encoder.inverse_transform(self.y_test)
        y_pred_rf_labels = self.label_encoder.inverse_transform(y_pred_rf)
        y_pred_gb_labels = self.label_encoder.inverse_transform(y_pred_gb)
        y_pred_xgb_labels = self.label_encoder.inverse_transform(y_pred_xgb)
        
        # ===== RANDOM FOREST EVALUATION =====
        print("\n🌲 RANDOM FOREST")
        print("-" * 80)
        
        accuracy_rf = accuracy_score(self.y_test, y_pred_rf)
        train_accuracy_rf = accuracy_score(self.y_train, y_train_pred_rf)
        print("\n📈 TRAINING ACCURACY")
        print(f"Random Forest Train Accuracy: {train_accuracy_rf:.4f}")
        print(f"Accuracy: {accuracy_rf:.4f} ({accuracy_rf * 100:.2f}%)")
        
        precision_rf, recall_rf, f1_rf, _ = precision_recall_fscore_support(
            self.y_test, y_pred_rf, average='weighted'
        )
        print(f"Precision: {precision_rf:.4f}")
        print(f"Recall: {recall_rf:.4f}")
        print(f"F1 Score: {f1_rf:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test_labels, y_pred_rf_labels))
        rf_report = classification_report(
            y_test_labels, y_pred_rf_labels, output_dict=True
        )

        with open(self.models_dir / 'classification_report_rf.json', 'w') as f:
            json.dump(rf_report, f, indent=2)

        # ===== GRADIENT BOOSTING EVALUATION =====
        print("\n🚀 GRADIENT BOOSTING")
        print("-" * 80)
        
        accuracy_gb = accuracy_score(self.y_test, y_pred_gb)
        train_accuracy_gb = accuracy_score(self.y_train, y_train_pred_gb)
        print("\n📈 TRAINING ACCURACY")
        print(f"Gradient Boosting Train Accuracy: {train_accuracy_gb:.4f}")
        print(f"Accuracy: {accuracy_gb:.4f} ({accuracy_gb * 100:.2f}%)")
        
        precision_gb, recall_gb, f1_gb, _ = precision_recall_fscore_support(
            self.y_test, y_pred_gb, average='weighted'
        )
        print(f"Precision: {precision_gb:.4f}")
        print(f"Recall: {recall_gb:.4f}")
        print(f"F1 Score: {f1_gb:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test_labels, y_pred_gb_labels))

        gb_report = classification_report(
            y_test_labels, y_pred_gb_labels, output_dict=True
        )

        with open(self.models_dir / 'classification_report_gb.json', 'w') as f:
            json.dump(gb_report, f, indent=2)

        # ===== XGBOOST EVALUATION =====
        print("\n🔥 XGBOOST")
        print("-" * 80)
        accuracy_xgb = accuracy_score(self.y_test, y_pred_xgb)
        train_accuracy_xgb = accuracy_score(self.y_train, y_train_pred_xgb)
        print("\n📈 TRAINING ACCURACY")
        print(f"XGBoost Train Accuracy: {train_accuracy_xgb:.4f}")
        print(f"Accuracy: {accuracy_xgb:.4f} ({accuracy_xgb * 100:.2f}%)")

        precision_xgb, recall_xgb, f1_xgb, _ = precision_recall_fscore_support(
            self.y_test, y_pred_xgb, average='weighted'
        )
        print(f"Precision: {precision_xgb:.4f}")
        print(f"Recall: {recall_xgb:.4f}")
        print(f"F1 Score: {f1_xgb:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test_labels, y_pred_xgb_labels))

        xgb_report = classification_report(
            y_test_labels, y_pred_xgb_labels, output_dict=True
        )

        with open(self.models_dir / 'classification_report_xgb.json', 'w') as f:
            json.dump(xgb_report, f, indent=2)

        # ===== 4. ENSEMBLE (RF + GB + XGB) =====
        print("\n⚖️  ENSEMBLE (RF + GB + XGB)")
        print("-" * 80)
        
        # Weighted ensemble (optimized weights)
        # RF is best at avoiding overfitting
        # GB is best at complex patterns
        # XGB is best at boosting weak learners
        y_proba_ensemble = (
            0.35 * y_proba_rf +      # Random Forest weight
            0.30 * y_proba_gb +      # Gradient Boosting weight
            0.35 * y_proba_xgb       # XGBoost weight
        )
        
        y_pred_ensemble = np.argmax(y_proba_ensemble, axis=1)
        y_pred_ensemble_labels = self.label_encoder.inverse_transform(y_pred_ensemble)
        
        accuracy_ensemble = accuracy_score(self.y_test, y_pred_ensemble)
        print(f"Accuracy: {accuracy_ensemble:.4f} ({accuracy_ensemble * 100:.2f}%)")
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred_ensemble, average='weighted'
        )
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("\nEnsemble Classification Report:")
        print(classification_report(y_test_labels, y_pred_ensemble_labels))

        ens_report = classification_report(y_test_labels, y_pred_ensemble_labels, output_dict=True)
        with open(self.models_dir / 'classification_report_ens.json', 'w') as f:
            json.dump(ens_report, f, indent=2)
        
        # ===== COMPARISON TABLE =====
        print("\n" + "=" * 80)
        print("ACCURACY COMPARISON")
        print("=" * 80)
        print(f"{'Model':<25} {'Accuracy':<15} {'Improvement':<15}")
        print("-" * 80)
        print(f"{'Random Forest':<25} {accuracy_rf:<15.4f} {'-':<15}")
        print(f"{'Gradient Boosting':<25} {accuracy_gb:<15.4f} {'-':<15}")
        print(f"{'XGBoost':<25} {accuracy_xgb:<15.4f} {'-':<15}")
        
        avg_individual = (accuracy_rf + accuracy_gb + accuracy_xgb) / 3
        improvement = accuracy_ensemble - avg_individual
        
        print(f"{'Ensemble (RF+GB+XGB)':<25} {accuracy_ensemble:<15.4f} {f'+{improvement:.4f}':<15}")
        print("=" * 80)

        # ===== FEATURE IMPORTANCE =====
        print("\n📊 FEATURE IMPORTANCE (Top 10)")
        print("-" * 80)
        
        importances = self.random_forest.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        for i, idx in enumerate(indices):
            print(f"{i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")
        
        # ===== CONFUSION MATRICES =====
        self._plot_confusion_matrices(self.y_test, y_pred_rf, y_pred_gb, y_pred_ensemble, y_pred_xgb)
        self._plot_train_confusion_matrices(
            self.y_train, y_train_pred_rf, y_train_pred_gb, y_train_pred_ensemble, y_train_pred_xgb
        )

        # ===== LEARNING CURVES =====
        self._plot_learning_curve(self.random_forest, 'random_forest')
        self._plot_learning_curve(self.gradient_boosting, 'gradient_boosting')
        self._plot_learning_curve(self.xgboost, 'xgboost')

        # ===== CONFIDENCE CALIBRATION =====
        self._plot_confidence_calibration(
            self.y_test,
            y_pred_rf,
            y_proba_rf,
            model_name='Random Forest',
            filename='calibration_rf.png'
        )

        self._plot_confidence_calibration(
            self.y_test,
            y_pred_gb,
            y_proba_gb,
            model_name='Gradient Boosting',
            filename='calibration_gb.png'
        )

        self._plot_confidence_calibration(
            self.y_test,
            y_pred_xgb,
            y_proba_xgb,
            model_name='XGBoost',
            filename='calibration_xgb.png'
        )

        self._plot_confidence_calibration(
            self.y_test,
            y_pred_ensemble,
            proba_ensemble,
            model_name='Ensemble',
            filename='calibration_ensemble.png'
        )

        # ===== DATA SLICE EVALUATION =====
        gate_counts = self.X_test['total_gates'].values
        slice_metrics = {}

        for name, mask in {
            'low_gates': gate_counts < 50,
            'medium_gates': (gate_counts >= 50) & (gate_counts < 200),
            'high_gates': gate_counts >= 200
        }.items():
            if mask.any():
                slice_metrics[name] = {
                    'accuracy': accuracy_score(self.y_test[mask], y_pred_rf[mask]),
                    'samples': int(mask.sum())
                }

        with open(self.models_dir / 'data_slice_metrics.json', 'w') as f:
            json.dump(slice_metrics, f, indent=2)

        print("=" * 80)
        
        return {
            'random_forest': {
                'accuracy': accuracy_rf,
                'precision': precision_rf,
                'recall': recall_rf,
                'f1': f1_rf
            },
            'gradient_boosting': {
                'accuracy': accuracy_gb,
                'precision': precision_gb,
                'recall': recall_gb,
                'f1': f1_gb
            },
            'xgboost': {                    
                'accuracy': accuracy_xgb,
                'precision': precision_xgb,
                'recall': recall_xgb,
                'f1': f1_xgb
            },
            'ensemble': {                    
                'accuracy': accuracy_ensemble,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        }
    
    def _plot_confusion_matrices(self, y_true, y_pred_rf, y_pred_gb, y_pred_ensemble, y_pred_xgb):
        """Plot confusion matrices for Random Forest, Gradient Boosting, Ensemble, and XGBoost models"""
        
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))  # 4 subplots now, one for each model
        
        # Random Forest
        cm_rf = confusion_matrix(y_true, y_pred_rf)
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=self.label_encoder.classes_,
                yticklabels=self.label_encoder.classes_)
        axes[0].set_title('Random Forest - Confusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # Gradient Boosting
        cm_gb = confusion_matrix(y_true, y_pred_gb)
        sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=self.label_encoder.classes_,
                yticklabels=self.label_encoder.classes_)
        axes[1].set_title('Gradient Boosting - Confusion Matrix')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')

        # Ensemble Model
        cm_ensemble = confusion_matrix(y_true, y_pred_ensemble)
        sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Purples', ax=axes[2],
                xticklabels=self.label_encoder.classes_,
                yticklabels=self.label_encoder.classes_)
        axes[2].set_title('Ensemble Model - Confusion Matrix')
        axes[2].set_ylabel('True Label')
        axes[2].set_xlabel('Predicted Label')

        # XGBoost
        cm_xgb = confusion_matrix(y_true, y_pred_xgb)
        sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', ax=axes[3],
                xticklabels=self.label_encoder.classes_,
                yticklabels=self.label_encoder.classes_)
        axes[3].set_title('XGBoost - Confusion Matrix')
        axes[3].set_ylabel('True Label')
        axes[3].set_xlabel('Predicted Label')

        plt.tight_layout()
        plt.savefig(self.models_dir / 'confusion_matrices_all_models.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Confusion matrices saved to: {self.models_dir / 'confusion_matrices_all_models.png'}")

    def _plot_train_confusion_matrices(
        self,
        y_train_true,
        y_train_pred_rf,
        y_train_pred_gb,
        y_train_pred_ensemble,
        y_train_pred_xgb
    ):
        """Plot TRAIN confusion matrices for all models"""

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        # Random Forest
        cm_rf = confusion_matrix(y_train_true, y_train_pred_rf)
        sns.heatmap(
            cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        axes[0].set_title('Random Forest - TRAIN')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')

        # Gradient Boosting
        cm_gb = confusion_matrix(y_train_true, y_train_pred_gb)
        sns.heatmap(
            cm_gb, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        axes[1].set_title('Gradient Boosting - TRAIN')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')

        # Ensemble
        cm_ensemble = confusion_matrix(y_train_true, y_train_pred_ensemble)
        sns.heatmap(
            cm_ensemble, annot=True, fmt='d', cmap='Purples', ax=axes[2],
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        axes[2].set_title('Ensemble - TRAIN')
        axes[2].set_ylabel('True Label')
        axes[2].set_xlabel('Predicted Label')

        # XGBoost
        cm_xgb = confusion_matrix(y_train_true, y_train_pred_xgb)
        sns.heatmap(
            cm_xgb, annot=True, fmt='d', cmap='Oranges', ax=axes[3],
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        axes[3].set_title('XGBoost - TRAIN')
        axes[3].set_ylabel('True Label')
        axes[3].set_xlabel('Predicted Label')

        plt.tight_layout()
        plt.savefig(
            self.models_dir / 'train_confusion_matrices_all_models.png',
            dpi=300,
            bbox_inches='tight'
        )

        print(
            f"\n📊 Train confusion matrices saved to: "
            f"{self.models_dir / 'train_confusion_matrices_all_models.png'}"
        )
    
    def _plot_confidence_calibration(self, y_true, y_pred, y_proba, model_name, filename):
        """Plot confidence calibration curve for a model"""

        confidences = np.max(y_proba, axis=1)
        correct = (y_pred == y_true).astype(int)

        bins = np.linspace(0, 1, 11)
        bin_acc = []

        for i in range(len(bins) - 1):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if mask.any():
                bin_acc.append(correct[mask].mean())
            else:
                bin_acc.append(np.nan)

        plt.figure(figsize=(8, 6))
        plt.plot(bins[:-1], bin_acc, marker='o', label=model_name)
        plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
        plt.xlabel('Predicted Confidence')
        plt.ylabel('Empirical Accuracy')
        plt.title(f'Confidence Calibration ({model_name})')
        plt.legend()
        plt.grid(True)

        plt.savefig(self.models_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_learning_curve(self, model, model_name):
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            self.X_train_scaled,
            self.y_train,
            cv=5,
            scoring='accuracy',
            train_sizes=np.linspace(0.1, 1.0, 5),
            n_jobs=-1
        )

        train_mean = train_scores.mean(axis=1)
        val_mean = val_scores.mean(axis=1)

        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_mean, 'o-', label='Training Accuracy')
        plt.plot(train_sizes, val_mean, 'o-', label='Validation Accuracy')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.title(f'Learning Curve - {model_name}')
        plt.legend()
        plt.grid(True)

        plt.savefig(self.models_dir / f'learning_curve_{model_name}.png', dpi=300)
        plt.close()

    def save_models(self):
        """Save trained models and artifacts"""
        
        print("\n💾 Saving models...")
        
        # Save models
        joblib.dump(self.random_forest, self.models_dir / 'random_forest.pkl')
        joblib.dump(self.gradient_boosting, self.models_dir / 'gradient_boosting.pkl')
        joblib.dump(self.xgboost, self.models_dir / 'xgboost.pkl')
        joblib.dump(self.scaler, self.models_dir / 'scaler.pkl')
        joblib.dump(self.label_encoder, self.models_dir / 'label_encoder.pkl')
        
        # Save feature names
        with open(self.models_dir / 'feature_names.json', 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        print(f"✅ Models saved to: {self.models_dir}")
    
    def load_models(self):
        """Load trained models"""
        
        self.random_forest = joblib.load(self.models_dir / 'random_forest.pkl')
        self.gradient_boosting = joblib.load(self.models_dir / 'gradient_boosting.pkl')
        self.xgboost = joblib.load(self.models_dir / 'xgboost.pkl')
        self.scaler = joblib.load(self.models_dir / 'scaler.pkl')
        self.label_encoder = joblib.load(self.models_dir / 'label_encoder.pkl')
        
        with open(self.models_dir / 'feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        
        print("✅ Models loaded successfully!")
    
    def predict(self, code: str, language: str, use_ensemble: bool = True):
        """
        Predict algorithm for new code
        
        Args:
            code: Source code
            language: Programming language
            use_ensemble: If True, average predictions from all models
        """
        
        # Extract features
        features = self.extract_features_from_code(code, language)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        if use_ensemble:
            # Ensemble prediction (average probabilities)
            proba_rf = self.random_forest.predict_proba(features_scaled)[0]
            proba_gb = self.gradient_boosting.predict_proba(features_scaled)[0]
            proba_xgb = self.xgboost.predict_proba(features_scaled)[0]
            
            # Weighted ensemble
            proba_ensemble = (
                0.35 * proba_rf +
                0.30 * proba_gb +
                0.35 * proba_xgb
            )
            prediction = self.label_encoder.classes_[np.argmax(proba_ensemble)]
            confidence = np.max(proba_ensemble)
            
        else:
             # Use XGBoost only (best single model)
            prediction = self.xgboost.predict(features_scaled)[0]
            prediction = self.label_encoder.inverse_transform([prediction])[0]
            confidence = np.max(self.xgboost.predict_proba(features_scaled)[0])
        
        return {
            'algorithm': prediction,
            'confidence': float(confidence)
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    pipeline = QuantumAlgorithmMLPipeline()
    
    # Step 1: Prepare dataset
    X, y = pipeline.prepare_dataset()
    
    # Step 2: Train models
    pipeline.train_models(X, y)
    
    # Step 3: Evaluate
    metrics = pipeline.evaluate_models()
    
    # Step 4: Save models
    pipeline.save_models()
    
    print("\n" + "=" * 80)
    print("🎉 ML PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"Random Forest Accuracy: {metrics['random_forest']['accuracy']:.2%}")
    print(f"Gradient Boosting Accuracy: {metrics['gradient_boosting']['accuracy']:.2%}")
    print(f"XGBoost Accuracy: {metrics['xgboost']['accuracy']:.2%}")
    print(f"Ensemble Accuracy: {metrics['ensemble']['accuracy']:.2%}")
    print("=" * 80)