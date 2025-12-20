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
            features.append(quantum_metrics.quantum_volume or 0)
            
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
            'circuit_depth', 'cx_ratio', 'superposition_score', 'entanglement_score', 'quantum_volume',
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
        Train Random Forest and Gradient Boosting models
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
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
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
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42,
            verbose=1
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
    
    def evaluate_models(self):
        """
        Evaluate both models and generate detailed reports
        """
        
        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)
        
        # Predictions
        y_pred_rf = self.random_forest.predict(self.X_test_scaled)
        y_pred_gb = self.gradient_boosting.predict(self.X_test_scaled)

        # ===== TRAINING SET EVALUATION =====
        y_train_pred_rf = self.random_forest.predict(self.X_train_scaled)
        y_train_pred_gb = self.gradient_boosting.predict(self.X_train_scaled)
        
        # Decode labels
        y_test_labels = self.label_encoder.inverse_transform(self.y_test)
        y_pred_rf_labels = self.label_encoder.inverse_transform(y_pred_rf)
        y_pred_gb_labels = self.label_encoder.inverse_transform(y_pred_gb)
        
        # ===== RANDOM FOREST EVALUATION =====
        print("\n🌲 RANDOM FOREST")
        print("-" * 80)
        
        accuracy_rf = accuracy_score(self.y_test, y_pred_rf)
        train_accuracy_rf = accuracy_score(self.y_train, y_train_pred_rf)
        print(f"Accuracy: {accuracy_rf:.4f} ({accuracy_rf * 100:.2f}%)")
        print("\n📈 TRAINING ACCURACY")
        print(f"Random Forest Train Accuracy: {train_accuracy_rf:.4f}")
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred_rf, average='weighted'
        )
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
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
        print(f"Accuracy: {accuracy_gb:.4f} ({accuracy_gb * 100:.2f}%)")
        print("\n📈 TRAINING ACCURACY")
        print(f"Gradient Boosting Train Accuracy: {train_accuracy_gb:.4f}")
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred_gb, average='weighted'
        )
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test_labels, y_pred_gb_labels))

        gb_report = classification_report(
            y_test_labels, y_pred_gb_labels, output_dict=True
        )

        with open(self.models_dir / 'classification_report_gb.json', 'w') as f:
            json.dump(gb_report, f, indent=2)

        # ===== FEATURE IMPORTANCE =====
        print("\n📊 FEATURE IMPORTANCE (Top 10)")
        print("-" * 80)
        
        importances = self.random_forest.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        for i, idx in enumerate(indices):
            print(f"{i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")
        
        # ===== CONFUSION MATRICES =====
        self._plot_confusion_matrices(y_test_labels, y_pred_rf_labels, y_pred_gb_labels)
        self._plot_train_confusion_matrices(
            self.y_train, y_train_pred_rf, y_train_pred_gb
        )

        # ===== LEARNING CURVES =====
        self._plot_learning_curve(self.random_forest, 'random_forest')
        self._plot_learning_curve(self.gradient_boosting, 'gradient_boosting')

        # ===== CONFIDENCE CALIBRATION =====
        proba_rf = self.random_forest.predict_proba(self.X_test_scaled)
        confidences = np.max(proba_rf, axis=1)
        correct = (y_pred_rf == self.y_test).astype(int)

        bins = np.linspace(0, 1, 11)
        bin_acc = []

        for i in range(len(bins) - 1):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if mask.any():
                bin_acc.append(correct[mask].mean())
            else:
                bin_acc.append(np.nan)

        plt.figure(figsize=(8, 6))
        plt.plot(bins[:-1], bin_acc, marker='o')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel('Predicted Confidence')
        plt.ylabel('Empirical Accuracy')
        plt.title('Confidence Calibration (Random Forest)')
        plt.grid(True)

        plt.savefig(self.models_dir / 'confidence_calibration.png', dpi=300)
        plt.close()

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
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'gradient_boosting': {
                'accuracy': accuracy_gb,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        }
    
    def _plot_confusion_matrices(self, y_true, y_pred_rf, y_pred_gb):
        """Plot confusion matrices for both models"""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
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
        
        plt.tight_layout()
        plt.savefig(self.models_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Confusion matrices saved to: {self.models_dir / 'confusion_matrices.png'}")

    def _plot_train_confusion_matrices(self, y_train_true, y_train_pred_rf, y_train_pred_gb):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        cm_rf = confusion_matrix(y_train_true, y_train_pred_rf)
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        axes[0].set_title('Random Forest - TRAIN Confusion Matrix')

        cm_gb = confusion_matrix(y_train_true, y_train_pred_gb)
        sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        axes[1].set_title('Gradient Boosting - TRAIN Confusion Matrix')

        plt.tight_layout()
        plt.savefig(self.models_dir / 'train_confusion_matrices.png', dpi=300)

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
            use_ensemble: If True, average predictions from both models
        """
        
        # Extract features
        features = self.extract_features_from_code(code, language)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        if use_ensemble:
            # Ensemble prediction (average probabilities)
            proba_rf = self.random_forest.predict_proba(features_scaled)[0]
            proba_gb = self.gradient_boosting.predict_proba(features_scaled)[0]
            
            proba_ensemble = (proba_rf + proba_gb) / 2
            prediction = self.label_encoder.classes_[np.argmax(proba_ensemble)]
            confidence = np.max(proba_ensemble)
            
        else:
            # Use Random Forest only
            prediction = self.random_forest.predict(features_scaled)[0]
            prediction = self.label_encoder.inverse_transform([prediction])[0]
            confidence = np.max(self.random_forest.predict_proba(features_scaled)[0])
        
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
    print("=" * 80)