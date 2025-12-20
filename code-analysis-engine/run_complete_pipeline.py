"""
Complete Pipeline Script
Runs entire workflow: Dataset Generation → Feature Extraction → Training → Evaluation

Usage:
    python run_complete_pipeline.py

This will:
1. Generate 1000+ quantum algorithm variations (10 algorithms × 4 languages × 100 variations)
2. Extract features using accurate unified AST analysis
3. Train Random Forest and Gradient Boosting models
4. Evaluate and save models
5. Generate accuracy reports and visualizations
"""

import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets.dataset_generator import QuantumAlgorithmDatasetGenerator
from modules.ml_training_pipeline import QuantumAlgorithmMLPipeline

def run_complete_pipeline(
    variations_per_algo: int = 100,
    regenerate_dataset: bool = True
):
    """
    Run complete ML pipeline from scratch
    
    Args:
        variations_per_algo: Number of code variations per algorithm per language
        regenerate_dataset: If False, skip dataset generation and use existing data
    """
    
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "QUANTUM ALGORITHM ML PIPELINE" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    start_time = time.time()
    
    # ========================================================================
    # STEP 1: DATASET GENERATION
    # ========================================================================
    
    if regenerate_dataset:
        print("\n📊 STEP 1: GENERATING DATASET")
        dataset_start = time.time()
        
        generator = QuantumAlgorithmDatasetGenerator()
        generator.generate_all_datasets(variations_per_algo=variations_per_algo)
        
        dataset_time = time.time() - dataset_start
        print(f"\n✅ Dataset generation complete in {dataset_time:.2f}s")
        print(f"📁 Dataset saved to: datasets/quantum_algorithms/")
    else:
        print("\n⏭️  STEP 1: SKIPPING DATASET GENERATION (using existing data)")
    
    # ========================================================================
    # STEP 2: FEATURE EXTRACTION & MODEL TRAINING
    # ========================================================================
    
    print("\n\n🤖 STEP 2: FEATURE EXTRACTION & MODEL TRAINING")
    print("=" * 80)
    
    training_start = time.time()
    
    pipeline = QuantumAlgorithmMLPipeline()
    
    # Extract features
    print("\n[1/3] Extracting features from unified AST...")
    X, y = pipeline.prepare_dataset()
    
    # Train models
    print("\n[2/3] Training Random Forest + Gradient Boosting...")
    pipeline.train_models(X, y, test_size=0.2)
    
    # Evaluate
    print("\n[3/3] Evaluating models...")
    metrics = pipeline.evaluate_models()
    
    training_time = time.time() - training_start
    print(f"\n✅ Training complete in {training_time:.2f}s")
    
    # ========================================================================
    # STEP 3: SAVE MODELS
    # ========================================================================
    
    print("\n\n💾 STEP 3: SAVING MODELS")
    print("=" * 80)
    
    pipeline.save_models()
    
    # ========================================================================
    # FINAL REPORT
    # ========================================================================
    
    total_time = time.time() - start_time
    
    print("\n\n╔" + "=" * 78 + "╗")
    print("║" + " " * 30 + "FINAL REPORT" + " " * 36 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    print("📊 DATASET")
    print(f"   Total samples: {len(y):,}")
    print(f"   Algorithms: {len(y.unique())}")
    print(f"   Features extracted: {X.shape[1]}")
    print()
    
    print("🤖 MODELS")
    print(f"   Random Forest:")
    print(f"      ├─ Accuracy: {metrics['random_forest']['accuracy']:.4f} ({metrics['random_forest']['accuracy']*100:.2f}%)")
    print(f"      ├─ Precision: {metrics['random_forest']['precision']:.4f}")
    print(f"      ├─ Recall: {metrics['random_forest']['recall']:.4f}")
    print(f"      └─ F1 Score: {metrics['random_forest']['f1']:.4f}")
    print()
    print(f"   Gradient Boosting:")
    print(f"      ├─ Accuracy: {metrics['gradient_boosting']['accuracy']:.4f} ({metrics['gradient_boosting']['accuracy']*100:.2f}%)")
    print(f"      ├─ Precision: {metrics['gradient_boosting']['precision']:.4f}")
    print(f"      ├─ Recall: {metrics['gradient_boosting']['recall']:.4f}")
    print(f"      └─ F1 Score: {metrics['gradient_boosting']['f1']:.4f}")
    print()
    
    print("⏱️  PERFORMANCE")
    if regenerate_dataset:
        print(f"   Dataset generation: {dataset_time:.2f}s")
    print(f"   Training: {training_time:.2f}s")
    print(f"   Total time: {total_time:.2f}s")
    print()
    
    print("📁 OUTPUT")
    print(f"   Dataset: datasets/quantum_algorithms/")
    print(f"   Models: models/trained/")
    print(f"   ├─ random_forest.pkl")
    print(f"   ├─ gradient_boosting.pkl")
    print(f"   ├─ scaler.pkl")
    print(f"   ├─ label_encoder.pkl")
    print(f"   └─ confusion_matrices.png")
    print()
    
    # Determine if models are production-ready
    avg_accuracy = (metrics['random_forest']['accuracy'] + 
                   metrics['gradient_boosting']['accuracy']) / 2
    
    if avg_accuracy >= 0.95:
        status = "🎉 EXCELLENT"
        message = "Models are production-ready!"
    elif avg_accuracy >= 0.90:
        status = "✅ GOOD"
        message = "Models are ready for deployment."
    elif avg_accuracy >= 0.85:
        status = "⚠️  FAIR"
        message = "Models may need more training data."
    else:
        status = "❌ POOR"
        message = "Models need significant improvement."
    
    print(f"🎯 MODEL STATUS: {status}")
    print(f"   {message}")
    print(f"   Average accuracy: {avg_accuracy:.2%}")
    print()
    
    print("=" * 80)
    print("✨ PIPELINE COMPLETE! Models ready for integration into main API.")
    print("=" * 80)
    
    return {
        'metrics': metrics,
        'total_time': total_time,
        'status': status
    }


def test_model_inference():
    """
    Test trained models with sample code
    """
    
    print("\n\n🧪 TESTING MODEL INFERENCE")
    print("=" * 80)
    
    pipeline = QuantumAlgorithmMLPipeline()
    pipeline.load_models()
    
    # Test Grover's algorithm
    test_code = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(3, 3)

# Superposition
for i in range(3):
    qc.h(i)

# Oracle
qc.cz(0, 1)

# Diffusion
for i in range(3):
    qc.h(i)
    qc.x(i)

qc.h(2)
qc.ccx(0, 1, 2)
qc.h(2)

for i in range(3):
    qc.x(i)
    qc.h(i)

qc.measure_all()
    """
    
    result = pipeline.predict(test_code, 'qiskit', use_ensemble=True)
    
    print(f"Test Code: Grover's Algorithm (3 qubits)")
    print(f"Prediction: {result['algorithm']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print()
    
    if result['algorithm'] == 'grover' and result['confidence'] > 0.9:
        print("✅ Model correctly identified Grover's algorithm with high confidence!")
    else:
        print("⚠️  Model prediction may need improvement")
    
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run complete ML pipeline')
    parser.add_argument(
        '--variations', 
        type=int, 
        default=100,
        help='Number of variations per algorithm per language (default: 100)'
    )
    parser.add_argument(
        '--skip-dataset', 
        action='store_true',
        help='Skip dataset generation and use existing data'
    )
    parser.add_argument(
        '--test-inference', 
        action='store_true',
        help='Test model inference after training'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    results = run_complete_pipeline(
        variations_per_algo=args.variations,
        regenerate_dataset=not args.skip_dataset
    )
    
    # Optional: Test inference
    if args.test_inference:
        test_model_inference()