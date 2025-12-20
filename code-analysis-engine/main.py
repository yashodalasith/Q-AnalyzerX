"""
Code Analysis Engine - Complete Implementation with Accurate Analysis
Port: 8002
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Import modules
from modules.language_detector import LanguageDetector, SupportedLanguage
from modules.ast_builder import ASTBuilder
from modules.complexity_analyzer import ComplexityAnalyzer
from modules.quantum_analyzer import QuantumAnalyzer
from modules.algorithm_detector import QuantumAlgorithmDetector 
from models.analysis_result import CodeAnalysisResult, ProblemType, TimeComplexity
from modules.ml_algorithm_classifier import MLAlgorithmClassifier

app = FastAPI(
    title="Code Analysis Engine",
    description="Analyzes quantum-classical code for intelligent routing",
    version="1.0.0"
)

# Initialize components
language_detector = LanguageDetector()
ast_builder = ASTBuilder()
complexity_analyzer = ComplexityAnalyzer()
quantum_analyzer = QuantumAnalyzer()
algorithm_detector = QuantumAlgorithmDetector() 
ml_classifier = MLAlgorithmClassifier()

# Request Models
class CodeSubmission(BaseModel):
    code: str
    filename: Optional[str] = None

class LanguageDetectionResponse(BaseModel):
    language: str
    confidence: float
    is_supported: bool
    details: str

# Routes
@app.get("/")
async def root():
    return {
        "service": "Code Analysis Engine",
        "version": "1.0.0",
        "status": "operational",
        "capabilities": [
            "Multi-language parsing",
            "Accurate complexity analysis (AST-based)",  
            "Quantum metrics extraction (state simulation)",  
            "Algorithm detection (pattern matching)",  
            "Circuit depth analysis (dependency graph)"  
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "code-analysis-engine"}

@app.post("/detect-language", response_model=LanguageDetectionResponse)
async def detect_language(submission: CodeSubmission):
    """Detect programming language"""
    try:
        result = language_detector.detect(code=submission.code, filename=submission.filename)
        return LanguageDetectionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=CodeAnalysisResult)
async def analyze_code(submission: CodeSubmission):
    """
    Complete code analysis pipeline with accurate metrics
    Returns metrics for Decision Engine
    """
    try:
        code = submission.code
        
        # Step 1: Detect language
        lang_result = language_detector.detect(code=code, filename=submission.filename)
        
        if not lang_result["is_supported"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported language: {lang_result['language']}"
            )
        
        detected_lang = SupportedLanguage(lang_result["language"])
        
        # Step 2: Parse and build unified AST
        parser = ast_builder.parsers[detected_lang]
        parsed_data = parser.parse(code)
        unified_ast = ast_builder.build(code, detected_lang)
        metadata = ast_builder.get_metadata(parsed_data)
        
        # Step 3: Determine if quantum or classical
        is_quantum = detected_lang in {
            SupportedLanguage.QISKIT, 
            SupportedLanguage.CIRQ,
            SupportedLanguage.OPENQASM,
            SupportedLanguage.QSHARP
        }
        
        classical_metrics = None
        quantum_metrics = None
        problem_type = ProblemType.CLASSICAL
        detected_algorithms = []
        algorithm_confidence = 0.0
        
        if is_quantum:
            # === QUANTUM ANALYSIS ===
            # quantum_analyzer uses:
            # - AccurateCircuitDepthCalculator
            # - QuantumStateSimulator
            quantum_metrics = quantum_analyzer.analyze(unified_ast)
            ml_result = ml_classifier.classify(
                unified_ast, 
                quantum_metrics,
                use_ensemble=True
            )
            
            if ml_result['confidence'] > 0.7:
                detected_algorithms = [ml_result['algorithm']]
                problem_type = ml_result['problem_type']
                algorithm_confidence = ml_result['confidence']
            else:
                # Use accurate algorithm detector 
                algorithm_result = algorithm_detector.detect(unified_ast)
                problem_type = algorithm_result['problem_type']
                detected_algorithms = algorithm_result['detected_algorithms']
                algorithm_confidence = algorithm_result['confidence']
            
            # Fallback to heuristics if algorithm detector has low confidence
            if algorithm_confidence < 0.5:
                problem_type = determine_problem_type_heuristic(code, is_quantum=True)
            
            # analyze classical parts if present (hybrid quantum-classical)
            if metadata['lines_of_code'] > 0 and metadata.get('function_count', 0) > 0:
                # complexity_analyzer uses:
                # - AccurateTimeComplexityAnalyzer
                # - AccurateSpaceComplexityAnalyzer
                classical_metrics = complexity_analyzer.analyze(code, metadata)
        else:
            # === CLASSICAL ANALYSIS ===
            # complexity_analyzer uses accurate methods
            classical_metrics = complexity_analyzer.analyze(code, metadata)
            problem_type = ProblemType.CLASSICAL
        
        # Step 4: Build result for Decision Engine
        result = build_analysis_result(
            detected_lang=detected_lang,
            lang_confidence=lang_result["confidence"],
            problem_type=problem_type,
            classical_metrics=classical_metrics,
            quantum_metrics=quantum_metrics,
            metadata=metadata,
            detected_algorithms=detected_algorithms,  # NEW
            algorithm_confidence=algorithm_confidence  # NEW
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/supported-languages")
async def get_supported_languages():
    """List supported quantum programming languages"""
    return {
        "languages": [
            {"name": "Python", "value": "python"},
            {"name": "Qiskit", "value": "qiskit"},
            {"name": "Q#", "value": "qsharp"},
            {"name": "Cirq", "value": "cirq"},
            {"name": "OpenQASM", "value": "openqasm"}
        ],
        "count": 5
    }

# Helper Functions

def determine_problem_type_heuristic(code: str, is_quantum: bool = False) -> ProblemType:
    """
    Fallback heuristic-based problem type classification
    Used only when algorithm detector has low confidence
    """
    code_lower = code.lower()
    
    if not is_quantum:
        return ProblemType.CLASSICAL
    
    # Check for algorithm patterns (simplified)
    if 'grover' in code_lower or 'oracle' in code_lower:
        return ProblemType.SEARCH
    
    if 'vqe' in code_lower or 'qaoa' in code_lower or 'optimizer' in code_lower:
        return ProblemType.OPTIMIZATION
    
    if 'shor' in code_lower or 'factor' in code_lower:
        return ProblemType.FACTORIZATION
    
    if 'qnn' in code_lower or 'machine' in code_lower:
        return ProblemType.MACHINE_LEARNING
    
    if 'qft' in code_lower or 'fourier' in code_lower:
        return ProblemType.SAMPLING
    
    # Default for quantum circuits
    return ProblemType.SIMULATION

def build_analysis_result(
    detected_lang: SupportedLanguage,
    lang_confidence: float,
    problem_type: ProblemType,
    classical_metrics,
    quantum_metrics,
    metadata: dict,
    detected_algorithms: list = None,
    algorithm_confidence: float = 0.0
) -> CodeAnalysisResult:
    """Build complete analysis result with accurate metrics"""
    
    if detected_algorithms is None:
        detected_algorithms = []
    
    # Extract values for Decision Engine
    if quantum_metrics:
        qubits = quantum_metrics.qubits_required
        depth = quantum_metrics.circuit_depth  
        gates = quantum_metrics.gate_count
        cx_ratio = quantum_metrics.cx_gate_ratio
        super_score = quantum_metrics.superposition_score  
        entangle_score = quantum_metrics.entanglement_score  
        
        # Determine time complexity based on problem type
        time_comp = determine_quantum_time_complexity(problem_type, quantum_metrics)
        
        # Memory requirement for simulation
        memory_mb = quantum_analyzer.estimate_memory_requirement(qubits)
        
        is_quantum_eligible = True
        problem_size = qubits
        
        # Build detailed notes
        notes = (
            f"Quantum analysis: {detected_lang.value} | "
            f"{qubits} qubits | Depth: {depth} (accurate) | "
            f"Superposition: {super_score:.2f} (simulated) | "
            f"Entanglement: {entangle_score:.2f} (simulated)"
        )
        
        if detected_algorithms:
            notes += f" | Detected: {', '.join(detected_algorithms)}"
        
        # Use algorithm confidence if available, otherwise language confidence
        confidence = max(algorithm_confidence, lang_confidence)
        
    else:
        # Classical code
        qubits = 0
        depth = 0
        gates = 0
        cx_ratio = 0.0
        super_score = 0.0
        entangle_score = 0.0
        
        # Time/space complexity from complexity_analyzer
        time_comp = classical_metrics.time_complexity if classical_metrics else TimeComplexity.LINEAR
        space_comp = classical_metrics.space_complexity if classical_metrics else "O(1)"
        
        # Estimate memory from space complexity
        memory_mb = estimate_classical_memory(space_comp)
        
        is_quantum_eligible = False
        problem_size = metadata.get('lines_of_code', 1)
        
        notes = (
            f"Classical analysis: {detected_lang.value} | "
            f"{metadata.get('lines_of_code', 0)} LOC | "
            f"Time: {time_comp.value} (accurate) | "
            f"Space: {space_comp} (accurate)"
        )
        
        confidence = lang_confidence
    
    return CodeAnalysisResult(
        detected_language=detected_lang.value,
        language_confidence=lang_confidence,
        problem_type=problem_type,
        problem_size=problem_size,
        
        # Detailed metrics
        classical_metrics=classical_metrics,
        quantum_metrics=quantum_metrics,
        
        # Unified fields (for Decision Engine)
        qubits_required=qubits,
        circuit_depth=depth,
        gate_count=gates,
        cx_gate_ratio=cx_ratio,
        superposition_score=super_score,
        entanglement_score=entangle_score,
        time_complexity=time_comp,
        memory_requirement_mb=memory_mb,
        
        is_quantum_eligible=is_quantum_eligible,
        confidence_score=confidence,
        analysis_notes=notes,
        detected_algorithms=detected_algorithms 
    )

def determine_quantum_time_complexity(
    problem_type: ProblemType, 
    quantum_metrics
) -> TimeComplexity:
    """
    Determine time complexity for quantum algorithms based on problem type
    """
    # Map problem types to known quantum time complexities
    complexity_map = {
        ProblemType.SEARCH: TimeComplexity.QUANTUM_ADVANTAGE,  # O(√n) - Grover
        ProblemType.FACTORIZATION: TimeComplexity.POLYNOMIAL,   # O(n³) - Shor
        ProblemType.OPTIMIZATION: TimeComplexity.POLYNOMIAL,    # VQE/QAOA
        ProblemType.SIMULATION: TimeComplexity.POLYNOMIAL,      # Quantum simulation
        ProblemType.SAMPLING: TimeComplexity.POLYNOMIAL,        # Sampling problems
        ProblemType.MACHINE_LEARNING: TimeComplexity.POLYNOMIAL,
        ProblemType.CRYPTOGRAPHY: TimeComplexity.EXPONENTIAL,   # Some crypto is exponential
    }
    
    # Use mapping if available
    if problem_type in complexity_map:
        return complexity_map[problem_type]
    
    # Fallback: check if circuit has quantum advantage characteristics
    if quantum_metrics.has_entanglement and quantum_metrics.entanglement_score > 0.7:
        return TimeComplexity.QUANTUM_ADVANTAGE
    
    return TimeComplexity.POLYNOMIAL

def estimate_classical_memory(space_complexity: str) -> float:
    """
    Estimate memory requirement in MB from space complexity
    Assumes n=1000 for estimation
    """
    estimates = {
        'O(1)': 0.001,          # Few variables
        'O(log(n))': 0.01,      # Logarithmic data structures
        'O(n)': 8.0,            # Array of 1000 doubles (8 bytes each)
        'O(n*log(n))': 10.0,    # Slightly more than linear
        'O(n^2)': 8000.0,       # 1000x1000 matrix
        'O(n*m)': 8000.0,       # 2D array
        'O(n^3)': 8_000_000.0,  # 3D array
        'O(2^n)': 1024.0,       # Exponential (capped at reasonable size)
    }
    
    return estimates.get(space_complexity, 1.0)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=True)