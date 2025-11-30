"""
Code Analysis Engine - Complete Implementation
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
from models.analysis_result import CodeAnalysisResult, ProblemType, TimeComplexity

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
            "Complexity analysis",
            "Quantum metrics extraction",
            "Algorithm detection (coming soon)"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "code-analysis-engine"}

@app.post("/detect-language", response_model=LanguageDetectionResponse)
async def detect_language(submission: CodeSubmission):
    """Detect programming language"""
    try:
        result = language_detector.detect(submission.code)
        return LanguageDetectionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=CodeAnalysisResult)
async def analyze_code(submission: CodeSubmission):
    """
    Complete code analysis pipeline
    Returns metrics for Decision Engine
    """
    try:
        code = submission.code
        
        # Step 1: Detect language
        lang_result = language_detector.detect(code)
        
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
        
        # Step 3: Analyze complexity
        is_quantum = detected_lang in {
            SupportedLanguage.QISKIT, 
            SupportedLanguage.CIRQ,
            SupportedLanguage.OPENQASM,
            SupportedLanguage.QSHARP
        }
        
        classical_metrics = None
        quantum_metrics = None
        
        if is_quantum:
            # Quantum analysis
            quantum_metrics = quantum_analyzer.analyze(unified_ast)
            
            # Also get classical metrics if there's classical code
            if metadata['lines_of_code'] > 0:
                classical_metrics = complexity_analyzer.analyze(code, metadata)
        else:
            # Pure classical analysis
            classical_metrics = complexity_analyzer.analyze(code, metadata)
        
        # Step 4: Determine problem type (heuristic-based)
        problem_type = determine_problem_type(unified_ast, code, is_quantum)
        
        # Step 5: Build result for Decision Engine
        result = build_analysis_result(
            detected_lang=detected_lang,
            lang_confidence=lang_result["confidence"],
            problem_type=problem_type,
            classical_metrics=classical_metrics,
            quantum_metrics=quantum_metrics,
            metadata=metadata
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
def determine_problem_type(unified_ast, code: str, is_quantum: bool) -> ProblemType:
    """Heuristic-based problem type classification"""
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
    metadata: dict
) -> CodeAnalysisResult:
    """Build complete analysis result"""
    
    # Extract values for Decision Engine
    if quantum_metrics:
        qubits = quantum_metrics.qubits_required
        depth = quantum_metrics.circuit_depth
        gates = quantum_metrics.gate_count
        cx_ratio = quantum_metrics.cx_gate_ratio
        super_score = quantum_metrics.superposition_score
        entangle_score = quantum_metrics.entanglement_score
        time_comp = TimeComplexity.QUANTUM_ADVANTAGE if quantum_metrics.has_entanglement else TimeComplexity.LINEAR
        memory_mb = quantum_analyzer.estimate_memory_requirement(qubits)
        is_quantum_eligible = True
        problem_size = qubits
    else:
        qubits = 0
        depth = 0
        gates = 0
        cx_ratio = 0.0
        super_score = 0.0
        entangle_score = 0.0
        time_comp = classical_metrics.time_complexity if classical_metrics else TimeComplexity.LINEAR
        memory_mb = 1.0  # Default for classical
        is_quantum_eligible = False
        problem_size = metadata.get('lines_of_code', 1)
    
    return CodeAnalysisResult(
        detected_language=detected_lang.value,
        language_confidence=lang_confidence,
        problem_type=problem_type,
        problem_size=problem_size,
        classical_metrics=classical_metrics,
        quantum_metrics=quantum_metrics,
        qubits_required=qubits,
        circuit_depth=depth,
        gate_count=gates,
        cx_gate_ratio=cx_ratio,
        superposition_score=super_score,
        entanglement_score=entangle_score,
        time_complexity=time_comp,
        memory_requirement_mb=memory_mb,
        is_quantum_eligible=is_quantum_eligible,
        confidence_score=lang_confidence,
        analysis_notes=f"Analyzed {detected_lang.value} code with {metadata.get('lines_of_code', 0)} LOC"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=True)