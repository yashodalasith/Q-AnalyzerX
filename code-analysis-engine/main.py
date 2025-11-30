"""
Code Analysis Engine - Main FastAPI Application
Port: 8002
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Import your modules
from modules.language_detector import LanguageDetector, SupportedLanguage

app = FastAPI(
    title="Code Analysis Engine",
    description="Analyzes quantum code and detects quantum algorithms",
    version="0.1.0"
)

# Initialize components
language_detector = LanguageDetector()

# Request/Response Models
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
        "message": "Code Analysis Engine API",
        "version": "0.1.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "code-analysis-engine"
    }

@app.post("/detect-language", response_model=LanguageDetectionResponse)
async def detect_language(submission: CodeSubmission):
    """
    Detect the programming language of submitted code
    
    Supports: Python, Qiskit, Q#, Cirq, OpenQASM
    """
    try:
        result = language_detector.detect(submission.code)
        return LanguageDetectionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/supported-languages")
async def get_supported_languages():
    """List all supported quantum programming languages"""
    return {
        "languages": [lang.value for lang in SupportedLanguage if lang != SupportedLanguage.UNKNOWN],
        "count": len(SupportedLanguage) - 1
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=True)