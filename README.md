# Q-AnalyzerX AI – Code Analysis Engine

### Hybrid Quantum–Classical Code Analyzer | AST Parsing | ML-Based Pattern Recognition

Q-AnalyzerX AI is an intelligent **Code Analysis Engine (CAE)** designed to analyze, classify, and understand **quantum + classical code**.
It is a core component of the **Quantum–Classical Code Router**, enabling automated workload routing to the best execution environment (CPU/GPU/QPU).

This engine uses:

- **AST Parsing** for Qiskit, Q#, Cirq, OpenQASM
- **Quantum Operation Detection** (gates, measurements, registers)
- **Complexity Analysis** (cyclomatic complexity, gate depth, qubit count)
- **Machine Learning–based Algorithm Pattern Recognition**
- **REST API (FastAPI)** for real-time analysis
- **React + Vite Frontend** for rapid UI access

---

## 🔥 Features

### ✅ Multi-Language Quantum Parsing

Supports:

- Python / Qiskit
- Q#
- Cirq
- OpenQASM

Converts code → AST → Unified Intermediate Representation for analysis.

### ✅ Quantum Operation Detection

Automatically detects:

- Quantum gates (H, X, Z, CNOT, Toffoli, etc.)
- Qubit/register allocations
- Measurement operations
- Quantum circuit structures
- Hybrid classical-quantum regions

### ✅ Complexity Analysis

Produces:

- Cyclomatic complexity
- Classical time complexity (Big-O estimation)
- Quantum circuit depth & width
- Qubit requirements
- Parallelization potential

### ✅ Machine Learning Pattern Recognition

Identifies well-known quantum algorithms such as:

- Grover’s Search
- Shor’s Algorithm
- QFT (Quantum Fourier Transform)
- VQE / Variational Algorithms

Outputs include:

- Classification
- Confidence scores
- Optimization recommendations

---

## 🧠 1. Code Analysis Engine (Python + FastAPI)

### Step 1: Navigate into the engine

```bash
cd code-analysis-engine
```

### Step 2: Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Linux / Mac
venv\Scripts\activate        # Windows
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the server

```bash
uvicorn main:app --reload
```

➡ Default Port: http://localhost:8000

---

## 🖥 2. Frontend (React + Vite)

### Step 1: Navigate to frontend

```bash
cd frontend
```

### Step 2: Install dependencies

```bash
npm install
```

### Step 3: Start development server

```bash
npm run dev
```

➡ Access UI at: http://localhost:5173

---

## 🌐 3. API Gateway (Node.js / TypeScript)

### Step 1: Navigate to API folder

```bash
cd api
```

### Step 2: Install dependencies

```bash
npm install
```

### Step 3: Start server

```bash
npm run dev
```

---

## 🧪 API Example

### POST /api/v1/analyze

```json
{
  "code": "from qiskit import QuantumCircuit",
  "language": "qiskit",
  "options": {
    "include_complexity": true,
    "include_patterns": true
  }
}
```

### Example Response

```json
{
  "classification": "quantum_advantageous",
  "confidence": 0.92,
  "complexity": {
    "qubit_count": 4,
    "circuit_depth": 23
  },
  "patterns": ["qft"]
}
```

---

## 📌 Prerequisites

### Python

- Python 3.9+
- pip

### Node.js

- Node.js 16+
- npm or yarn

### System Requirements

- 8+ GB RAM recommended
- Internet connection (for model loading & API calling)

---

## 📚 Technologies Used

- FastAPI — REST backend
- Pydantic — request validation
- ANTLR4 — multi-language parsing
- TensorFlow / Scikit-Learn — ML layer
- React + Vite — UI
- Node.js + TypeScript — API layer

---

## 📄 License

MIT License © 2025 Yashodha Jayasinghe

---

## 🤝 Contribution

Pull requests & suggestions are welcome.

---

## ⭐ If you use this project…

Give the repo a star to support further development!
