# 🛡️ LogScouts - Web Threat Prediction System

A machine learning-based web application that analyzes Apache server log files to detect and predict security threats including DDoS attacks, brute force attempts, XSS, and SQL injection.

## Overview

This system uses ensemble machine learning (Random Forest + Gradient Boosting) to analyze server logs and identify potential security threats. It features a user-friendly web interface where users can upload log files and receive detailed threat analysis with visualizations and security recommendations.

### Key Features

- **Multi-Threat Detection**: Detects DDoS, Brute Force, XSS, and SQL Injection attacks
- **93%+ Accuracy**: Ensemble model combining Random Forest and Gradient Boosting
- **Real-Time Analysis**: Processes 110,000+ log entries in 3-4 minutes
- **Interactive Dashboard**: Visual threat distribution and detailed reports
- **Security Recommendations**: Actionable advice based on detected threats

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- 4GB RAM minimum (8GB recommended)

### Installation

**1. Clone or download the project**

```bash
git clone <repository-url>
cd threat-prediction-system
```

**2. Create virtual environment**

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

On Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install required packages**

```bash
pip install fastapi uvicorn pandas numpy scikit-learn python-multipart pydantic seaborn matplotlib
```

Or using requirements.txt:
```bash
pip install -r requirements.txt
```

**4. Create necessary folders**

On Windows:
```bash
mkdir backend frontend models data\raw data\processed uploads results scripts
```

On Linux/Mac:
```bash
mkdir -p backend frontend models data/raw data/processed uploads results scripts
```

---

## Running the Application

**Start Backend (Terminal 1):**
```bash
python backend/api.py
```

**Start Frontend (Terminal 2):**
```bash
cd frontend
python -m http.server 3000
```

**Access the application:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

---

## Usage

1. Open http://localhost:3000 in your browser
2. Drag and drop your log file (CSV)
    - Expected CSV format:
        ```
        ip,timestamp,method,uri,status,payload,refer,user_agent
        ```
3. Click "Analyze Log File"
4. Wait 2-3 minutes for processing
5. View results:
   - Total requests analyzed
   - Threats detected with percentages
   - Threat distribution chart
   - High-risk IP addresses
   - Security recommendations

---

## Troubleshooting

**Models not found:**
```bash
# Ensure models exist
ls models/

# If missing, train them
python scripts/train_models.py
```

**Port already in use:**
```bash
# Change port in backend/api.py or frontend command
python -m http.server 3001  # Use different port
```

**Import errors:**
```bash
# Reinstall packages
pip install --upgrade -r requirements.txt
```
