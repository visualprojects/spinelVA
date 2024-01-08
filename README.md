# spinelva

Welcome to the SpinelVA repository. This tool integrates a Machine Learning approach to automatically identify and analyze the geochemical composition of spinel group minerals based on their host rocks, providing several views and configurations for analysi. 
Follow the steps below to set up the environment and run the application.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- Python (version 3.10.0)
- Pip (package installer for Python)
- Flask
- scikit-learn

## Getting Started

Follow these steps to get the project up and running:

### 1. Create a Virtual Environment

```bash
# Navigate to the project directory
cd path/to/project

# Create a virtual environment
python -m venv venv
Activate the virtual environment:
```

* On Windows:
```bash
venv\Scripts\activate
```

* On macOS/Linux:
```bash
source venv/bin/activate
```

### 2. Install Dependencies
```python
# Ensure you are in the virtual environment
# Install required packages
pip install -r requirements.txt
```

### 3. Run the Application
```python
# Execute the main.py script
python main.py
```

### 4. Open Browser
Once the application is running, open your browser and navigate to http://localhost:5501/geoviz to access the SpinelVA.
Due to security and performance issues, the frontend will connect to a server (similar to the one detailed in the main.py file) at the address where the tool is running natively.
