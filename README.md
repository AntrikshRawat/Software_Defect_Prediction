# 🚨 Software Defect Risk Analyzer

An interactive machine learning web application that predicts the probability of software defects in Python code using Halstead metrics, McCabe's Cyclomatic Complexity, and a trained Random Forest model.

## 🌟 Features

* **Direct Code Analysis:** Paste raw Python code (or upload a `.py` file) directly into the browser. The built-in Abstract Syntax Tree (AST) engine automatically extracts 21 engineering metrics and generates a risk score.
* **Bulk CSV Analysis:** Upload a dataset of modules (like the JM1 dataset) to instantly evaluate hundreds of files at once, complete with a visual probability dashboard and risk highlighting.
* **Manual Metric Entry:** A form for rapid testing or analyzing modules written in other languages where pre-calculated metrics are available.
* **Dynamic Visualization:** Generates real-time risk distribution pie charts, probability histograms, and individual risk gauges using Matplotlib and base64 encoding.

## 🛠️ Technology Stack

* **Backend:** Python, Flask
* **Machine Learning:** Scikit-Learn (`RandomForestClassifier`), Pandas, NumPy, Joblib
* **AST Parsing & Metrics:** Radon (`radon.raw`, `radon.metrics`, `radon.complexity`)
* **Data Visualization:** Matplotlib, Seaborn
* **Frontend:** HTML5, CSS3 Grid, Jinja2 Templating

## 🚀 How to Run Locally

### Prerequisites
Ensure you have Python 3.8+ installed on your machine. 

### 1. Clone the repository
```bash
git clone [https://github.com/AntrikshRawat/Software_Defect_Prediction.git](https://github.com/AntrikshRawat/Software_Defect_Prediction.git)
cd Software_Defect_Prediction
```

### 2. Install dependencies

It is highly recommended to use a virtual environment.
```bash
pip install flask scikit-learn pandas numpy matplotlib seaborn radon joblib
```

### 3. Start the server
```bash
python app.py
```
The application will be live at --> ```bash http://127.0.0.1:5000```

## 🧠 The Machine Learning Model

The core of this application is a **Random Forest Classifier** trained on the **JM1 Software Defect Dataset** (originally sourced from NASA). The model evaluates 21 distinct code complexity metrics to determine if a module is **"High Risk"** (defective) or **"Low Risk"** (clean).

### Extracted Metrics Include:

* **McCabe's Cyclomatic Complexity `v(g)`:** Measures the number of linearly independent paths through the code.
* **Halstead Base Metrics:** Unique/Total Operators (`n1`, `N1`) and Operands (`n2`, `N2`).
* **Halstead Derived Metrics:** Volume (`V`), Difficulty (`D`), Effort (`E`), and Estimated Delivered Bugs (`B`).
* **Raw Counts:** Lines of Code (LOC), Branch Counts, and Blank/Comment lines.

