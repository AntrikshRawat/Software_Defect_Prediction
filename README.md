# Software Defect Risk Analyzer

Software Defect Risk Analyzer is a Flask web application that estimates defect risk for Python code and CSV datasets using a trained Random Forest model. The app extracts the same 21 software metrics expected by the model, then converts the prediction into an easy-to-read risk label and visualization.

## Overview

The project has two main workflows:

1. Analyze a single Python module by pasting code or uploading a `.py` or `.txt` file.
2. Analyze a CSV file in bulk when the file already contains the 21 required JM1-style metric columns.

The frontend presents these workflows as separate tabs and shows the prediction result, extracted metrics, and generated charts directly in the browser.

## Features

* Python code analysis from pasted source or uploaded files.
* Bulk prediction for CSV files with JM1-compatible feature columns.
* Automatic metric extraction using Radon and Python AST parsing.
* Probability output with a risk threshold of 35%.
* Generated charts embedded in the page as base64-encoded images.
* Clean tabbed interface for snippet, file, and CSV workflows.

## How It Works

### Single code analysis

When you submit Python code, the app:

1. Reads the pasted code or uploaded file.
2. Extracts 21 metrics with `code_parser.extract_metrics()`.
3. Builds a one-row pandas DataFrame in the exact column order expected by the model.
4. Loads `rf_model.pkl` and calls `predict_proba()`.
5. Marks the result as high risk when the predicted probability is greater than 35%.
6. Renders a gauge chart showing the module's defect probability.

### Bulk CSV analysis

When you upload a CSV file, the app:

1. Reads the file with pandas.
2. Verifies that all required model features are present.
3. Reorders the columns to match the model input order.
4. Runs batch predictions across all rows.
5. Displays a pie chart for the high-risk vs low-risk distribution.
6. Displays a histogram of defect probabilities.
7. Lists each row with its predicted probability and risk label.

## Project Structure

* `app.py` - Flask application, routes, model loading, prediction workflow, and chart generation.
* `code_parser.py` - Converts raw Python code into the 21 model features.
* `rf_model.pkl` - Trained Random Forest model used for scoring.
* `dataset.csv` - Sample CSV data using the expected metric columns.
* `templates/index.html` - Main interface for code and CSV analysis.
* `templates/style.css` - Styling for the dashboard and tab layout.
* `README.md` - Project documentation.

## Required Feature Columns

The model expects these exact columns, in this order:

1. Lines of Code (LOC)
2. Cyclomatic Complexity (v(g))
3. Essential Complexity (ev(g))
4. Design Complexity (iv(g))
5. Halstead Length (n)
6. Halstead Volume (v)
7. Halstead Program Length (l)
8. Halstead Difficulty (d)
9. Halstead Intelligence (i)
10. Halstead Effort (e)
11. Halstead Delivered Bugs (b)
12. Halstead Time Estimator (t)
13. Lines of Executable Code
14. Lines of Comments
15. Lines of Blank Space
16. Lines of Code & Comments
17. Unique Operators
18. Unique Operands
19. Total Operators
20. Total Operands
21. Branch Count

## Metric Extraction

The parser uses Radon and Python's AST module to derive metrics from source code:

* Raw counts from `radon.raw.analyze()`.
* Halstead metrics from `radon.metrics.h_visit()`.
* Cyclomatic complexity from `radon.complexity.cc_visit()`.
* Branch count from AST traversal across `if`, `for`, `while`, and `try` nodes.

If parsing fails, the app returns a readable error message to the UI.

## CSV Format

The bulk upload workflow requires a CSV that already contains the full set of 21 feature columns listed above. The included `dataset.csv` file demonstrates the expected schema.

If any required column is missing, the app stops and shows an error.

## Setup

### Prerequisites

* Python 3.8 or later
* The repository files, including `rf_model.pkl`

### Install dependencies

If you are using a virtual environment, activate it first, then install the project packages:

```bash
pip install -r requirements.txt
```

### Run the app

```bash
python app.py
```

Open the app in your browser at `http://127.0.0.1:5000`.

## Dependencies

The project uses:

* Flask for the web server and templating.
* Joblib for loading the trained model.
* Pandas for tabular data handling.
* Matplotlib and Seaborn for charts.
* Radon for code metrics.
* scikit-learn for the trained classifier stored in `rf_model.pkl`.

## UI Behavior

The interface includes three tabs:

* Code Snippet - paste Python code directly.
* Code File - upload a `.py` or `.txt` file.
* CSV File - upload a metric table for batch scoring.

The page also keeps the selected file name visible after upload and hides or shows the result panel based on the active tab.

## Notes

* The app is designed for Python source code when extracting metrics from raw code.
* The model threshold for high risk is hardcoded at 35% in `app.py`.
* The application loads `rf_model.pkl` at startup, so the file must exist in the project root.
* Static styling is served from `templates/style.css` through a Flask route.

## License

This project is distributed under the terms of the included `LICENSE` file.


