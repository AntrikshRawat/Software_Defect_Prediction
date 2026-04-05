from flask import Flask, render_template, request
import joblib
import pandas as pd
import io
import base64

import matplotlib
matplotlib.use('Agg') # Safe for web threading
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# 1. Load the freshly trained Random Forest model
model = joblib.load('rf_model.pkl')

feature_names = [
    'Lines of Code (LOC)', 'Cyclomatic Complexity (v(g))', 'Essential Complexity (ev(g))',
    'Design Complexity (iv(g))', 'Halstead Length (n)', 'Halstead Volume (v)',
    'Halstead Program Length (l)', 'Halstead Difficulty (d)', 'Halstead Intelligence (i)',
    'Halstead Effort (e)', 'Halstead Delivered Bugs (b)', 'Halstead Time Estimator (t)',
    'Lines of Executable Code', 'Lines of Comments', 'Lines of Blank Space',
    'Lines of Code & Comments', 'Unique Operators', 'Unique Operands',
    'Total Operators', 'Total Operands', 'Branch Count'
]

def generate_bulk_plots(probs, bulk_results):
    """Generates visualization image for bulk results as base64."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Chart 1: Risk Distribution Pie Chart
    high_risk_count = sum(1 for r in bulk_results if r['is_high_risk'])
    low_risk_count = len(bulk_results) - high_risk_count
    
    if len(bulk_results) > 0:
        axes[0].pie([high_risk_count, low_risk_count], 
                    labels=['High Risk', 'Low Risk'], 
                    autopct='%1.1f%%', 
                    colors=['#ef5350', '#66bb6a'], 
                    startangle=90,
                    wedgeprops={'edgecolor': 'white'})
    axes[0].set_title('Overall Risk Distribution', fontsize=14, fontweight='bold')

    # Chart 2: Probability Histogram
    sns.histplot(probs, bins=10, ax=axes[1], color='#2c3e50', kde=True)
    axes[1].set_title('Defect Probability Spread', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Probability of Defect (%)')
    axes[1].set_ylabel('Number of Modules')
    
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close() 
    
    return plot_url

def generate_single_gauge_plot(probability, threshold):
    """Generates a simple gauge image for a single probability as base64."""
    fig, ax = plt.subplots(figsize=(6, 1)) 
    
    # Draw colored background zones
    ax.axvspan(0, threshold, color='#66bb6a', alpha=0.3) # Low Risk Zone
    ax.axvspan(threshold, 100, color='#ef5350', alpha=0.3) # High Risk Zone
    ax.axvline(x=threshold, color='#f44336', linestyle='--', linewidth=2) # Threshold Line
    
    # Draw probability point
    ax.plot(probability, 0, marker='D', markersize=12, markerfacecolor='#2c3e50', markeredgecolor='white', markeredgewidth=1.5)
    
    # Text labels
    ax.text(probability, 0.2, f'{probability:.1f}%', color='#2c3e50', fontsize=11, fontweight='bold', ha='center', va='bottom')
    ax.text(0, -0.4, '0%', color='#66bb6a', ha='left')
    ax.text(100, -0.4, '100%', color='#ef5350', ha='right')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, 0.7)
    ax.set_yticks([]) 
    ax.set_title('Module Defect Probability Gauge', fontsize=12, fontweight='bold')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    
    return plot_url

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    
    if request.method == 'POST':
        try:
            # CASE A: CSV File Upload
            if 'csv_file' in request.files and request.files['csv_file'].filename != '':
                file = request.files['csv_file']
                df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")))
                df = df[feature_names] 
                
                probs = model.predict_proba(df)[:, 1] * 100
                
                bulk_results = []
                for p in probs:
                    is_high = p > 50
                    bulk_results.append({
                        "probability": f"{p:.1f}%",
                        "status": "High Risk" if is_high else "Low Risk",
                        "is_high_risk": is_high
                    })
                
                plot_b64 = generate_bulk_plots(probs, bulk_results)
                result = {"type": "bulk", "data": bulk_results, "plot_url": plot_b64}

            # CASE B: Manual Form Submission
            else:
                input_data = {}
                for feature in feature_names:
                    val = request.form.get(feature, 0.0)
                    input_data[feature] = [float(val)]
                
                df = pd.DataFrame(input_data)
                prob_defective = model.predict_proba(df)[0][1] * 100
                is_high_risk = prob_defective > 50
                
                single_plot_b64 = generate_single_gauge_plot(prob_defective, 50)
                
                result = {
                    "type": "single",
                    "probability": f"{prob_defective:.1f}%",
                    "status": "High Risk - Code Review Recommended" if is_high_risk else "Low Risk - Metrics Stable",
                    "is_high_risk": is_high_risk,
                    "plot_url": single_plot_b64 
                }

        except Exception as e:
            result = {"error": f"Error: {str(e)}. Ensure CSV columns match exactly."}

    return render_template('index.html', result=result, feature_names=feature_names)

# THIS IS THE VITAL PART THAT WAS MISSING!
if __name__ == '__main__':
    app.run(debug=True)