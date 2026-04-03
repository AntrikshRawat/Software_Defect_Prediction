from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# 1. Load your trained Random Forest model
model = joblib.load('rf_model.pkl')

# 2. Define the exact mapped feature names your model expects
feature_names = [
    'Lines of Code (LOC)', 'Cyclomatic Complexity (v(g))', 'Essential Complexity (ev(g))',
    'Design Complexity (iv(g))', 'Halstead Length (n)', 'Halstead Volume (v)',
    'Halstead Program Length (l)', 'Halstead Difficulty (d)', 'Halstead Intelligence (i)',
    'Halstead Effort (e)', 'Halstead Delivered Bugs (b)', 'Halstead Time Estimator (t)',
    'Lines of Executable Code', 'Lines of Comments', 'Lines of Blank Space',
    'Lines of Code & Comments', 'Unique Operators', 'Unique Operands',
    'Total Operators', 'Total Operands', 'Branch Count'
]

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    
    if request.method == 'POST':
        try:
            # 3. Collect all 21 inputs from the HTML form
            input_data = {}
            for feature in feature_names:
                # Get the value, default to 0.0 if empty
                val = request.form.get(feature, 0.0)
                input_data[feature] = [float(val)]
            
            # 4. Convert to a Pandas DataFrame (matches training format)
            df = pd.DataFrame(input_data)
            
            # 5. Get probability of defect (Index 1 is the 'defective' class)
            prob_defective = model.predict_proba(df)[0][1] * 100
            
            # 6. Apply your Risk Logic
            is_high_risk = prob_defective > 50
            status = "High Risk - Code Review Recommended" if is_high_risk else "Low Risk - Code metrics look stable."
            
            # Package result to send to HTML
            result = {
                "probability": f"{prob_defective:.1f}%",
                "status": status,
                "is_high_risk": is_high_risk
            }

        except Exception as e:
            result = {"error": f"Error processing input: {str(e)}"}

    # Pass the feature list so Jinja can build the form dynamically
    return render_template('index.html', result=result, feature_names=feature_names)

if __name__ == '__main__':
    app.run(debug=True)