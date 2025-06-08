from flask import Flask, request, render_template
import numpy as np
from joblib import load

model = load('model.joblib')
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    inputs = [float(x) for x in request.form.values()]
    prediction = model.predict([inputs])[0]
    return render_template('index.html', prediction_text=f'Tumor is {"Malignant" if prediction == 0 else "Benign"}')

if __name__ == "__main__":
    app.run(debug=True)
