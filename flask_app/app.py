from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "breast_cancer_best_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURE_ORDER = [
    'Clump_thickness',
    'Uniformity_of_cell_size',
    'Uniformity_of_cell_shape',
    'Marginal_adhesion',
    'Single_epithelial_cell_size',
    'Bare_nuclei',
    'Bland_chromatin',
    'Normal_nucleoli',
    'Mitoses'
]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    values = [float(request.form[feat]) for feat in FEATURE_ORDER]
    X_input = np.array([values])
    pred = int(model.predict(X_input)[0])
    return render_template("index.html", result=pred)

if __name__ == "__main__":
    app.run(debug=True)