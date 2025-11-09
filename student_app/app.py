import os, pickle
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

pipeline = model["pipeline"]
classes = model["classes"]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    math = float(request.form["math_score"])
    reading = float(request.form["reading_score"])
    writing = float(request.form["writing_score"])
    X = np.array([[math, reading, writing]])
    pred_idx = pipeline.predict(X)[0]
    result = classes[int(pred_idx)]
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)