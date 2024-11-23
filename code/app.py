from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model pipeline
pipeline = joblib.load("malware_pipeline.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    signature = data.get("signature", "")
    prediction = pipeline.predict([signature])
    probability = pipeline.predict_proba([signature])[:, 1][0]
    return jsonify(
        {
            "prediction": "malware" if prediction[0] == 1 else "benign",
            "probability": probability,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
