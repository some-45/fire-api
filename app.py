from flask import Flask, request, jsonify
import joblib

# تحميل النموذج باسمك
model = joblib.load("svm_fire_model_balanced.joblib")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    values = data["data"]  # بيانات الحساسات
    prediction = int(model.predict([values])[0])
    return jsonify({"prediction": prediction})