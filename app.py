from flask import Flask, request, jsonify
import joblib

# تحميل النموذج
model = joblib.load("svm_fire_model_balanced.joblib")

# إنشاء التطبيق
app = Flask(name)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    values = data["data"]
    prediction = int(model.predict([values])[0])
    return jsonify({"prediction": prediction})

# تشغيل التطبيق
if name == "main":
    app.run(host="0.0.0.0",
    port=10000, debug=True)