from flask import Flask, request, jsonify
import joblib
import numpy as np

# تحميل النموذج
model = joblib.load("svm_fire_model_balanced.joblib")

# إنشاء التطبيق
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "data" not in data:
        return jsonify({"error":"Missing 'data' field"}),400
    try:
            features=np.array(data["data"]).reshape(1, -1)
            prediction=model.predict(features)[0]
            return jsonify({"partition":int (prediction)})
   except Exception as e:
                return jsonify({"error":str(e)}),500
 
# تشغيل التطبيق
if __name__ == "__main__":
    app.run(host="0.0.0.0",
    port=10000, debug=True)