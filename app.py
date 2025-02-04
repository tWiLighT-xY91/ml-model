from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("lung_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get input data from form
            input_data = [float(x) for x in request.form.values()]
            input_data = np.array(input_data).reshape(1, -1)

            # Scale the input
            scaled_data = scaler.transform(input_data)

            # Predict
            prediction = model.predict(scaled_data)

            # Interpret result
            result = "Malignant" if prediction[0] == 1 else "Benign"

            return render_template("index.html", prediction_text=f"Prediction: {result}")

        except Exception as e:
            return render_template("index.html", prediction_text=f"Error: {e}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
