from flask import Flask, request, render_template
import numpy as np
import joblib

# Load the model, scaler, and label encoder using joblib
rf_model = joblib.load("crop_app/random_forest_crop_model.pkl")
scaler = joblib.load("crop_app/feature_scaler.pkl")
le = joblib.load("crop_app/crop_label_encoder.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Fetching input values from form
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorous'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        # Prepare the input array (make sure it's a 2D array)
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        # Ensure the input is in the form (1, 7), i.e., 1 row with 7 columns
        single_pred = np.array(feature_list).reshape(1, -1)

        # Apply scaling transformations
        scaled_features = scaler.transform(single_pred)

        # Predict the crop
        prediction = rf_model.predict(scaled_features)
        
        # If you used LabelEncoder, you can inverse the label encoding if needed
        predicted_crop = le.inverse_transform(prediction)

        result = f"{predicted_crop[0]} is the best crop to be cultivated in the provided conditions."
        return render_template('index.html', result=result)
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
