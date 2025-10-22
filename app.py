# --- Heart Disease Prediction Web App with Landing Page ---

# 1. Importing necessary libraries
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string, Response
import pickle
import io

# 2. Initialize the Flask app
app = Flask(__name__)

# 3. Load the pre-trained model
# IMPORTANT: Make sure 'random_forest_heart_model.pkl' is in the same directory.
try:
    model = pickle.load(open('random_forest_heart_model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: 'random_forest_heart_model.pkl' not found.")
    model = None

# 4. Define the updated HTML template with a landing page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Heart Disease Prediction</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
        body {
            font-family: 'Roboto', sans-serif;
        }
        .form-select {
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%236b7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e");
            background-position: right 0.5rem center;
            background-repeat: no-repeat;
            background-size: 1.5em 1.5em;
            -webkit-print-color-adjust: exact;
            color-adjust: exact;
            appearance: none;
        }
        /* Styles for smooth transition */
        .hidden-panel {
            opacity: 0;
            display: none;
            transition: opacity 0.5s ease-in-out;
        }
        .visible-panel {
            opacity: 1;
            display: block;
        }
    </style>
</head>
<body class="bg-gradient-to-br from-indigo-100 via-purple-50 to-pink-100 flex items-center justify-center min-h-screen">
    <div class="container mx-auto p-4 md:p-8 text-center">

        <div id="landing-page" class="transition-opacity duration-500 ease-in-out">
             <div class="mb-6">
                <svg class="w-20 h-20 mx-auto text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"></path></svg>
            </div>
            <h1 class="text-4xl md:text-6xl font-bold text-gray-800 mb-4">Heart Disease Predictor</h1>
            <p class="max-w-2xl mx-auto text-lg text-gray-600 mb-8">
                Utilize our advanced machine learning model to assess the risk of heart disease based on key health indicators.
            </p>
            <button id="start-btn" class="inline-flex justify-center items-center px-10 py-4 border border-transparent text-lg font-medium rounded-full shadow-lg text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-all transform hover:scale-105">
                Start Prediction
            </button>
        </div>

        <div id="prediction-panel" class="bg-white rounded-2xl shadow-lg p-6 md:p-10 max-w-4xl mx-auto hidden-panel">
            <h1 class="text-3xl md:text-4xl font-bold text-center text-gray-800 mb-2">Patient Health Data</h1>
            <p class="text-center text-gray-500 mb-8">Enter the values below to get a prediction.</p>

            <div class="mb-6 border-b border-gray-200">
                <nav class="flex -mb-px" aria-label="Tabs">
                    <button onclick="changeTab('single')" id="tab-single" class="w-1/2 py-4 px-1 text-center border-b-2 font-medium text-sm text-indigo-600 border-indigo-500">
                        Single Prediction
                    </button>
                    <button onclick="changeTab('bulk')" id="tab-bulk" class="w-1/2 py-4 px-1 text-center border-b-2 font-medium text-sm text-gray-500 border-transparent hover:text-gray-700 hover:border-gray-300">
                        Bulk Prediction
                    </button>
                </nav>
            </div>

            <div id="panel-single">
                 <form id="prediction-form" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    <div>
                        <label for="age" class="block text-sm font-medium text-gray-700">Age</label>
                        <input type="number" name="age" id="age" required class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2" placeholder="e.g., 52">
                    </div>
                    <div>
                        <label for="sex" class="block text-sm font-medium text-gray-700">Sex</label>
                        <select name="sex" id="sex" required class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2 form-select">
                            <option value="1">Male</option>
                            <option value="0">Female</option>
                        </select>
                    </div>
                    <div>
                        <label for="cp" class="block text-sm font-medium text-gray-700">Chest Pain Type (CP)</label>
                        <select name="cp" id="cp" required class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2 form-select">
                            <option value="0">Typical Angina</option>
                            <option value="1">Atypical Angina</option>
                            <option value="2">Non-anginal Pain</option>
                            <option value="3">Asymptomatic</option>
                        </select>
                    </div>
                     <div>
                        <label for="trestbps" class="block text-sm font-medium text-gray-700">Resting Blood Pressure</label>
                        <input type="number" name="trestbps" id="trestbps" required class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2" placeholder="e.g., 125">
                    </div>
                    <div>
                        <label for="chol" class="block text-sm font-medium text-gray-700">Cholesterol (chol)</label>
                        <input type="number" name="chol" id="chol" required class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2" placeholder="e.g., 212">
                    </div>
                     <div>
                        <label for="fbs" class="block text-sm font-medium text-gray-700">Fasting Blood Sugar > 120 mg/dl</label>
                        <select name="fbs" id="fbs" required class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2 form-select">
                            <option value="1">True</option>
                            <option value="0">False</option>
                        </select>
                    </div>
                    <div>
                        <label for="restecg" class="block text-sm font-medium text-gray-700">Resting ECG (restecg)</label>
                        <select name="restecg" id="restecg" required class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2 form-select">
                            <option value="0">Normal</option>
                            <option value="1">ST-T wave abnormality</option>
                            <option value="2">Probable or definite LVH</option>
                        </select>
                    </div>
                    <div>
                        <label for="thalach" class="block text-sm font-medium text-gray-700">Max Heart Rate Achieved</label>
                        <input type="number" name="thalach" id="thalach" required class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2" placeholder="e.g., 168">
                    </div>
                    <div>
                        <label for="exang" class="block text-sm font-medium text-gray-700">Exercise Induced Angina</label>
                         <select name="exang" id="exang" required class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2 form-select">
                            <option value="1">Yes</option>
                            <option value="0">No</option>
                        </select>
                    </div>
                    <div>
                        <label for="oldpeak" class="block text-sm font-medium text-gray-700">ST depression (oldpeak)</label>
                        <input type="number" step="0.1" name="oldpeak" id="oldpeak" required class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2" placeholder="e.g., 1.0">
                    </div>
                    <div>
                        <label for="slope" class="block text-sm font-medium text-gray-700">Slope of Peak Exercise ST</label>
                        <select name="slope" id="slope" required class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2 form-select">
                            <option value="0">Upsloping</option>
                            <option value="1">Flat</option>
                            <option value="2">Downsloping</option>
                        </select>
                    </div>
                     <div>
                        <label for="ca" class="block text-sm font-medium text-gray-700">Number of Major Vessels (ca)</label>
                        <input type="number" name="ca" id="ca" required class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2" placeholder="0-4">
                    </div>
                    <div>
                        <label for="thal" class="block text-sm font-medium text-gray-700">Thalassemia (thal)</label>
                        <select name="thal" id="thal" required class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2 form-select">
                            <option value="0">N/A</option>
                            <option value="1">Normal</option>
                            <option value="2">Fixed Defect</option>
                            <option value="3">Reversible Defect</option>
                        </select>
                    </div>
                    <div class="md:col-span-2 lg:col-span-3 text-center mt-4">
                         <button type="submit" class="w-full md:w-auto inline-flex justify-center items-center px-8 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-all">
                            Predict
                        </button>
                    </div>
                </form>
                <div id="result-container" class="mt-8 text-center" style="display: none;">
                    <h2 class="text-2xl font-semibold text-gray-800">Prediction Result</h2>
                    <p id="result-text" class="text-xl mt-2 p-4 rounded-lg"></p>
                </div>
            </div>

            <div id="panel-bulk" style="display: none;">
                 <form action="/bulk_predict" method="post" enctype="multipart/form-data">
                    <div class="text-center">
                        <label for="bulk_file" class="block text-sm font-medium text-gray-700 mb-2">Upload a CSV file with patient data:</label>
                        <input type="file" name="bulk_file" id="bulk_file" required class="mx-auto block w-full max-w-xs text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100">
                        <p class="mt-2 text-xs text-gray-500">CSV should have columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal</p>
                    </div>
                    <div class="text-center mt-6">
                        <button type="submit" class="w-full md:w-auto inline-flex justify-center items-center px-8 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-all">
                            Predict and Download CSV
                        </button>
                    </div>
                </form>
            </div>
             <div id="error-message" class="mt-6 text-center text-red-500 font-semibold"></div>
        </div>
    </div>

    <script>
        // --- NEW: Landing Page Logic ---
        document.getElementById('start-btn').addEventListener('click', function() {
            const landingPage = document.getElementById('landing-page');
            const predictionPanel = document.getElementById('prediction-panel');

            // Fade out the landing page
            landingPage.style.opacity = '0';

            setTimeout(() => {
                landingPage.style.display = 'none';
                // Make prediction panel visible and fade it in
                predictionPanel.classList.remove('hidden-panel');
                setTimeout(() => {
                    predictionPanel.style.opacity = '1';
                }, 50); // A tiny delay to ensure the display property is set before opacity transition
            }, 500); // This duration should match the CSS transition duration
        });


    
        // Tab switching logic
        function changeTab(tabName) {
            const singlePanel = document.getElementById('panel-single');
            const bulkPanel = document.getElementById('panel-bulk');
            const singleTab = document.getElementById('tab-single');
            const bulkTab = document.getElementById('tab-bulk');

            if (tabName === 'single') {
                singlePanel.style.display = 'block';
                bulkPanel.style.display = 'none';
                singleTab.classList.add('text-indigo-600', 'border-indigo-500');
                singleTab.classList.remove('text-gray-500', 'border-transparent', 'hover:text-gray-700', 'hover:border-gray-300');
                bulkTab.classList.add('text-gray-500', 'border-transparent', 'hover:text-gray-700', 'hover:border-gray-300');
                bulkTab.classList.remove('text-indigo-600', 'border-indigo-500');
            } else {
                singlePanel.style.display = 'none';
                bulkPanel.style.display = 'block';
                bulkTab.classList.add('text-indigo-600', 'border-indigo-500');
                bulkTab.classList.remove('text-gray-500', 'border-transparent', 'hover:text-gray-700', 'hover:border-gray-300');
                singleTab.classList.add('text-gray-500', 'border-transparent', 'hover:text-gray-700', 'hover:border-gray-300');
                singleTab.classList.remove('text-indigo-600', 'border-indigo-500');
            }
        }

        // Handle single prediction form submission
        document.getElementById('prediction-form').addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            const data = {};
            formData.forEach((value, key) => {
                data[key] = parseFloat(value);
            });

            const resultContainer = document.getElementById('result-container');
            const resultText = document.getElementById('result-text');

            fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json',},
                body: JSON.stringify(data),
            })
            .then(response => response.json())
            .then(result => {
                resultContainer.style.display = 'block';
                resultText.classList.remove('bg-red-100', 'text-red-700', 'bg-green-100', 'text-green-700');
                if (result.prediction == 1) {
                    resultText.textContent = 'Result: Heart Disease is Likely';
                    resultText.classList.add('bg-red-100', 'text-red-700');
                } else {
                    resultText.textContent = 'Result: Heart Disease is Unlikely';
                    resultText.classList.add('bg-green-100', 'text-green-700');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                resultContainer.style.display = 'block';
                resultText.textContent = 'An error occurred. Please try again.';
                resultText.classList.add('bg-yellow-100', 'text-yellow-700');
            });
        });

        // Display error if model is not loaded
        const modelLoaded = {{ 'true' if model else 'false' }};
        if (!modelLoaded) {
            document.getElementById('error-message').innerText = "Model file 'random_forest_heart_model.pkl' not found. Please make sure it's in the correct directory and restart the app.";
            // Disable forms
            document.querySelectorAll('input, select, button').forEach(el => el.disabled = true);
        }
    </script>
</body>
</html>
"""

# 5. Define the routes for the web application 
@app.route('/')
def home():
    """Renders the main page."""
    return render_template_string(HTML_TEMPLATE, model=model)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles single prediction requests."""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        data = request.get_json(force=True)
        features = [
            data['age'], data['sex'], data['cp'], data['trestbps'],
            data['chol'], data['fbs'], data['restecg'], data['thalach'],
            data['exang'], data['oldpeak'], data['slope'], data['ca'], data['thal']
        ]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/bulk_predict', methods=['POST'])
def bulk_predict():
    """Handles bulk prediction from a CSV file."""
    if not model:
        return "Error: Model not loaded.", 500
    if 'bulk_file' not in request.files:
        return "No file part", 400
    file = request.files['bulk_file']
    if file.filename == '':
        return "No selected file", 400
    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            required_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            if not all(col in df.columns for col in required_columns):
                return f"CSV must contain the following columns: {', '.join(required_columns)}", 400
            predictions = model.predict(df[required_columns])
            df['prediction'] = predictions
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            return Response(
                output,
                mimetype="text/csv",
                headers={"Content-Disposition": "attachment;filename=predictions.csv"}
            )
        except Exception as e:
            return f"An error occurred: {str(e)}", 500
    return "Invalid file type. Please upload a CSV.", 400

# 6. Run the app
if __name__ == "__main__":
    app.run(debug=True)