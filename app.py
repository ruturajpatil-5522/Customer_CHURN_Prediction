from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__, static_url_path='/static')

# Load the pre-trained model and transformer
model = joblib.load('model.pkl')
trans = joblib.load('transformer.pkl')

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index')
def home2():
    return render_template('index.html')

@app.route('/form')
def show_form():
    return render_template('form.html')

@app.route('/predictorexcel')
def predictorexcel():
    return render_template('predictorexcel.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    contract = request.form['contract']
    onlinesecurity = request.form['onlinesecurity']
    techsupport = request.form['techsupport']
    internetservice = request.form['internetservice']
    onlinebackup = request.form['onlinebackup']
    tenure = int(request.form['tenure'])
    monthlycharges = float(request.form['monthlycharges'])
    totalcharges = float(request.form['totalcharges'])

    # Create a dictionary from the form data
    cust = {'contract': contract,
            'onlinesecurity': onlinesecurity,
            'techsupport': techsupport,
            'internetservice': internetservice,
            'onlinebackup': onlinebackup,
            'tenure': tenure,
            'monthlycharges': monthlycharges,
            'totalcharges': totalcharges}

    # Create a DataFrame from the dictionary
    cust_df = pd.DataFrame(cust, index=[0])

    # Transform the DataFrame using the transformer
    cust_transformed = trans.transform(cust_df)

    # Predict churn using the model
    prediction = model.predict(cust_transformed)[0]

    # Determine the result message
    result_message = 'The customer is predicted to churn from the company.' if prediction == 1 else 'The customer is predicted to stay with the company.'
    return render_template('form.html', pred=result_message)

@app.route('/predict_excel', methods=['POST'])
def predict_excel():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)
        
        data = pd.read_csv(filepath)
        transformed_data = trans.transform(data)
        predictions = model.predict(transformed_data)
        
        # Map predictions to 'Churn' and 'No-churn'
        data['Churn'] = predictions
        data['Churn'] = data['Churn'].map({0: 'No-churn', 1: 'Churn'})
        
        result_filepath = os.path.join('uploads', 'predictions_' + file.filename)
        data.to_csv(result_filepath, index=False)
        
        return send_file(result_filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
