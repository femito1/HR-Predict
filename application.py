from flask import Flask, request,render_template, redirect, url_for, flash, send_file
import numpy as np
import pandas as pd
import io
import zipfile
import sys
import traceback

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

from src.exception import CustomException
from src.logger import logging

# Temp storage
csv_storage = {}

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict-single', methods=['POST'])
def predict_datapoint():
    data=CustomData(
        satisfaction_level = float(request.form.get('satisfaction_level')),
        last_evaluation = float(request.form.get('last_evaluation')),
        number_project = int(request.form.get('number_project')),
        average_montly_hours = int(request.form.get('average_montly_hours')),
        time_spend_company = int(request.form.get('time_spend_company')),
        Work_accident = int(request.form.get('Work_accident')),
        promotion_last_5years = int(request.form.get('promotion_last_5years')),
        department= request.form.get('department'),
        salary = request.form.get('salary')
                    )
    
    # From CustomData class, we call this function on the inputs that we are getting 
    # and converting it all to a Dataframe

    pred_df = data.get_data_as_dataframe()
    print(pred_df)

    # This is initiating the PredictPipeline class and calling its predict() function 
    # on pred_df dataframe of user-input values  
    predict_pipeline = PredictPipeline()

    # This is what is sent to html file for rendering back to user (THE FINAL PREDICTION)
    results = predict_pipeline.predict(pred_df)

    return render_template('single_employee.html', results=results) #maybe results = results[0] in case i am getting a list from model
    # The {results} var name on home.html should be same


@app.route('/predict_from_csv', methods=['POST'])
def upload_csv():
    try:
        if request.method == 'POST':
            uploaded_files = request.files.getlist("csv_file")
            if len(uploaded_files) > 5:
                return render_template('multiple_employees.html', error="Maximum of 5 files allowed.")

            dataframes = []
            filenames = []

            for file in uploaded_files:
                if not file or file.filename == '':
                    return render_template('multiple_employees.html', error="No file selected.")

                if not file.filename.endswith('.csv'):
                    return render_template('multiple_employees.html', error="Only CSV files are allowed.")

                df = pd.read_csv(file)
                dataframes.append(df)
                filenames.append(file.filename)

            predict_pipeline = PredictPipeline()
            predictions_list, turnover_rates = predict_pipeline.predict_from_csv(dataframes)

            turnover_rates_dict = {filename: rate for filename, rate in zip(filenames, turnover_rates)}
            predictions = {}

            for i, filename in enumerate(filenames):
                if 'probability' not in predictions_list[i].columns:
                    raise ValueError(f"Column 'probability' not found in predictions for {filename}")
                predictions[filename] = predictions_list[i].to_dict('records')

                # Store CSV data in temporary storage
                csv_storage[filename] = predictions_list[i].to_csv(index=False)
                logging.info(f"Stored CSV data for file: {filename}")  # Log the filename

            return render_template('multiple_employees.html', filenames=filenames, turnover_rates=turnover_rates_dict, predictions=predictions)

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        logging.error(traceback.format_exc())
        return render_template('multiple_employees.html', error="An error occurred while processing the files. Please check the logs for details.")
    
@app.route('/download_csv')
def download_csv():
    try:
        filename = request.args.get('filename')
        if not filename:
            return render_template('multiple_employees.html', error="No filename provided.")

        if filename not in csv_storage:
            return render_template('multiple_employees.html', error="File not found.")

        output = io.BytesIO()
        output.write(csv_storage[filename].encode('utf-8'))
        output.seek(0)

        return send_file(output, as_attachment=True, download_name=f"predicted_{filename}")

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        logging.error(traceback.format_exc())
        return render_template('multiple_employees.html', error="An error occurred while downloading the file. Please check the logs for details.")

@app.route('/single-employee')
def single_employee():
    return render_template('single_employee.html')

@app.route('/multiple-employees')
def multiple_employees():
    return render_template('multiple_employees.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

# RUN USING python application.py, port = 5000