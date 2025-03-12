from flask import Flask, request,render_template, redirect, url_for, flash, send_file
import numpy as np
import pandas as pd
import io
import zipfile
import sys

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

from src.exception import CustomException
from src.logger import logging



application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_from_csv', methods=['GET', 'POST'])

def upload_csv():

    try:
        if request.method == 'POST':
            uploaded_file = request.files.getlist("csv_file") #We get Filestorage objects that behave as files
            dataframes = []  # Store dataframes, each dataframe contains the contents of respective file, not the actual file
            filenames = []  # Store original filenames

            # if not uploaded_file or uploaded_file.filename == '': # Checking if file exists
            #     return redirect(url_for("predict_from_csv"))

            # if not uploaded_file.filename.endswith('.csv'): # Checking if file is a CSV
            #     return redirect(url_for("predict_from_csv"))

            # df = pd.read_csv(uploaded_file)  # Read the uploaded CSV
            for files in uploaded_file:

                if not files or files.filename == '': # Checking if file exists
                    return redirect(url_for("predict_from_csv"))

                if not files.filename.endswith('.csv'): # Checking if file is a CSV
                    return redirect(url_for("predict_from_csv"))
                
                df = pd.read_csv(files)
                dataframes.append(df)
                filenames.append(files.filename)

            predict_pipeline = PredictPipeline()
            
            predictions = []  # Replace  with `predict_pipeline.predict_from_csv(df)
            predictions_list = predict_pipeline.predict_from_csv(dataframes) #dataframe with added new col, which we later save as csv and return to user

            if len(predictions_list) == 1:
            # If only one file, return a single CSV
                output = io.BytesIO()
                predictions_list[0].to_csv(output, index=False)
                output.seek(0)
                return send_file(output, as_attachment=True, attachment_filename="predicted_results.csv") #Return one single file
                # send_file returns file-like objects ByteIO
                # FRONTEND recieves a downloadable file

            # If multiple files, return a ZIP archive
            # zip_buffer = io.BytesIO()
            # with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            #     for df, filename in zip(predictions_list, filenames):
            #         file_buffer = io.BytesIO()
            #         df.to_csv(file_buffer, index=False)
            #         file_buffer.seek(0) #Moving file pointer to position 0
            #         zip_file.writestr(f"predicted_{filename}", file_buffer.getvalue())

            # zip_buffer.seek(0)
            # return send_file(zip_buffer, as_attachment=True, download_name="predicted_results.zip") #Return zip if user uploads multiple files


    except Exception as e:
        raise CustomException(e,sys)
        return redirect(url_for("predict_from_csv"))

    return render_template('csv_file.html')




@app.route('/predict', methods=['GET', 'POST'])


def predict_datapoint():

    if request.method == 'GET':
        return render_template('home.html')
    
    else: 
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

        return render_template('home.html', results=results) #maybe results = results[0] in case i am getting a list from model
        # The {results} var name on home.html should be same
       

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

# RUN USING python application.py, port = 5000