from flask import Flask, request, render_template, redirect, url_for, flash, send_file
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

VALID_DEPARTMENTS = ["sales", "accounting", "hr", "technical", "support", "management", "IT", "product_mng", "marketing", "RandD"]
VALID_SALARIES = ["low", "medium", "high"]

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict-single", methods=["POST"])
def predict_datapoint():
    errors = {}
    form_data = request.form

    # List of required fields
    required_fields = [
        "satisfaction_level",
        "last_evaluation",
        "number_project",
        "average_montly_hours",
        "time_spend_company",
        "Work_accident",
        "promotion_last_5years",
        "department",
        "salary",
    ]

    # Check for missing fields
    for field in required_fields:
        if field not in form_data or not form_data[field]:
            errors[field] = f"{field.replace('_', ' ').title()} is required."

    if errors:
        return render_template("single_employee.html", errors=errors, form_data=form_data), 400

    # If no missing fields, proceed with validation
    satisfaction_level = form_data.get("satisfaction_level")
    last_evaluation = form_data.get("last_evaluation")
    number_project = form_data.get("number_project")
    average_montly_hours = form_data.get("average_montly_hours")
    time_spend_company = form_data.get("time_spend_company")
    Work_accident = form_data.get("Work_accident")
    promotion_last_5years = form_data.get("promotion_last_5years")
    department = form_data.get("department")
    salary = form_data.get("salary")

    # Validate satisfaction_level
    try:
        satisfaction_level = float(satisfaction_level)
        if satisfaction_level < 0 or satisfaction_level > 1:
            errors["satisfaction_level"] = "Satisfaction level must be between 0 and 1."
        elif not (satisfaction_level * 100).is_integer():  # Check if it's a multiple of 0.01
            errors["satisfaction_level"] = "Satisfaction level must be in increments of 0.01."
    except (ValueError, TypeError):
        errors["satisfaction_level"] = "Satisfaction level must be a number between 0 and 1."

    # Validate last_evaluation
    try:
        last_evaluation = float(last_evaluation)
        if last_evaluation < 0 or last_evaluation > 1:
            errors["last_evaluation"] = "Last evaluation score must be between 0 and 1."
        elif not (last_evaluation * 100).is_integer():  # Check if it's a multiple of 0.01
            errors["last_evaluation"] = "Last evaluation score must be in increments of 0.01."
    except (ValueError, TypeError):
        errors["last_evaluation"] = "Last evaluation score must be a number between 0 and 1."

    # Validate other fields
    try:
        number_project = int(number_project)
        if number_project < 0:
            errors["number_project"] = "Number of projects must be a positive integer."
    except (ValueError, TypeError):
        errors["number_project"] = "Number of projects must be an integer."

    try:
        average_montly_hours = int(average_montly_hours)
        if average_montly_hours < 80 or average_montly_hours > 320:
            errors["average_montly_hours"] = "Average monthly hours must be between 80 and 320."
    except (ValueError, TypeError):
        errors["average_montly_hours"] = "Average monthly hours must be an integer."

    try:
        time_spend_company = int(time_spend_company)
        if time_spend_company < 0:
            errors["time_spend_company"] = "Years spent in company must be a positive integer."
    except (ValueError, TypeError):
        errors["time_spend_company"] = "Years spent in company must be an integer."

    if promotion_last_5years not in ["0", "1"]:
        errors["promotion_last_5years"] = "Invalid promotion_last_5years value."
    elif promotion_last_5years == "1" and time_spend_company < 5:
        errors["promotion_last_5years"] = "Employee cannot be promoted in the last 5 years if they have worked for less than 5 years."

    if Work_accident not in ["0", "1"]:
        errors["Work_accident"] = "Invalid Work_accident value."

    if department not in VALID_DEPARTMENTS:
        errors["department"] = "Invalid department selected."

    if salary not in VALID_SALARIES:
        errors["salary"] = "Invalid salary selected."

    if errors:
        return render_template("single_employee.html", errors=errors, form_data=form_data), 400

    data = CustomData(
        satisfaction_level=satisfaction_level,
        last_evaluation=last_evaluation,
        number_project=number_project,
        average_montly_hours=average_montly_hours,
        time_spend_company=time_spend_company,
        Work_accident=int(Work_accident),
        promotion_last_5years=int(promotion_last_5years),
        department=department,
        salary=salary,
    )

    pred_df = data.get_data_as_dataframe()
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)

    return render_template("single_employee.html", results=results, form_data=form_data)

@app.route("/predict_from_csv", methods=["POST"])
def upload_csv():
    try:
        if request.method == "POST":
            uploaded_files = request.files.getlist("csv_file")
            if len(uploaded_files) > 5:
                return render_template(
                    "multiple_employees.html", error="Maximum of 5 files allowed."
                )

            dataframes = []
            filenames = []

            for file in uploaded_files:
                if not file or file.filename == "":
                    return render_template(
                        "multiple_employees.html", error="No file selected."
                    )

                if not file.filename.endswith(".csv"):
                    return render_template(
                        "multiple_employees.html", error="Only CSV files are allowed."
                    )

                df = pd.read_csv(file)
                dataframes.append(df)
                filenames.append(file.filename)

            predict_pipeline = PredictPipeline()
            predictions_list, turnover_rates = predict_pipeline.predict_from_csv(
                dataframes
            )

            turnover_rates_dict = {
                filename: rate for filename, rate in zip(filenames, turnover_rates)
            }
            predictions = {}

            return render_template(
                "multiple_employees.html",
                filenames={},
                turnover_rates=turnover_rates_dict,
                predictions=predictions,
            )

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        logging.error(traceback.format_exc())
        return render_template(
            "multiple_employees.html",
            error="An error occurred while processing the files. Please check the logs for details.",
        )


@app.route("/download_csv")
def download_csv():
    try:
        filename = request.args.get("filename")
        if not filename:
            return render_template(
                "multiple_employees.html", error="No filename provided."
            )

        if filename not in csv_storage:
            return render_template("multiple_employees.html", error="File not found.")

        output = io.BytesIO()
        output.write(csv_storage[filename].encode("utf-8"))
        output.seek(0)

        return send_file(
            output, as_attachment=True, download_name=f"predicted_{filename}"
        )

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        logging.error(traceback.format_exc())
        return render_template(
            "multiple_employees.html",
            error="An error occurred while downloading the file. Please check the logs for details.",
        )


@app.route("/single-employee")
def single_employee():
    return render_template("single_employee.html")


@app.route("/multiple-employees")
def multiple_employees():
    return render_template("multiple_employees.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)