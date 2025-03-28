from flask import Flask, request, render_template, send_file, send_from_directory
import numpy as np
import pandas as pd
import io
import zipfile
import sys
import traceback
from werkzeug.utils import secure_filename

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

from src.exception import CustomException
from src.logger import logging

# Temp storage
csv_storage = {}

application = Flask(__name__)

app = application

VALID_DEPARTMENTS = [
    "sales",
    "accounting",
    "hr",
    "technical",
    "support",
    "management",
    "IT",
    "product_mng",
    "marketing",
    "RandD",
]
VALID_SALARIES = ["low", "medium", "high"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

"""Validation of CSV data with detailed error/warning reporting"""
def validate_csv_file(df, filename):
    errors = []
    warnings = []
    
    # Mandatory columns
    mandatory_columns = [
        "satisfaction_level",
        "last_evaluation",
        "number_project",
        "time_spend_company"
    ]
    
    # Optional columns with default values if not provided
    optional_columns = {
        "average_montly_hours": 200,
        "Work_accident": 0,
        "promotion_last_5years": 0,
        "department": "sales",
        "salary": "low"
    }

    missing_mandatory = [col for col in mandatory_columns if col not in df.columns]
    if missing_mandatory:
        errors.append(f"Missing mandatory columns: {', '.join(missing_mandatory)}")
        return errors, warnings, None  

    if 'id' not in df.columns:
        df.insert(0, 'id', range(1, len(df) + 1))
    else:
        df['id'] = pd.to_numeric(df['id'], errors='coerce')
        invalid_ids = df['id'].isna()
        if invalid_ids.any():
            bad_rows = df[invalid_ids]['id'].index.tolist()
            errors.append(f"Invalid ID values in rows: {', '.join(map(str, bad_rows))}")
            df = df[~invalid_ids].copy()
        if df.empty:
            errors.append("No valid rows remaining after ID validation")
            return errors, warnings, None

    validation_issues = {}

    numerical_fields = {
        "satisfaction_level": (0, 1),
        "last_evaluation": (0, 1),
        "number_project": (0, 20),
        "average_montly_hours": (80, 320),
        "time_spend_company": (0, 50),
        "Work_accident": [0, 1],
        "promotion_last_5years": [0, 1]
    }

    for col, default_val in optional_columns.items():
        if col not in df.columns:
            df[col] = default_val
            warnings.append(f"Added missing column '{col}' with default value {default_val}")

    for col in optional_columns:
        if col in df.columns and df[col].isna().any():
            default_val = optional_columns[col]
            na_count = df[col].isna().sum()
            df[col].fillna(default_val, inplace=True)
            warnings.append(f"Filled {na_count} missing values in '{col}' with default value {default_val}")

    if 'department' in df.columns:
        df['department'] = df['department'].astype(str).str.strip().str.lower()
        invalid_dept = ~df['department'].isin([d.lower() for d in VALID_DEPARTMENTS])
        if invalid_dept.any():
            bad_rows = df[invalid_dept]['id'].tolist()
            for row_id in bad_rows:
                if row_id not in validation_issues:
                    validation_issues[row_id] = []
                invalid_value = df.loc[df['id'] == row_id, 'department'].values[0] if not df[df['id'] == row_id].empty else "unknown"
                validation_issues[row_id].append(
                    f"Invalid department '{invalid_value}' (valid options: {', '.join(VALID_DEPARTMENTS)})"
                )

    if 'salary' in df.columns:
        df['salary'] = df['salary'].astype(str).str.strip().str.lower()
        invalid_salary = ~df['salary'].isin([s.lower() for s in VALID_SALARIES])
        if invalid_salary.any():
            bad_rows = df[invalid_salary]['id'].tolist()
            for row_id in bad_rows:
                if row_id not in validation_issues:
                    validation_issues[row_id] = []
                invalid_value = df.loc[df['id'] == row_id, 'salary'].values[0] if not df[df['id'] == row_id].empty else "unknown"
                validation_issues[row_id].append(
                    f"Invalid salary '{invalid_value}' (valid options: {', '.join(VALID_SALARIES)})"
                )

    for field, valid_range in numerical_fields.items():
        if field in df.columns:
            original_values = df[field]
            df[field] = pd.to_numeric(df[field], errors='coerce')
            non_numeric = df[field].isna() & original_values.notna()
            
            if non_numeric.any():
                bad_rows = df[non_numeric]['id'].tolist()
                for row_id in bad_rows:
                    if row_id not in validation_issues:
                        validation_issues[row_id] = []
                    invalid_value = original_values.iloc[row_id-1] if row_id-1 < len(original_values) else "unknown"
                    validation_issues[row_id].append(
                        f"Non-numeric value '{invalid_value}' in column '{field}'"
                    )

            if isinstance(valid_range, tuple):
                min_val, max_val = valid_range
                out_of_range = (df[field] < min_val) | (df[field] > max_val)
                if out_of_range.any():
                    bad_rows = df[out_of_range]['id'].tolist()
                    for row_id in bad_rows:
                        if row_id not in validation_issues:
                            validation_issues[row_id] = []
                        invalid_value = df.loc[df['id'] == row_id, field].values[0] if not df[df['id'] == row_id].empty else "unknown"
                        validation_issues[row_id].append(
                            f"Value {invalid_value} out of range ({min_val}-{max_val}) in column '{field}'"
                        )
            else: 
                invalid = ~df[field].isin(valid_range)
                if invalid.any():
                    bad_rows = df[invalid]['id'].tolist()
                    for row_id in bad_rows:
                        if row_id not in validation_issues:
                            validation_issues[row_id] = []
                        invalid_value = df.loc[df['id'] == row_id, field].values[0] if not df[df['id'] == row_id].empty else "unknown"
                        validation_issues[row_id].append(
                            f"Invalid value {invalid_value} in column '{field}' (must be one of: {', '.join(map(str, valid_range))})"
                        )

    valid_rows = ~df['id'].isin(validation_issues.keys())
    df = df[valid_rows].copy()

    if validation_issues:
        warning_messages = [f"{len(validation_issues)} rows had invalid data and were removed:"]
        for row_id, issues in list(validation_issues.items())[:10]:  
            warning_messages.append(f"Row ID {row_id}: {', '.join(issues)}")
        if len(validation_issues) > 10:
            warning_messages.append(f"...and {len(validation_issues)-10} more rows with issues")
        warnings.extend(warning_messages)

    required_columns = ['id'] + mandatory_columns + list(optional_columns.keys())
    df = df[[col for col in required_columns if col in df.columns]]

    if df.empty:
        errors.append("No valid rows remaining after all validations")

    return errors, warnings, df

@app.route("/")
def index():
    return render_template("index.html")

"""Predicting the probability only on single employee"""
@app.route("/predict-single", methods=["POST"])
def predict_datapoint():
    errors = {}
    warnings = {}
    form_data = request.form
    form_data_dict = {key: value for key, value in form_data.items()}

    # Mandatory fields
    mandatory_fields = [
        "satisfaction_level",
        "last_evaluation",
        "number_project",
        "time_spend_company"
    ]

    for field in mandatory_fields:
        if field not in form_data_dict or not form_data_dict[field]:
            errors[field] = f"{field.replace('_', ' ').title()} is required."

    if errors:
        return (
            render_template("single_employee.html", errors=errors, form_data=form_data_dict),
            400,
        )

    # Optional fields with default values if not provided
    optional_fields = {
        "average_montly_hours": "200",
        "Work_accident": "0",
        "promotion_last_5years": "0",
        "department": "sales",
        "salary": "low"
    }

    for field, default_value in optional_fields.items():
        if field not in form_data_dict or not form_data_dict[field]:
            display_value = str(default_value)
            if field in ['Work_accident', 'promotion_last_5years']:
                display_value = "Yes" if default_value == "1" else "No"
            warnings[field] = f"Using default value for {field.replace('_', ' ')}: {display_value}"
            form_data_dict[field] = default_value

    satisfaction_level = form_data_dict.get("satisfaction_level")
    last_evaluation = form_data_dict.get("last_evaluation")
    number_project = form_data_dict.get("number_project")
    average_montly_hours = form_data_dict.get("average_montly_hours")
    time_spend_company = form_data_dict.get("time_spend_company")
    Work_accident = form_data_dict.get("Work_accident")
    promotion_last_5years = form_data_dict.get("promotion_last_5years")
    department = form_data_dict.get("department")
    salary = form_data_dict.get("salary")

    try:
        satisfaction_level = float(satisfaction_level)
        if satisfaction_level < 0 or satisfaction_level > 1:
            errors["satisfaction_level"] = "Satisfaction level must be between 0 and 1."
        else:
            scaled_value = satisfaction_level * 100
            if not abs(scaled_value - round(scaled_value)) < 1e-6:
                errors["satisfaction_level"] = "Satisfaction level must be in increments of 0.01."
    except (ValueError, TypeError):
        errors["satisfaction_level"] = "Satisfaction level must be a number between 0 and 1."

    try:
        last_evaluation = float(last_evaluation)
        if last_evaluation < 0 or last_evaluation > 1:
            errors["last_evaluation"] = "Last evaluation score must be between 0 and 1."
        else:
            scaled_value = last_evaluation * 100
            if not abs(scaled_value - round(scaled_value)) < 1e-6:
                errors["last_evaluation"] = "Last evaluation score must be in increments of 0.01."
    except (ValueError, TypeError):
        errors["last_evaluation"] = "Last evaluation score must be a number between 0 and 1."

    try:
        number_project = int(number_project)
        if number_project < 0:
            errors["number_project"] = "Number of projects must be a positive integer."
        if number_project > 20:
            errors["number_project"] = "Number of projects must be between 0 and 20."
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
        if time_spend_company > 50:
            errors["time_spend_company"] = "Years spent in company must be between 0 and 50."
    except (ValueError, TypeError):
        errors["time_spend_company"] = "Years spent in company must be an integer."

    if promotion_last_5years not in ["0", "1"]:
        errors["promotion_last_5years"] = "Invalid promotion_last_5years value."

    if Work_accident not in ["0", "1"]:
        errors["Work_accident"] = "Invalid Work_accident value."

    if department not in VALID_DEPARTMENTS:
        errors["department"] = "Invalid department selected."

    if salary not in VALID_SALARIES:
        errors["salary"] = "Invalid salary selected."

    if errors:
        return (
            render_template("single_employee.html", errors=errors, form_data=form_data_dict),
            400,
        )

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
    suggestions = predict_pipeline.suggest_improvements(pred_df)

    return render_template(
        "single_employee.html",
        results=results,
        form_data=form_data_dict,
        suggestions=suggestions,
        warnings=warnings
    )

"""Predicting the retention rate from multiple CSV files"""
@app.route("/predict_from_csv", methods=["POST"])
def upload_csv():
    try:
        if request.method == "POST":
            uploaded_files = request.files.getlist("csv_file")
            if not uploaded_files or all(file.filename == "" for file in uploaded_files):
                return render_template("multiple_employees.html", error="No file selected."), 400

            if len(uploaded_files) > 5:
                return render_template("multiple_employees.html", error="Maximum of 5 files allowed."), 400

            dataframes = []
            filenames = []
            all_warnings = {}

            for file in uploaded_files:
                if not file or file.filename == "":
                    continue

                if not file.filename.endswith(".csv"):
                    return render_template("multiple_employees.html", 
                                        error=f"File '{file.filename}' is not a CSV."), 400

                file.seek(0, 2)
                file_size = file.tell()
                file.seek(0)
                if file_size > MAX_FILE_SIZE:
                    return render_template("multiple_employees.html",
                                        error=f"File '{file.filename}' exceeds 10MB limit."), 400

                try:
                    try:
                        df = pd.read_csv(file)
                    except Exception as e:
                        return render_template("multiple_employees.html",
                                            error=f"Error reading '{file.filename}': {str(e)}"), 400

                    df.columns = df.columns.str.strip()

                    errors, warnings, validated_df = validate_csv_file(df, file.filename)
                    
                    if errors:
                        error_message = f"Validation failed for '{file.filename}':\n" + "\n".join(errors)
                        return render_template("multiple_employees.html", error=error_message), 400

                    if warnings:
                        all_warnings[file.filename] = warnings

                    if validated_df is not None and not validated_df.empty:
                        dataframes.append(validated_df)
                        filenames.append(file.filename)

                except Exception as e:
                    return render_template("multiple_employees.html",
                                        error=f"Error processing '{file.filename}': {str(e)}"), 400

            if not dataframes:
                return render_template("multiple_employees.html",
                                    error="No valid data found in any uploaded files."), 400

            try:
                predict_pipeline = PredictPipeline()
                predictions_list, turnover_rates = predict_pipeline.predict_from_csv(dataframes)
                suggestions_list = predict_pipeline.suggest_improvements_batch(dataframes)

                turnover_rates_dict = {
                    filename: {"turnover_rate": rate, "suggestions": suggestions}
                    for filename, rate, suggestions in zip(filenames, turnover_rates, suggestions_list)
                }

                predictions = {}
                for i, filename in enumerate(filenames):
                    if "probability" not in predictions_list[i].columns:
                        raise ValueError(f"Missing probability column in predictions for {filename}")
                    predictions[filename] = predictions_list[i].to_dict("records")
                    csv_storage[filename] = predictions_list[i].to_csv(index=False)

                return render_template(
                    "multiple_employees.html",
                    filenames=filenames,
                    turnover_rates=turnover_rates_dict,
                    predictions=predictions,
                    warnings=all_warnings
                )

            except Exception as e:
                return render_template("multiple_employees.html",
                                    error=f"Prediction error: {str(e)}"), 400

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return render_template("multiple_employees.html",
                            error=f"An unexpected error occurred: {str(e)}"), 400
    
"""Function to download CSV file with probability"""
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
            error="An error occurred while downloading the file.",
        )

"""Downloading/providing the default template for users"""
@app.route("/download_template")
def download_template():
    sample_data = {
        "satisfaction_level": [0.72, 0.65],
        "last_evaluation": [0.85, 0.70],
        "number_project": [5, 3],
        "average_montly_hours": [180, 160],
        "time_spend_company": [3, 2],
        "Work_accident": [0, 0],
        "promotion_last_5years": [0, 0],
        "department": ["sales", "IT"],
        "salary": ["medium", "low"]
    }
    df = pd.DataFrame(sample_data)
    
    output = io.BytesIO()
    output.write(df.to_csv(index=False).encode('utf-8'))
    output.seek(0)
    
    return send_file(
        output, 
        as_attachment=True, 
        download_name="employee_template.csv",
        mimetype="text/csv"
    )

"""Downloading the User Guide"""
@app.route('/pdf/<filename>')
def serve_pdf(filename):
    return send_from_directory('static/pdfs', filename)

@app.route("/single-employee")
def single_employee():
    return render_template("single_employee.html")


@app.route("/multiple-employees")
def multiple_employees():
    return render_template("multiple_employees.html")


@app.route('/user-guide')
def user_guide():
    return render_template('user_guide.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)