# EMPLOYEE RETENTION PREDICTIVE MODEL

## Overview  
This project provides a web-based application to predict employee turnover and the likelihood of an employee leaving the company based on historical data. Users can input employee details manually for a **single prediction** or upload a **CSV file** to process multiple employee records at once. The goal is to help companies optimize resource planning and improve employee retention through predictive analytics.  

## Features  
- **Single Employee Prediction:** Users can manually enter employee details to receive a turnover prediction.  
- **Multiple Employee Prediction via CSV:** Users can upload CSV files containing multiple employee records for batch processing.  
- **User-Friendly Interface:** A web-based UI built using Flask and Jinja for seamless interaction.  
- **Data-Driven Insights:** The application uses machine learning model to provide turnover probability for employees.  


## Setup and Installation  

### **Step 1: Clone the Repository**  
```bash
git clone http://gitlab.pccube.com:8081/gitlab/codingcamp/aicodingcamp/examples/ai-camp-2/employee-retention-predictive-model.git 
```
```bash
cd employee-retention-predictive-model
```

### **Step 2: Create a Virtual Environment (Optional but Recommended)**

To create a virtual environment using Conda, run the following command:

```bash
conda create --name employee_retention_app python=3.12 -y
conda activate employee_retention_app
```


## Step 3: Activate the virtual Environment & Install Dependencies

Once inside the Conda environment, install the required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Step 4: Run the Flask Application

To start the Flask application, run the following command:

```bash
python application.py
```

The application will be available at:

```
http://0.0.0.0:5000/
```

## Explanation of the Structure:

1. **Root Files**:
   - `application.py`: The main Flask application file.
   - `README.md`: Documentation for the project.
   - `requirements.txt`: Lists the Python dependencies for the project.
   - `xgboost_scratch.ipynb`: A Jupyter notebook with the results using `PyTorchGradientBoosting`.

2. **`artifacts` Directory**:
   - Contains serialized files like pre-trained models (`xgboost.pkl`, `xgboost.pth`) and one-hot encoded feature names (`ohe_feature_names.pkl`).

3. **`src` Directory**:
   - Contains the source code for the project, including:
     - `components`: Data processing for the project.
     - `pipeline`: Data processing and prediction pipelines.
     - `exception.py`: Custom exception handling.
     - `logger.py`: Logging configuration.
     - `utils.py`: Utility functions.

4. **`static` Directory**:
   - Contains static assets like images (`img`) and CSS files (`styles.css`).

5. **`templates` Directory**:
   - Contains HTML templates for the Flask application:
     - `base.html`: Base template for all pages.
     - `index.html`: Home page.
     - `multiple_employees.html`: Page for multiple employee predictions.
     - `single_employee.html`: Page for single employee predictions.

7. **`tests` Directory**:
   - Contains test files for the application:
     - `test_predict_multiple.py`: Tests for multiple employee predictions.
     - `test_predict_single.py`: Tests for single employee predictions.
