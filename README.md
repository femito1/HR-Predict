# US01: EMPLOYEE RETENTION PREDICTIVE MODEL

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