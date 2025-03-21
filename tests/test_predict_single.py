import unittest
from application import app
import json

class TestPredictSingleEndpoint(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_valid_input(self):
        # Test with valid input data
        data = {
            "satisfaction_level": 0.75,
            "last_evaluation": 0.8,
            "number_project": 5,
            "average_montly_hours": 200,
            "time_spend_company": 3,
            "Work_accident": 0,
            "promotion_last_5years": 0,
            "department": "sales",
            "salary": "low"
        }
        response = self.app.post('/predict-single', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Prediction Result", response.data)

    def test_missing_fields(self):
    # Test with missing fields
        data = {
            "satisfaction_level": 0.75,
            "last_evaluation": 0.8,
            "number_project": 5,
            "average_montly_hours": 200,
            "time_spend_company": 3,
            "Work_accident": 0,
            "promotion_last_5years": 0,
        }
        response = self.app.post('/predict-single', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Department is required.", response.data)
        self.assertIn(b"Salary is required.", response.data)

    def test_invalid_department(self):
        # Test with invalid department
        data = {
            "satisfaction_level": 0.75,
            "last_evaluation": 0.8,
            "number_project": 5,
            "average_montly_hours": 200,
            "time_spend_company": 3,
            "Work_accident": 0,
            "promotion_last_5years": 0,
            "department": "invalid_department",
            "salary": "low"
        }
        response = self.app.post('/predict-single', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Invalid department selected", response.data)

    def test_invalid_salary(self):
        # Test with invalid salary
        data = {
            "satisfaction_level": 0.75,
            "last_evaluation": 0.8,
            "number_project": 5,
            "average_montly_hours": 200,
            "time_spend_company": 3,
            "Work_accident": 0,
            "promotion_last_5years": 0,
            "department": "sales",
            "salary": "invalid_salary"
        }
        response = self.app.post('/predict-single', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Invalid salary selected", response.data)

    def test_invalid_satisfaction_level(self):
        # Test with invalid satisfaction level
        data = {
            "satisfaction_level": 1.5,
            "last_evaluation": 0.8,
            "number_project": 5,
            "average_montly_hours": 200,
            "time_spend_company": 3,
            "Work_accident": 0,
            "promotion_last_5years": 0,
            "department": "sales",
            "salary": "low"
        }
        response = self.app.post('/predict-single', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Satisfaction level must be between 0 and 1", response.data)

    def test_invalid_last_evaluation(self):
        # Test with invalid last evaluation
        data = {
            "satisfaction_level": 0.75,
            "last_evaluation": 1.5,
            "number_project": 5,
            "average_montly_hours": 200,
            "time_spend_company": 3,
            "Work_accident": 0,
            "promotion_last_5years": 0,
            "department": "sales",
            "salary": "low"
        }
        response = self.app.post('/predict-single', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Last evaluation score must be between 0 and 1", response.data)

    def test_invalid_number_project(self):
        # Test with invalid number of projects
        data = {
            "satisfaction_level": 0.75,
            "last_evaluation": 0.8,
            "number_project": -5,  
            "average_montly_hours": 200,
            "time_spend_company": 3,
            "Work_accident": 0,
            "promotion_last_5years": 0,
            "department": "sales",
            "salary": "low"
        }
        response = self.app.post('/predict-single', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Number of projects must be a positive integer", response.data)

    def test_invalid_average_monthly_hours(self):
        # Test with invalid average monthly hours
        data = {
            "satisfaction_level": 0.75,
            "last_evaluation": 0.8,
            "number_project": 5,
            "average_montly_hours": 50,  
            "time_spend_company": 3,
            "Work_accident": 0,
            "promotion_last_5years": 0,
            "department": "sales",
            "salary": "low"
        }
        response = self.app.post('/predict-single', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Average monthly hours must be between 80 and 320", response.data)

    def test_invalid_time_spend_company(self):
        # Test with invalid time spent in company
        data = {
            "satisfaction_level": 0.75,
            "last_evaluation": 0.8,
            "number_project": 5,
            "average_montly_hours": 200,
            "time_spend_company": -3,  
            "Work_accident": 0,
            "promotion_last_5years": 0,
            "department": "sales",
            "salary": "low"
        }
        response = self.app.post('/predict-single', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Years spent in company must be a positive integer", response.data)

    def test_invalid_promotion_last_5years(self):
        # Test with invalid promotion_last_5years
        data = {
            "satisfaction_level": 0.75,
            "last_evaluation": 0.8,
            "number_project": 5,
            "average_montly_hours": 200,
            "time_spend_company": 3,
            "Work_accident": 0,
            "promotion_last_5years": 2,
            "department": "sales",
            "salary": "low"
        }
        response = self.app.post('/predict-single', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Invalid promotion_last_5years value", response.data)

    def test_invalid_work_accident(self):
        # Test with invalid work accident
        data = {
            "satisfaction_level": 0.75,
            "last_evaluation": 0.8,
            "number_project": 5,
            "average_montly_hours": 200,
            "time_spend_company": 3,
            "Work_accident": 2, 
            "promotion_last_5years": 0,
            "department": "sales",
            "salary": "low"
        }
        response = self.app.post('/predict-single', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Invalid Work_accident value", response.data)
    
    def test_invalid_increments_satisfaction_level(self):
        # Test with invalid satisfaction_level increments
        data = {
            "satisfaction_level": 0.50001,  
            "last_evaluation": 0.8,
            "number_project": 5,
            "average_montly_hours": 200,
            "time_spend_company": 3,
            "Work_accident": 0,
            "promotion_last_5years": 0,
            "department": "sales",
            "salary": "low"
        }
        response = self.app.post('/predict-single', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Satisfaction level must be in increments of 0.01.", response.data)

    def test_invalid_increments_last_evaluation(self):
        # Test with invalid last_evaluation increments
        data = {
            "satisfaction_level": 0.75,
            "last_evaluation": 0.80001,  
            "number_project": 5,
            "average_montly_hours": 200,
            "time_spend_company": 3,
            "Work_accident": 0,
            "promotion_last_5years": 0,
            "department": "sales",
            "salary": "low"
        }
        response = self.app.post('/predict-single', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Last evaluation score must be in increments of 0.01.", response.data)

    def test_valid_input_boundary_values(self):
        # Test with maximum and minimum valid values
        data = {
            "satisfaction_level": 1.0,  
            "last_evaluation": 1.0,     
            "number_project": 0,       
            "average_montly_hours": 80, 
            "time_spend_company": 0,    
            "Work_accident": 0,
            "promotion_last_5years": 0,
            "department": "sales",
            "salary": "low"
        }
        response = self.app.post('/predict-single', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Prediction Result", response.data)
    
    def test_negative_values(self):
        # Test with negative values in positive fields
        data = {
            "satisfaction_level": 0.75,
            "last_evaluation": 0.8,
            "number_project": -5,
            "average_montly_hours": -200,  
            "time_spend_company": -3,  
            "Work_accident": 0,
            "promotion_last_5years": 0,
            "department": "sales",
            "salary": "low"
        }
        response = self.app.post('/predict-single', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Number of projects must be a positive integer", response.data)
        self.assertIn(b"Average monthly hours must be between 80 and 320", response.data)
        self.assertIn(b"Years spent in company must be a positive integer", response.data)

    def test_non_numeric_inputs(self):
        # Test with non-numeric inputs for numeric fields
        data = {
            "satisfaction_level": "not_a_number",  
            "last_evaluation": "not_a_number",     
            "number_project": "not_a_number",      
            "average_montly_hours": "not_a_number",
            "time_spend_company": "not_a_number", 
            "Work_accident": 0,
            "promotion_last_5years": 0,
            "department": "sales",
            "salary": "low"
        }
        response = self.app.post('/predict-single', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Satisfaction level must be a number between 0 and 1", response.data)
        self.assertIn(b"Last evaluation score must be a number between 0 and 1", response.data)
        self.assertIn(b"Number of projects must be an integer", response.data)
        self.assertIn(b"Average monthly hours must be an integer", response.data)
        self.assertIn(b"Years spent in company must be an integer", response.data)

    def test_invalid_promotion_logic(self):
        # Test with invalid promotion logic
        data = {
            "satisfaction_level": 0.75,
            "last_evaluation": 0.8,
            "number_project": 5,
            "average_montly_hours": 200,
            "time_spend_company": 3,
            "Work_accident": 0,
            "promotion_last_5years": 1,
            "department": "sales",
            "salary": "low"
        }
        response = self.app.post('/predict-single', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Employee cannot be promoted in the last 5 years if they have worked for less than 5 years", response.data)

    def test_empty_inputs(self):
        # Test with empty inputs for required fields
        data = {
            "satisfaction_level": "",
            "last_evaluation": "",
            "number_project": "",
            "average_montly_hours": "",
            "time_spend_company": "",
            "Work_accident": "",
            "promotion_last_5years": "",
            "department": "",
            "salary": ""
        }
        response = self.app.post('/predict-single', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Satisfaction Level is required.", response.data) 
        self.assertIn(b"Last Evaluation is required.", response.data)    
        self.assertIn(b"Number Project is required.", response.data)     
        self.assertIn(b"Average Montly Hours is required.", response.data)  
        self.assertIn(b"Time Spend Company is required.", response.data)  
        self.assertIn(b"Work Accident is required.", response.data)      
        self.assertIn(b"Promotion Last 5Years is required.", response.data)  
        self.assertIn(b"Department is required.", response.data)         
        self.assertIn(b"Salary is required.", response.data)           

    def test_large_input_values(self):
        # Test with excessively large values
        data = {
            "satisfaction_level": 2.0,
            "last_evaluation": 2.0,
            "number_project": 1000,
            "average_montly_hours": 1000,
            "time_spend_company": 1000,
            "Work_accident": 0,
            "promotion_last_5years": 0,
            "department": "sales",
            "salary": "low"
        }
        response = self.app.post('/predict-single', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Satisfaction level must be between 0 and 1", response.data)
        self.assertIn(b"Last evaluation score must be between 0 and 1", response.data)
        self.assertIn(b"Average monthly hours must be between 80 and 320", response.data)

    def test_non_integer_values(self):
        # Test with non-integer values in integer fields
        data = {
            "satisfaction_level": 0.75,
            "last_evaluation": 0.8,
            "number_project": 5.5,
            "average_montly_hours": 200.5,
            "time_spend_company": 3.5,
            "Work_accident": 0,
            "promotion_last_5years": 0,
            "department": "sales",
            "salary": "low"
        }
        response = self.app.post('/predict-single', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Number of projects must be an integer", response.data)
        self.assertIn(b"Average monthly hours must be an integer", response.data)
        self.assertIn(b"Years spent in company must be an integer", response.data)

if __name__ == '__main__':
    unittest.main()