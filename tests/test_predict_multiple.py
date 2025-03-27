import unittest
from application import app
import io
import pandas as pd

class TestPredictMultipleEndpoint(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_valid_csv_upload(self):
        # Test with a valid CSV file
        data = {
            "csv_file": (io.BytesIO(b"satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department,salary\n0.75,0.8,5,200,3,0,0,sales,low"), "test.csv")
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Prediction Results", response.data)

    def test_multiple_valid_csv_upload(self):
        # Test with multiple valid CSV files
        data = {
            "csv_file": [
                (io.BytesIO(b"satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department,salary\n0.75,0.8,5,200,3,0,0,sales,low"), "test1.csv"),
                (io.BytesIO(b"satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department,salary\n0.65,0.7,4,180,2,1,0,hr,medium"), "test2.csv")
            ]
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Prediction Results", response.data)

    def test_too_many_files(self):
        # Test with more than 5 files
        data = {
            "csv_file": [
                (io.BytesIO(b"satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department,salary\n0.75,0.8,5,200,3,0,0,sales,low"), "test1.csv"),
                (io.BytesIO(b"satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department,salary\n0.65,0.7,4,180,2,1,0,hr,medium"), "test2.csv"),
                (io.BytesIO(b"satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department,salary\n0.55,0.6,3,160,1,0,1,technical,high"), "test3.csv"),
                (io.BytesIO(b"satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department,salary\n0.85,0.9,6,220,4,0,0,marketing,low"), "test4.csv"),
                (io.BytesIO(b"satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department,salary\n0.95,1.0,7,240,5,1,1,management,high"), "test5.csv"),
                (io.BytesIO(b"satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department,salary\n0.45,0.5,2,140,0,0,0,support,medium"), "test6.csv")
            ]
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Maximum of 5 files allowed.", response.data)

    def test_non_csv_file(self):
        # Test with a non-CSV file
        data = {
            "csv_file": (io.BytesIO(b"This is not a CSV file"), "test.txt")
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Only CSV files are allowed.", response.data)

    def test_empty_file(self):
        # Test with an empty file
        data = {
            "csv_file": (io.BytesIO(b""), "empty.csv")
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"File empty.csv is empty or improperly formatted.", response.data)

    def test_missing_columns(self):
        # Test with a CSV file missing required columns
        data = {
            "csv_file": (io.BytesIO(b"satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department\n0.75,0.8,5,200,3,0,0,sales"), "missing_columns.csv")
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Missing required columns in file", response.data)

    def test_invalid_csv_content(self):
        # Test with invalid CSV content
        data = {
            "csv_file": (io.BytesIO(b"satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department,salary\ninvalid,0.8,5,200,3,0,0,sales,low"), "invalid_content.csv")
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Non-numerical value found in column \'satisfaction_level\' in file \'invalid_content.csv\'.", response.data)

    def test_large_csv_file(self):
        # Test with a large CSV file
        large_data = b"satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department,salary\n" + b"0.75,0.8,5,200,3,0,0,sales,low\n" * 1000000
        data = {
            "csv_file": (io.BytesIO(large_data), "large_file.csv")
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"File large_file.csv exceeds the maximum allowed size of 10 MB.", response.data)

if __name__ == '__main__':
    unittest.main()