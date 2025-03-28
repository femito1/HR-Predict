import unittest
from application import app
import io
class TestPredictMultipleEndpoint(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_valid_csv_upload(self):
        data = {
            "csv_file": (io.BytesIO(b"satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department,salary\n0.75,0.8,5,200,3,0,0,sales,low"), "test.csv")
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Prediction Results", response.data)

    def test_multiple_valid_csv_upload(self):
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
        self.assertIn(b"Maximum of 5 files allowed", response.data)

    def test_non_csv_file(self):
        data = {
            "csv_file": (io.BytesIO(b"This is not a CSV file"), "test.txt")
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"is not a CSV", response.data)

    def test_empty_file(self):
        data = {
            "csv_file": (io.BytesIO(b""), "empty.csv")
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Error reading", response.data)

    def test_missing_columns(self):
        data = {
            "csv_file": (io.BytesIO(b"last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department, salary\n,0.8,5,200,3,0,0,sales,low"), "missing_columns.csv")
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Missing mandatory columns", response.data)

    def test_invalid_csv_content(self):
        data = {
            "csv_file": (io.BytesIO(b"satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department,salary\ninvalid,0.8,5,200,3,0,0,sales,low"), "invalid_content.csv")
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Validation failed", response.data)

    def test_large_csv_file(self):
        large_data = b"satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department,salary\n" + b"0.75,0.8,5,200,3,0,0,sales,low\n" * 1000000
        data = {
            "csv_file": (io.BytesIO(large_data), "large_file.csv")
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"exceeds 10MB limit", response.data)

    def test_missing_optional_columns(self):
        data = {
            "csv_file": (io.BytesIO(b"satisfaction_level,last_evaluation,number_project,time_spend_company\n0.75,0.8,5,3"), "missing_optional.csv")
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Added missing column", response.data)

    def test_missing_values_in_optional_columns(self):
        csv_data = b"satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department,salary\n" + \
                   b"0.75,0.8,5,,3,,0,sales,\n" + \
                   b"0.65,0.7,4,180,2,1,,,medium"
        data = {
            "csv_file": (io.BytesIO(csv_data), "missing_values.csv")
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Filled", response.data)
        self.assertIn(b"default value", response.data)

    def test_invalid_department_values(self):
        data = {
            "csv_file": (io.BytesIO(b"satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department,salary\n0.75,0.8,5,200,3,0,0,invalid_dept,low"), "invalid_dept.csv")
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Validation failed", response.data)

    def test_invalid_salary_values(self):
        data = {
            "csv_file": (io.BytesIO(b"satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department,salary\n0.75,0.8,5,200,3,0,0,sales,invalid_salary"), "invalid_salary.csv")
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Validation failed", response.data)

    def test_out_of_range_values(self):
        data = {
            "csv_file": (io.BytesIO(b"satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department,salary\n1.5,0.8,25,400,60,2,2,sales,low"), "out_of_range.csv")
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Validation failed", response.data)

    def test_valid_csv_with_warnings(self):
        csv_data = b"satisfaction_level,last_evaluation,number_project,time_spend_company,department\n" + \
                   b"0.75,0.8,5,3,sales\n" + \
                   b"0.65,0.7,4,2,"
        data = {
            "csv_file": (io.BytesIO(csv_data), "warnings.csv")
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Added missing column", response.data)
        self.assertIn(b"default value", response.data)

    def test_csv_with_id_column(self):
        data = {
            "csv_file": (io.BytesIO(b"id,satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department,salary\n1,0.75,0.8,5,200,3,0,0,sales,low"), "with_id.csv")
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Prediction Results", response.data)

    def test_csv_with_invalid_id_column(self):
        data = {
            "csv_file": (io.BytesIO(b"id,satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,department,salary\ninvalid,0.75,0.8,5,200,3,0,0,sales,low"), "invalid_id.csv")
        }
        response = self.app.post('/predict_from_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Invalid ID values", response.data)

if __name__ == '__main__':
    unittest.main()