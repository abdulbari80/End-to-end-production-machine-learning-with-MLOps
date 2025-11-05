import pandas as pd
import joblib
from pathlib import Path
import mlflow

class Prediction:
    def __init__(
        self,
        experience_level: str,
        employment_type: str,
        remote_ratio: str,
        company_size: str,
        job_title_freq: str,
        employee_residence_top: str,
        company_location_top: str,
        data_pipeline=None,
        model=None):

        self.experience_level = experience_level
        self.employment_type = employment_type
        self.remote_ratio = remote_ratio
        self.company_size = company_size
        self.job_title_freq = job_title_freq
        self.employee_residence_top = employee_residence_top
        self.company_location_top = company_location_top
        self.data_pipeline = data_pipeline
        self.model = model

    def get_prediction(self):
        # user input dictionary
        fields = {
            'experience_level': self.experience_level,
            'employment_type': self.employment_type,
            'remote_ratio': self.remote_ratio,
            'company_size': self.company_size,
            'job_title_freq': self.job_title_freq,
            'employee_residence_top': self.employee_residence_top,
            'company_location_top': self.company_location_top
        }

        if not all(fields.values()) or any(v.strip() == "" for v in fields.values()):
            raise ValueError("Missing or invalid input fields for prediction.")

        # --- Build DataFrame ---
        df_user_input = pd.DataFrame([fields])

        # --- Load model & transformer if not provided ---
        if self.data_pipeline is None or self.model is None:
            try:
                data_process_pipe_obj = joblib.load("artifacts/data_transformation/data_trans_obj_v1.3.joblib")
                # prod_model = joblib.load("artifacts/model_evaluation/model_v1.3.joblib")
                model_name = "XGB_Regressor"
                model_version = 4  # Example
                prod_model = mlflow.pyfunc.load_model(
                    model_uri=f"models:/{model_name}/{model_version}")
            except Exception as e:
                raise RuntimeError(f"Error loading model artifacts: {e}")
        else:
            data_process_pipe_obj = self.data_pipeline
            prod_model = self.model

        # --- Transform & predict ---
        user_input_arr = data_process_pipe_obj.transform(df_user_input)
        pred_result = prod_model.predict(user_input_arr)[0]

        return float(pred_result)


if __name__ == "__main__":
    print("Prediction module is ready for Flask app integration.")
    # pred_obj = Prediction(
    #     experience_level="Senior",
    #     employment_type="Full-time",
    #     remote_ratio="On-site",
    #     company_size="Large",
    #     job_title_freq ="Data Scientist",
    #     employee_residence_top="United States",
    #     company_location_top="United States"
    # )
    # result = pred_obj.get_prediction()
    # print(f"Predicted Salary: {result}")
