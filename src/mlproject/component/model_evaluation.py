import joblib
import os
from sklearn.metrics import (r2_score, mean_absolute_error, 
                             root_mean_squared_error)

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from src.mlproject.entity.config_entity import ModelEvaluationConfig
from src.mlproject import logging
from box.exceptions import BoxValueError
from datetime import datetime

EXP_NAME = "ds-salary-2023-prediction"

class ModelEvaluation():
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate_models(self) -> list:
        try: 
            test_arr = joblib.load(self.config.test_data_path)
            X_test = test_arr[:, :-1]
            y_test = test_arr[:,-1]
            results = joblib.load(self.config.grid_result_path)
            # update grid result with 3 parameters predicted on test dataset
            for name, value in results.items():
            
                model = value['model']
                y_pred = model.predict(X_test)
                mae = int(round(mean_absolute_error(y_test, y_pred), 0))
                rmse = int(round(root_mean_squared_error(y_test, y_pred), 0))
                test_r2 = round(r2_score(y_test, y_pred), 4)
                test_scores = {
                    'mae' : mae,
                    'rmse' : rmse,
                    'test_r2': test_r2}
                results[name]['test_scores'] = test_scores
            # sort models based on test MAE score
            sorted_results = sorted(results.items(), 
                                  key=lambda x: x[1]['test_scores']['mae'], 
                                  reverse=False)
            # store results metric
            result_metrics = dict(sorted_results)
            eval_config = os.path.join(self.config.root_dir, self.config.resuslt_metrics)
            joblib.dump(result_metrics, eval_config)
            
            return sorted_results
        
        except BoxValueError as e:
             logging.error(f"Error: {e}")

    def find_prod_model(self):
        try:
            top_model = self.evaluate_models()[0]
            # save best model for deploymnet
            champ_model = top_model[1]['model']
            scores = top_model[1]['test_scores']
            champ_config = os.path.join(self.config.root_dir, self.config.champ_model)
            joblib.dump(champ_model, champ_config)
            logging.info(f"Champion Model: {top_model[0]} saved to {self.config.root_dir}")
            logging.info(f"Champion model mae:{scores['mae']}, \
                         rmse: {scores['rmse']}, r2: {scores['test_r2']:.3f}")
        
        except BoxValueError as e:
            logging.error("Error: {e}")

    def run_mlflow(self, top_n: int = 3, version: str = "v1"):
        try:
            logging.info("MLflow experiment starts")
            top_n_models = self.evaluate_models()[:top_n]
            
            mlflow.set_experiment("ds-salary-2023-prediction")
            mlflow.set_tracking_uri(uri=self.config.mlflow_uri)

            for element in top_n_models:
                name = f"{element[0]}_{version}"
                model_obj = element[1]['model']
                test_arr = joblib.load(self.config.test_data_path)
                X_test = test_arr[:, :-1]


                # Create an input example (small subset)
                input_example = X_test[:5]
                signature = infer_signature(X_test, model_obj.predict(X_test))

                with mlflow.start_run(run_name=name):
                    mlflow.log_params(element[1]['best_params'])
                    mlflow.log_metric('mae', element[1]['test_scores']['mae'])
                    mlflow.log_metric('rmse', element[1]['test_scores']['rmse'])
                    mlflow.log_metric('r2_score', element[1]['test_scores']['test_r2'])

                    if 'xgb' in element[0].lower():
                        mlflow.xgboost.log_model(
                            model_obj,
                            name,
                            signature=signature,
                            input_example=input_example)
                    else:
                        mlflow.sklearn.log_model(
                            model_obj,
                            name,
                            signature=signature,
                            input_example=input_example)

        except Exception as e:
            logging.exception(f"MLflow run failed: {e}")

    def register_best_model(self, model_name: str):
        client = MlflowClient()
        experiment = client.get_experiment_by_name("ds-salary-2023-prediction")

        # --- 1. Get the best run by MAE ---
        runs = client.search_runs(
            [experiment.experiment_id],
            order_by=["metrics.mae ASC"],
            max_results=1
        )
        best_run = runs[0]
        model_uri = f"runs:/{best_run.info.run_id}/xgb_v8"  

        # --- 2. Register the model ---
        registered_model = mlflow.register_model(model_uri, model_name)

        # --- 3. Use aliases instead of deprecated stages ---
        # Alias "production" replaces the old "Production" stage
        try:
            # Remove the old alias if it exists, so only one "production" model is active
            existing_version = client.get_model_version_by_alias(model_name, "production")
            if existing_version:
                client.delete_registered_model_alias(model_name, "production")
        except Exception:
            # It's fine if alias doesn't exist yet
            pass

        # Assign alias to the new model version
        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=registered_model.version
        )

        # --- 4. Log metrics & tags cleanly ---
        metric_history = client.get_metric_history(best_run.info.run_id, "mae")
        mae_value = float(metric_history[-1].value) if metric_history else None

        tags = {
            "run_id": best_run.info.run_id,
            "registered_by": "evaluation_pipeline",
            "timestamp": str(datetime.now())
        }

        if mae_value is not None:
            tags["mae"] = str(mae_value)

        for key, value in tags.items():
            client.set_model_version_tag(model_name, registered_model.version, key, value)

        logging.info(f"Model '{model_name}' version {registered_model.version} registered and aliased as 'production'.")

