from src.mlproject.config.configuration import ConfigurationManager
from src.mlproject.component.model_evaluation import ModelEvaluation
from src.mlproject import logging

class ModelEvalPipeline:
    def __init__(self):
        pass

    def evaluate_model(self):
        config_obj = ConfigurationManager()
        eval_config_obj = config_obj.get_model_evaluation_config()
        model_eval_obj = ModelEvaluation(config=eval_config_obj)
        # model_eval_obj.evaluate_models()
        # model_eval_obj.find_prod_model()
        # model_eval_obj.run_mlflow(top_n=5, version="v8")
        model_eval_obj.register_best_model(model_name="XGB_Regressor")

if __name__ == "__main__":
    print("This evaluates trained ML models")

        