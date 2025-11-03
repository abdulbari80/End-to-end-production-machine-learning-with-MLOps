from src.mlproject import logging
from src.mlproject.config.configuration import ConfigurationManager
from src.mlproject.component.model_trainer import ModelTrainer

class ModelTrainingPipeline:
    def __init__(self):
        pass
    def train_models(self):
        config = ConfigurationManager().get_model_training_config()
        model_train_obj = ModelTrainer(config=config)
        model_train_obj.tune_hyperpara_select_model()

if __name__ == '__main__':
    "This is a model training pipeline"
