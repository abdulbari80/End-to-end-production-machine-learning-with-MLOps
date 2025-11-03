from src.mlproject.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.mlproject.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.mlproject.pipeline.model_training_pipeline import ModelTrainingPipeline
from src.mlproject.pipeline.model_evaluation_pipeline import ModelEvalPipeline

from src.mlproject import logging

STEP_1 = "Data Ingestion"
STEP_2 = "Data Transformation"
STEP_3 = "Model Training"
STEP_4 = "Model Evaluation"

def main():
    """
    # Triggers data ingestion
    logging.info(f"{STEP_1} starts >>>>>")
    DataIngestionPipeline().ingest_data()
    logging.info(">>>>> f{STEP_1} finished!")
    
    # Triggers data transformation
    logging.info(f"{STEP_2} starts >>>>>")
    data_transform_obj = DataTransformationPipeline()
    data_transform_obj.transform_data()
    logging.info(f">>>>> {STEP_2} finished!")
    
    # Trigger model training
    logging.info(f"{STEP_3} starts >>>>>")
    ModelTrainingPipeline().train_models()
    logging.info(f">>>>> {STEP_3} finished!")
    """
    # Trigger model evaluation
    logging.info(f"{STEP_4} starts >>>>>")
    ModelEvalPipeline().evaluate_model()
    logging.info(f">>>>> {STEP_4} finished!")


if __name__ == '__main__':
    main()