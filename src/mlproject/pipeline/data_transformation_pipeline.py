from src.mlproject.config.configuration import ConfigurationManager
from src.mlproject.component.data_transformation import DataTransformation
from src.mlproject import logging

class DataTransformationPipeline:
    def __init__(self):
        pass
    def transform_data(self):
        config_obj = ConfigurationManager()
        data_transform_config_obj = config_obj.get_data_transformation_config()
        data_transorm_obj = DataTransformation(config=data_transform_config_obj)
        data_transorm_obj.initiate_data_transformation()
        logging.info("Data transformed and pipeline instance is saved!")

if __name__ == "__main__":
    print("This is a data transormation pipeline.")

