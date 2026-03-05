import logging
from src.data_ingestion import DataIngestion
from src.data_transformation import DataTransformation
from src.model_trainer import ModelTrainer


class TrainingPipeline:

    def __init__(self, file_path):
        self.file_path = file_path

    def run_pipeline(self):

        logging.info("Pipeline Execution Started")

        # Data Ingestion
        ingestion = DataIngestion(self.file_path)
        df = ingestion.load_data()
        X_train, X_test, y_train, y_test = ingestion.split_data(df)

        # Data Transformation
        transformer = DataTransformation()
        X_train_trans, X_test_trans, y_train_res = transformer.transform(
            X_train, X_test, y_train
        )

        # Model Training
        trainer = ModelTrainer()
        results = trainer.train_and_evaluate(
            X_train_trans, X_test_trans, y_train_res, y_test
        )

        kfold_results = trainer.kfold_validation(X_train_trans, y_train_res)

        logging.info("Pipeline Execution Completed")

        return results, kfold_results