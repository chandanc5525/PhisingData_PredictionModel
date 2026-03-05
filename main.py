import os
import logging
from src.pipeline import TrainingPipeline

# Create logs folder if not exists
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename="logs/training.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)

if __name__ == "__main__":

    file_path = r"data/raw/data.csv"

    pipeline = TrainingPipeline(file_path)
    pipeline.run_pipeline()