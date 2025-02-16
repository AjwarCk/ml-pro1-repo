import os  # Module for interacting with the operating system (e.g., file paths)
import sys  # Module providing access to system-specific parameters and functions
from src.exception import CustomException  # Custom exception class for handling errors
from src.logger import logging  # Logging instance to log messages for debugging and tracking
import pandas as pd  # Pandas library for data manipulation and analysis

from sklearn.model_selection import train_test_split  # Function to split data into training and test sets
from dataclasses import dataclass  # Decorator to simplify class creation for storing data

# Define a configuration data class for data ingestion settings
@dataclass
class DataIngestionConfig:
    # File path for saving training data
    train_data_path: str = os.path.join('artifacts', "train.csv")
    # File path for saving test data
    test_data_path: str = os.path.join('artifacts', "test.csv")
    # File path for saving the raw data
    raw_data_path: str = os.path.join('artifacts', "data.csv")

# Define the DataIngestion class to handle the ingestion process
class DataIngestion:
    def __init__(self):
        # Initialize the configuration for data ingestion
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        # Log the start of the data ingestion process
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset from a CSV file into a DataFrame
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            # Create the directory for the train data path if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            # Split the dataset into training (80%) and testing (20%) sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training set to a CSV file
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # Save the testing set to a CSV file
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Return the file paths of the training and testing datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # If any exception occurs, raise a custom exception with the error details
            raise CustomException(e, sys)

# When this script is run directly, execute the following code
if __name__ == "__main__":
    # Create an instance of the DataIngestion class
    obj = DataIngestion()
    # Initiate the data ingestion process
    obj.initiate_data_ingestion()
