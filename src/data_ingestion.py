import os 
import pandas as pd
from google.cloud import storage 
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
from utils.common_functions import read_yaml

from google.oauth2 import service_account



logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]
        
        os.makedirs(RAW_DIR , exist_ok=True)
        
        logger.info(f"Data Ingestion Started with {self.bucket_name} and the file name is {self.file_name} ")
        
    def download_csv_from_gcp(self):
        try:
            credentials = service_account.Credentials.from_service_account_file("C:\COURSE\Project1\gen-lang-client-0422234397-07a053453f09.json")
            client = storage.Client(credentials=credentials)
        
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.download_to_filename(RAW_FILE_PATH)
        
            logger.info(f"RAW file is successfully downloaded to {RAW_FILE_PATH}")
        
        except Exception as e:
            logger.error(f"Error while downloading the file from GCP: {e}")
            raise CustomException("Failed to download CSV file")
        
    def split_data (self):
        try:
            logger.info("Splitting the data into train and test sets")
            data = pd.read_csv(RAW_FILE_PATH)
            
            train_data , test_data = train_test_split(data , test_size=1-self.train_test_ratio, random_state= 42)
            
            train_data.to_csv(TRAIN_FILE_PATH)
            test_data.to_csv(TEST_FILE_PATH)
            
            logger.info(f"Train data saved to {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved to {TEST_FILE_PATH}")
            
            
        except Exception as e:
            logger.error(f"Error while splitting data")
            raise CustomException ("Faied to split data into training and test sets", e)
        
        
    def run(self):
        try:
            logger.info("Starting Data ingestion process")
            
            self.download_csv_from_gcp()
            self.split_data()
            
            logger.info("Data Ingestion completed succesfully")
            
        except CustomException as ce:
            logger.error(f"Custom Exception :  {str(ce)}")
            
        finally:
            logger.info("Data Ingestion process completed")
            
if __name__ == "__main__":
    
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()