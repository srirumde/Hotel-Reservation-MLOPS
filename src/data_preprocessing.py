import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
from utils.common_functions import read_yaml,load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:
    
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        
        
    def preprocess_data(self,df):
        try:
            logger.info("Starting our data processing step")
            
            logger.info("Dropping the Columns")
            cols_to_drop = ['Unnamed: 0', 'Booking_ID']
            cols_to_drop = [col for col in cols_to_drop if col in df.columns]
            df.drop(columns=cols_to_drop, inplace=True)
            df.drop_duplicates(inplace=True)
            
            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]
            
            logger.info("Applying Label Encoding")
            
            label_encoder = LabelEncoder()
            mappings={}
            
            for col in cat_cols:
                df[col]= label_encoder.fit_transform(df[col])
                mappings[col] = {label: code for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
            
            logger.info("Label Mappings are : ")
            for col,mappings in mappings.items():
                logger.info(f"{col} : {mappings}")
                
            logger.info("Doing skewness Handeling")
            skewness_threshold = self.config["data_processing"]["skewness_threshold"]
            skewness = df[num_cols].apply(lambda x:x.skew())
            
            for column in skewness[skewness>skewness_threshold].index:
                df[column] = np.log1p(df[column])
            return df
        
        except Exception as e:
            logger.error(f"Error during preprocess Occured : {e}")
            raise CustomException("Error while preprocess data", e)
        
        
    def balance_data(self,df):
        try:
            logger.info("Starting our data balancing step")
            X = df.drop(columns='booking_status')
            Y = df["booking_status"]
            
            smote = SMOTE(random_state=42)
            X_resampled, Y_resampled= smote.fit_resample(X,Y)       
                
            balanced_df= pd.DataFrame(X_resampled, columns= X.columns)
            balanced_df["booking_status"]= Y_resampled
            
            logger.info("Data balanced sucessful")
            return balanced_df
        
        except Exception as e:
            logger.error(f"Error during Balancing data Occured : {e}")
            raise CustomException("Error while balancing data", e)
        
        
    def select_features(self, df):
        try:
            logger.info("Starting our feature selection step")
            X = df.drop(columns='booking_status')
            Y = df["booking_status"]    
        
            model = RandomForestClassifier(random_state=42)
            model.fit(X,Y)      
        
            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importance
             })
            top_features_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        
            num_features_to_select = self.config["data_processing"]["no_of_features"]
            top_features = top_features_importance_df["feature"].head(num_features_to_select).values
            logger.info(f"Features Selected {top_features}")
            top_df = df[top_features.tolist() + ["booking_status"]]
        
            logger.info("Feature Selection Completed successfully")
            return top_df  # THIS WAS MISSING
        
        except Exception as e:
            logger.error(f"Error during Feature selection Occurred: {e}")
            raise CustomException("Error while Feature Selection", e)
        
        
    def save_data(self,df,file_path):
        try:
            logger.info("Saving our data in processed folder")
            
            df.to_csv(file_path, index=False)
            
            logger.info(f"Data Saved sucessfully to {file_path}")
            
        except Exception as e:
            logger.error(f"Error during Saving data Occured : {e}")
            raise CustomException("Error while saving data", e)
        
    def process(self):
        try:
            print("DEBUG: Starting process()")  # Temporary debug
            logger.info("Loading data from RAW")
            print("DEBUG: Attempting to load train data")  # Temporary debug
            train_df = load_data(self.train_path)
            print("DEBUG: Train data loaded successfully")  # Temporary debug
            test_df = load_data(self.test_path)
            print("DEBUG: Test data loaded successfully")  # Temporary debug
        
        # ... rest of your process method ...
            logger.info("Loading data from RAW")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)
        
            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)
        
            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)
        
        # Get feature names from training set
            train_df = self.select_features(train_df)
            selected_features = train_df.columns.tolist()
        
        # Ensure test set has same columns (booking_status will always be included)
            test_df = test_df[selected_features]
        
            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)
        
            logger.info("Data processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during Preproccessing pipeline Occured : {e}")
            raise CustomException("Error while preprocessing data pipeline", e)
            

if __name__ == "__main__":
    try:
        print("Starting data processing...")
        processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
        processor.process()
        print("Data processing completed successfully")
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()         
            
            
            
            
            
            