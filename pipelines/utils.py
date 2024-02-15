import logging

import pandas as pd
from src.data_cleaning import DataCleaning, DataPreprocessStrategy


def get_data_for_test():
    try:
        df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\zenml_project\data\loan_data.csv")
        df = df.sample(n=100)
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.handle_data()
        df.drop(["Loan_Status"], axis=1, inplace=True)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e
