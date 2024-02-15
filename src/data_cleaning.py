import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Data preprocessing strategy which preprocesses the data.
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
            try:
                # Drop unnecessary columns
                cols_to_drop = ['Loan_ID']
                data = data.drop(cols_to_drop, axis=1)

                # Handling missing values
                data['Dependents'] = data['Dependents'].replace('3+', 3)
                data['Dependents'] = data['Dependents'].fillna(0).astype(int)
                data['Self_Employed'] = data['Self_Employed'].fillna('No')
                data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].mean())
                data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])
                data['Credit_History'] = data['Credit_History'].fillna(1)


                # Convert categorical variables to numerical using Label Encoding
                label_encoder = LabelEncoder()
                categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
                for column in categorical_columns:
                    data[column] = label_encoder.fit_transform(data[column])

                return data
            except Exception as e:
                logging.error(e)
                raise e


class DataDivideStrategy(DataStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides the data into train and test data.
        """
        try:
            X = data.drop('Loan_Status', axis=1)
            y = data['Loan_Status']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e


class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)

