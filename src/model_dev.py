import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression

class Model(ABC):
    """
    Abstract the class for all model
    """
    @abstractmethod
    def train(self, X_train,y_train):
        """
        Trains the model
        Args:
             X_train: training data
             y_train: trainig labels
        Returns:
              trained model
        """
        pass


class LogisticRegressionModel(Model):
    """
    Logistic regression Model(classification)
    """
    def train(self, X_train,y_train,**kwargs):
        try:
            reg = LogisticRegression(**kwargs)
            reg.fit(X_train,y_train)
            logging.info("Model traning completed")
            return reg
        except Exception as e:
            logging.error("Error in traning model: {}".format(e))
            raise e

        

