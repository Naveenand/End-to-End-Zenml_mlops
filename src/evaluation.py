import logging
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluation(ABC):
    """
    Abstract Class defining strategy for evaluation our models
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        """
        pass

class AccuracyScore(Evaluation):
    """
    Evaluation strategy that uses Accuracy Score for classification models
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            accuracy: float
        """
        try:
            logging.info("Entered the calculate_score method of the AccuracyScore class")
            accuracy = accuracy_score(y_true, y_pred)
            logging.info("The accuracy score is: " + str(accuracy))
            return accuracy
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the AccuracyScore class. Exception message:  "
                + str(e)
            )
            raise e


class PrecisionScore(Evaluation):
    """
    Evaluation strategy that uses Precision Score for classification models
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            precision: float
        """
        try:
            logging.info("Entered the calculate_score method of the PrecisionScore class")
            precision = precision_score(y_true, y_pred)
            logging.info("The precision score is: " + str(precision))
            return precision
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the PrecisionScore class. Exception message:  "
                + str(e)
            )
            raise e


class RecallScore(Evaluation):
    """
    Evaluation strategy that uses Recall Score for classification models
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            recall: float
        """
        try:
            logging.info("Entered the calculate_score method of the RecallScore class")
            recall = recall_score(y_true, y_pred)
            logging.info("The recall score is: " + str(recall))
            return recall
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the RecallScore class. Exception message:  "
                + str(e)
            )
            raise e


class F1Score(Evaluation):
    """
    Evaluation strategy that uses F1 Score for classification models
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            f1: float
        """
        try:
            logging.info("Entered the calculate_score method of the F1Score class")
            f1 = f1_score(y_true, y_pred)
            logging.info("The F1 score is: " + str(f1))
            return f1
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the F1Score class. Exception message:  "
                + str(e)
            )
            raise e

