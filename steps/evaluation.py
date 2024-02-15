import logging

import pandas as pd
from zenml import step
from sklearn.base import ClassifierMixin
from src.evaluation import AccuracyScore,PrecisionScore,RecallScore,F1Score

from typing import Tuple
from typing_extensions import Annotated

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker = experiment_tracker.name)
def evaluate_model(model: ClassifierMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame) -> Tuple[
                       Annotated[float,'accuray_s'],
                       Annotated[float,'precision_s'],
                       Annotated[float,'recall_s'],
                       Annotated[float,'fl_scores'],
                   ]:
    
    """
    """
    try:
        prediction = model.predict(X_test)
        accuray_score = AccuracyScore()
        accuray_s = accuray_score.calculate_scores(y_test,prediction)
        mlflow.log_metric("accuray_s",accuray_s)
        precision = PrecisionScore()
        precision_s = precision.calculate_scores(y_test,prediction)
        mlflow.log_metric("precision_s",precision_s)
        recall = RecallScore()
        recall_s = recall.calculate_scores(y_test,prediction)
        mlflow.log_metric("recall_s",recall_s)
        f1_score = F1Score()
        fl_scores = f1_score.calculate_scores(y_test,prediction)
        mlflow.log_metric("fl_scores",fl_scores)

        return accuray_s, precision_s,recall_s,fl_scores
    except Exception as e:
        logging.error("Error while doing evaluation:{}".format(e))
        raise e
    
    