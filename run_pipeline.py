from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    # Run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path = r'C:\Users\HP\OneDrive\Desktop\zenml_project\data\loan_data.csv')

#mlflow ui --backend-store-uri "file:C:\Users\HP\AppData\Roaming\zenml\local_stores\938535d0-4862-4a49-9358-901593c6db7f\mlruns"