import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main

def main():
    st.title("Loan Approval Prediction App")

    st.header("Applicant Information")

    # Example input widgets
    gender = st.number_input("Male", value=True)
    married = st.number_input("Married", value=True)
    dependents = st.number_input("Number of Dependents", min_value=0)
    education = st.number_input("Graduate", value=True)
    self_employed = st.number_input("Self Employed", value=True)
    applicant_income = st.number_input("Applicant Income")
    coapplicant_income = st.number_input("Coapplicant Income")
    loan_amount = st.number_input("Loan Amount")
    loan_amount_term = st.number_input("Loan Amount Term")
    credit_history = st.number_input("Credit History")
    property_area = st.number_input("Urban", value=True)

    if st.button("Predict Loan Approval"):
        # Create a DataFrame with the input features
        df = pd.DataFrame({
            "Gender": [1 if gender else 0],
            "Married": [1 if married else 0],
            "Dependents": [dependents],
            "Education": [1 if education else 0],
            "Self_Employed": [1 if self_employed else 0],
            "ApplicantIncome": [applicant_income],
            "CoapplicantIncome": [coapplicant_income],
            "LoanAmount": [loan_amount],
            "Loan_Amount_Term": [loan_amount_term],
            "Credit_History": [credit_history],
            "Property_Area": [1 if property_area else 0],
        })

        # Get the prediction service
        service = prediction_service_loader(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            running=False,
        )

        if service is None:
            st.warning("No service found. Please run the deployment pipeline first.")
        else:
            try:
                # Convert the preprocessed data to the required format for prediction
                json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
                data_for_prediction = np.array(json_list)

                # Make the prediction
                prediction = service.predict(data_for_prediction)

                # Display the prediction
                st.success(f"The loan approval prediction is: {prediction}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

    # Add other sections of your Streamlit app as needed

if __name__ == "__main__":
    main()
