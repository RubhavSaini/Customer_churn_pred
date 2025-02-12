import streamlit as st
import pickle
import pandas as pd

# loading the model
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["features_names"]

# loading the encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)


def main():
    st.title("✨ Customer Information Form ✨")
    st.markdown("---")
    placeholder = st.empty()

    # getting the input
    with placeholder.container():
        input_data = {
            'gender': st.selectbox('Gender', ['Male', 'Female']),
            # 'SeniorCitizen': st.selectbox('Senior Citizen', [0, 1]),
            'SeniorCitizen': st.selectbox('Senior Citizen', ['Yes', 'No']),
            'Partner': st.selectbox('Partner', ['Yes', 'No']),
            'Dependents': st.selectbox('Dependents', ['Yes', 'No']),
            'tenure': st.number_input('Tenure (months)', min_value=0, step=1),
            'PhoneService': st.selectbox('Phone Service', ['Yes', 'No']),
            'MultipleLines': st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service']),
            'InternetService': st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No']),
            'OnlineSecurity': st.selectbox('Online Security', ['Yes', 'No','No internet service']),
            'OnlineBackup': st.selectbox('Online Backup', ['Yes', 'No','No internet service']),
            'DeviceProtection': st.selectbox('Device Protection', ['Yes', 'No','No internet service']),
            'TechSupport': st.selectbox('Tech Support', ['Yes', 'No','No internet service']),
            'StreamingTV': st.selectbox('Streaming TV', ['Yes', 'No','No internet service']),
            'StreamingMovies': st.selectbox('Streaming Movies', ['Yes', 'No','No internet service']),
            'Contract': st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year']),
            'PaperlessBilling': st.selectbox('Paperless Billing', ['Yes', 'No']),
            'PaymentMethod': st.selectbox('Payment Method',
                                          ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']),
            'MonthlyCharges': st.number_input('Monthly Charges', min_value=0.0, step=0.01),
            'TotalCharges': st.number_input('Total Charges', min_value=0.0, step=0.01)
        }
        submit = st.button("🚀 Predict")



    if submit:
        # for clearing screen on submit
        placeholder.empty()

        input_data['SeniorCitizen'] = 1 if input_data['SeniorCitizen'] == 'Yes' else 0

        input_data_df = pd.DataFrame([input_data])


        # encoding categorical features using the saved encoders
        for column, encoder in encoders.items():
            input_data_df[column] = encoder.transform(input_data_df[column])

        # making prediction
        prediction = loaded_model.predict(input_data_df)
        pred_prob = loaded_model.predict_proba(input_data_df)


        st.markdown("📊 Prediction Result")
        result='Churn' if prediction[0] == 1 else 'No Churn'
        if result=="Churn":
            st.error(f"**Prediction:** {result}")
            st.error("i.e the customer may not continue the services")
        else:
            st.success(f"**Prediction:** {result}")
            st.success("i.e the customer will continue the services")
        st.info(f"**Prediction Probability:** {pred_prob[0][1]*100 if pred_prob[0][1]>pred_prob[0][0] else pred_prob[0][0]}%")

        print(f"Prediciton Probability: {pred_prob}")

        if st.button("New prediction?"):
            st.rerun()

if __name__ == "__main__":
    main()