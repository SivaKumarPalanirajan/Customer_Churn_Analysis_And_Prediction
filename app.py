from sklearn.compose import ColumnTransformer 
import pickle 
import streamlit as st 
import pandas as pd
scaler=pickle.load(open('Models/scaler.sav',"rb"))
encoders=pickle.load(open('Models/label_encoders.sav',"rb"))
model=pickle.load(open("Models/Predictor_Model.sav","rb"))
st.markdown("WELCOME TO THE CUSTOMER CHURN PREDICTOR")
st.info("""We can predict if a customer is likely to Churn or Stay if they reside in California""")
st.info("ENTER DETAILS BELOW")

number_of_dependents=st.number_input('Number of Dependents:')
city=st.text_input('City:').lower()
contract=st.radio('Contract:',['Month-to-Month','Two Year','One Year'])
total_charges=st.number_input('Total charges:')
total_long_distance_charges=st.number_input('Total long distance charges:')
total_revenue=st.number_input("Total revenue:")
tenure=st.number_input("Tenure:")
number_of_referrals=st.number_input("Number of Referrals:")
input=pd.DataFrame({'number_of_dependents':number_of_dependents,'city':city,'contract':contract,'total_charges':total_charges,'total_long_distance_charges':total_long_distance_charges,'total_revenue':total_revenue,'tenure':tenure,'number_of_referrals':number_of_referrals},index=[0])
proceed=st.radio('GIVE PREDICTION?',["No","Yes"])
if proceed=='Yes':
    cat_feat=['city','contract']
    num_feat=['number_of_dependents', 'total_charges', 'total_long_distance_charges', 'total_revenue', 'tenure', 'number_of_referrals']
    for i in cat_feat:
        input[i]=encoders[i].transform(input[i])
    input[num_feat]=scaler.transform(input[num_feat])

    st.info(f"The customer's predicted status: {model.predict(input)[0]}")
else:
    st.stop()