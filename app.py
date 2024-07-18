# prompt: using streamlit  above model make web applicaton

import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('best_house_price_model.pkl')

# Define the features
features = ['id', 'Date', 'No of bedrooms', 'No of bathrooms', 'living area',
       'lot area', 'No of floors', 'waterfront present', 'No of views',
       'house condition', 'house grade', 'house area(excluding basement)',
       'Area of the basement', 'Built Year', 'Renovation Year', 'Postal Code',
       'Lattitude', 'Longitude', 'living_area_renov', 'lot_area_renov',
       'No of schools nearby', 'Distance from the airport']

# Create the Streamlit app
st.title("House Price Prediction")

# Create input fields for each feature
user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"Enter value for {feature}:")

# Create a DataFrame from user input
input_df = pd.DataFrame([user_input])

# Make prediction when the user clicks the button
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")
