import streamlit as st
import pickle
import numpy as np
import os

# Load the trained model (replace 'model.pkl' with your actual file name)

#with open('hhmodel.pkl', 'rb') as file:
#file_path="C:\\Users\\sakshi\\OneDrive\\Desktop\\BE_Project\\Project_ML\\trained_model.pkl"
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)



# Define the customized ranges for each feature based on dataset statistics
custom_ranges = {
    'Engine rpm': (61.0, 2239.0),
    'Lub oil pressure': (0.003384, 7.265566),
    'Fuel pressure': (0.003187, 21.138326),
    'Coolant pressure': (0.002483, 7.478505),
    'lub oil temp': (71.321974, 89.580796),
    'Coolant temp': (61.673325, 195.527912),
    'Temperature_difference': (-22.669427, 119.008526)
}

if "reset" not in st.session_state:
    st.session_state.reset = False

default_values = {
    "engine_rpm": float(custom_ranges['Engine rpm'][1] / 2),
    "lub_oil_pressure": (custom_ranges['Lub oil pressure'][0] + custom_ranges['Lub oil pressure'][1]) / 2,
    "fuel_pressure": (custom_ranges['Fuel pressure'][0] + custom_ranges['Fuel pressure'][1]) / 2,
    "coolant_pressure": (custom_ranges['Coolant pressure'][0] + custom_ranges['Coolant pressure'][1]) / 2,
    "lub_oil_temp": (custom_ranges['lub oil temp'][0] + custom_ranges['lub oil temp'][1]) / 2,
    "coolant_temp": (custom_ranges['Coolant temp'][0] + custom_ranges['Coolant temp'][1]) / 2,
    "temp_difference": (custom_ranges['Temperature_difference'][0] + custom_ranges['Temperature_difference'][1]) / 2
}

# If reset is True, apply defaults before rendering widgets
if st.session_state.reset:
    for key, val in default_values.items():
        st.session_state[key] = val
    st.session_state.reset = False
    st.rerun()



# Feature Descriptions
feature_descriptions = {
    'Engine rpm': 'Revolution per minute of the engine.',
    'Lub oil pressure': 'Pressure of the lubricating oil.',
    'Fuel pressure': 'Pressure of the fuel.',
    'Coolant pressure': 'Pressure of the coolant.',
    'lub oil temp': 'Temperature of the lubricating oil.',
    'Coolant temp': 'Temperature of the coolant.',
    'Temperature_difference': 'Temperature difference between components.'
}



# Engine Condition Prediction App
def main():
    st.title("Engine Condition Prediction")

    # Display feature descriptions
    st.sidebar.title("Feature Descriptions")
    for feature, description in feature_descriptions.items():
        st.sidebar.markdown(f"**{feature}:** {description}")


    # Input widgets with customized ranges
    engine_rpm = st.slider("Engine RPM", min_value=float(custom_ranges['Engine rpm'][0]), 
                           max_value=float(custom_ranges['Engine rpm'][1]), 
                           value=st.session_state.get("engine_rpm", default_values["engine_rpm"]),
                           key="engine_rpm"
                           )
    lub_oil_pressure = st.slider("Lub Oil Pressure", min_value=custom_ranges['Lub oil pressure'][0], 
                                 max_value=custom_ranges['Lub oil pressure'][1], 
                                 value=st.session_state.get("lub_oil_pressure", default_values["lub_oil_pressure"]),
                                 key="lub_oil_pressure"
                                 )
    fuel_pressure = st.slider("Fuel Pressure", min_value=custom_ranges['Fuel pressure'][0], 
                              max_value=custom_ranges['Fuel pressure'][1], 
                              value=st.session_state.get("fuel_pressure", default_values["fuel_pressure"]),
                              key="fuel_pressure"
                              )
    coolant_pressure = st.slider("Coolant Pressure", min_value=custom_ranges['Coolant pressure'][0], 
                                 max_value=custom_ranges['Coolant pressure'][1], 
                                 value=st.session_state.get("coolant_pressure", default_values["coolant_pressure"]),
                                  key="coolant_pressure"
                                 )
    lub_oil_temp = st.slider("Lub Oil Temperature", min_value=custom_ranges['lub oil temp'][0], 
                             max_value=custom_ranges['lub oil temp'][1], 
                             value=st.session_state.get("lub_oil_temp", default_values["lub_oil_temp"]),
                             key="lub_oil_temp"
                             )
    coolant_temp = st.slider("Coolant Temperature", min_value=custom_ranges['Coolant temp'][0], 
                             max_value=custom_ranges['Coolant temp'][1], 
                             value=st.session_state.get("coolant_temp", default_values["coolant_temp"]),
                             key="coolant_temp"
                             )
    temp_difference = st.slider("Temperature Difference", min_value=custom_ranges['Temperature_difference'][0], 
                                max_value=custom_ranges['Temperature_difference'][1], 
                                value=st.session_state.get("temp_difference", default_values["temp_difference"]),
                                key="temp_difference"
                                )

    # Predict button
    if st.button("Predict Engine Condition"):
        result, confidence = predict_condition(engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference)
        

        # Explanation
        if result == 0:
            st.info(f"The engine is predicted to be in a normal condition. As the  Confidence level of the machine is : {1.0 - confidence:.2%}")
        else:
            st.warning(f"Warning! Please investigate further,As the  Confidence level of the machine is : {1.0 - confidence:.2%}")

    # Reset button
    if st.button("Reset Values"):
        st.session_state.reset = True
        st.rerun()
# Function to predict engine condition
def predict_condition(engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference):
    input_data = np.array([engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference]).reshape(1, -1)
    prediction = model.predict(input_data)
    confidence = model.predict_proba(input_data)[:, 1]  # For binary classification, adjust as needed
    return prediction[0], confidence[0]

if __name__ == "__main__":
    main()
