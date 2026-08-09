import streamlit as st
import pandas as pd
import joblib 

#load the saved models and encoders
model=joblib.loadimport=joblib.load("student_performance_model.pkl")
feature_encoder=joblib.load("feature_encoder.pkl")
target_encoder=joblib.load("target_encoder.pkl")

#App title
st.title("Student Academic Performance Prediction")
st.write(
    "Enter student information below to predict Academic Performance Class"
)
#User inputs
gender=st.selectbox(
    "Gender",
    ["Male","Female"]
)
age=st.number_input(
    "Age",
    min_value=10,
    max_value=30,
    value=18
)
district=st.selectbox(
    "District",
    [
     "Rangpur",
     "Rajshahi",
     "Barisal",
     "Mymesignh",
     "Shylet",
     "Khulna",
     "Dhaka",
     "Chattogram"
     ]
)
school_type=st.selectbox(
    "School Type",
    ["Private","Public"]
)
parent_education=st.selectbox(
    "Parent Education",
   ["Graduate","Primary","High School","Secondary"] 
)
study_hours=st.number_input(
    "Study Hours per Week",
    min_value=0,
    max_value=100,
    value=10

)
attendance=st.number_input(
    "Attendance",
    min_value=0,
    max_value=100,
    value=75
)
Family_Income_BDT=st.number_input(
    "Family Income",
    min_value=0,
    value=0
    )
internet_access=st.selectbox(
    "Internet Access",
    ["Yes","No"]
)
private_tuition=st.selectbox(
    "Private Tuition",
    ["Yes","No"]
)
previous_gpa=st.number_input(
    "Previous GPA",
    min_value=0.0,
    max_value=5.0,
    value=3.0,
    step=0.01
)
ssc_result=st.number_input(
    "SSC Result",
    min_value=0.0,
    max_value=5.0,
    value=3.0,
    step=0.01
)

input_data=pd.DataFrame({
    "Gender":[gender],
    "Age":[age],
    "District":[district],
    "School_Type":[school_type],
    "Study_Hours_per_Week":[study_hours],
    "Attendance":[attendance],
    "Parent_Education":[parent_education],
    "Family_Income_BDT":[Family_Income_BDT],
    "Internet_Access":[internet_access],
    "Private_Tuition":[private_tuition],
    "Previous_GPA":[previous_gpa],
    "SSC_Result":[ssc_result]
})
if st.button("Predict Performance"):
    input_encoded=input_data.copy()

    input_encoded["Gender"]=input_encoded["Gender"].map({
        "Male":0,
        "Female":1
    })
    input_encoded["School_Type"]=input_encoded["School_Type"].map({
        "Public":0,
        "Private":1
    })
    input_encoded["Internet_Access"]=input_encoded["Internet_Access"].map({
        "No":0,
        "Yes":1
    })
    input_encoded["Private_Tuition"]=input_encoded["Private_Tuition"].map({
        "No":0,
        "Yes":1
    })
    input_encoded["District"]=input_encoded["District"].map({
        "Barisal": 0,
        "Chattogram": 1,
        "Dhaka": 2,
        "Khulna": 3,
        "Mymensingh": 4,
        "Rajshahi": 5,
        "Rangpur": 6,
        "Sylhet": 7
    })
    input_encoded["Parent_Education"]=input_encoded["Parent_Education"].map({
        "Graduate": 0,
        "Higher Secondary": 1,
        "Primary": 2,
        "Secondary": 3
    })
    prediction=model.predict(input_encoded)
    predicted_class=target_encoder.inverse_transform(prediction)
    st.success(f"Predicted Performance Class:{predicted_class[0]}")


