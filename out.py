import joblib
import pandas as pd
import numpy as np

# Load the exported assets from the directory
model = joblib.load('student_gpa_model.pkl')
x_scaler = joblib.load('x_scaler_student.pkl')
y_scaler = joblib.load('y_scaler_student.pkl')
label_encoder1 = joblib.load('label_encoder1_student.pkl')
label_encoder2 = joblib.load('label_encoder2_student.pkl')
label_encoder3 = joblib.load('label_encoder3_student.pkl')

print("--- Real-time Student GPA Assessment Interface ---")

# Gather human inputs
age = int(input("Enter age: "))
gender = input("Enter gender (Male/Female): ")
study_hours = float(input("Enter weekly study hours: "))
attendance = float(input("Enter attendance rate (0-100): "))
major = input("Enter major (Science/Arts/Business/Engineering): ")
part_time_job = input("Enter part-time job status (Yes/No): ")

# Encode the inputs using saved mappings
gender_encoded = label_encoder1.transform([gender])[0]
major_encoded = label_encoder2.transform([major])[0]
part_time_job_encoded = label_encoder3.transform([part_time_job])[0]

# Construct structural matrix row
new_data = pd.DataFrame([[
    age, 
    gender_encoded, 
    study_hours, 
    attendance, 
    major_encoded, 
    part_time_job_encoded
]], columns=['age', 'gender_encoded', 'study_hours', 'attendance', 'major_encoded', 'part_time_job_encoded'])

# Standardize inputs using original scaling weights
new_data_scaled = x_scaler.transform(new_data)

# Compute prediction metrics
predicted_gpa_scaled = model.predict(new_data_scaled)

# Reverse-scale prediction matrix back into original GPA boundaries
predicted_gpa = y_scaler.inverse_transform(predicted_gpa_scaled)

print(f"\nCalculated Predicted GPA: {predicted_gpa[0][0]:.2f}")
