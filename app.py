import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Student Placement Predictor", layout="wide")

# load model
@st.cache_resource
def load_models():
    with open('classificationModel.pkl', 'rb') as f:
        clf_model = pickle.load(f) 
    with open('regressionModel.pkl', 'rb') as f:
        reg_model = pickle.load(f)
    return clf_model, reg_model

clf_model, reg_model = load_models()

# sidebar
st.sidebar.header("Input Data Mahasiswa")

def user_input_features():
    # numerik
    cgpa = st.sidebar.slider('CGPA (IPK)', 0.0, 10.0, 8.0)
    tech_skill = st.sidebar.number_input('Technical Skill Score', 0, 100, 80)
    soft_skill = st.sidebar.number_input('Soft Skill Score', 0, 100, 85)
    internship = st.sidebar.selectbox('Internship Count', [0, 1, 2, 3])
    backlogs = st.sidebar.number_input('Backlogs (Mata Kuliah Mengulang)', 0, 10, 0)
    
    # kategorikal
    gender = st.sidebar.radio('Gender', ['Male', 'Female'])
    extra = st.sidebar.selectbox('Extracurricular Activities', ['Yes', 'No'])
    
    data = {
        'ssc_percentage': 80, 'hsc_percentage': 80, 'degree_percentage': 80, 
        'cgpa': cgpa, 'entrance_exam_score': 80,
        'technical_skill_score': tech_skill, 'soft_skill_score': soft_skill,
        'internship_count': internship, 'live_projects': 1,
        'work_experience_months': 0, 'certifications': 1,
        'attendance_percentage': 90, 'backlogs': backlogs,
        'gender': 1 if gender == 'Male' else 0,
        'extracurricular_activities': 1 if extra == 'Yes' else 0
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# main page
st.title("🎓 Student Career Analytics")
st.markdown("Aplikasi ini memprediksi status penempatan kerja dan estimasi gaji mahasiswa")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Placement Status")
    prediction = clf_model.predict(input_df)
    prob = clf_model.predict_proba(input_df)[0][1]
    
    if prediction[0] == 1:
        st.success(f"### **PLACED**")
        st.write(f"Confidence: {prob:.2%}")
    else:
        st.error(f"### **NOT PLACED**")
        st.write(f"Confidence: {1-prob:.2%}")

with col2:
    st.subheader("Salary Estimation")
    if prediction[0] == 1:
        salary_pred = reg_model.predict(input_df)
        st.metric("Estimated Salary", f"{salary_pred[0]:.2f} LPA")
    else:
        st.info("Salary is only estimated for placed students.")

st.divider()
st.subheader("Visualisasi Skor Input")
st.bar_chart(input_df[['cgpa', 'technical_skill_score', 'soft_skill_score']].T)