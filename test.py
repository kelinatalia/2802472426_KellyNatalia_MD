import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Student Career Analytics", layout="wide")

# load model
@st.cache_resource
def load_models():
    with open('classificationModel.pkl', 'rb') as f:
        clf_model = pickle.load(f) 
    with open('regressionModel.pkl', 'rb') as f:
        reg_model = pickle.load(f)
    return clf_model, reg_model

clf_model, reg_model = load_models()

with st.sidebar:
    st.title("Tentang Aplikasi")
    st.info("""
    Platform analisis prediktif ini mengevaluasi korelasi antara performa akademik dan kompetensi teknis terhadap peluang karir mahasiswa.
    """)
    
    st.markdown("---")
    st.write("**Petunjuk:**")
    st.caption("Lengkapi parameter profil Anda di tab Data Input. Sistem akan memproses prediksi secara otomatis melalui dashboard.")

def create_gauge(value, title, max_val, color):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        gauge = {'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "white"},
                 'bar': {'color': color},
                 'bgcolor': "rgba(0,0,0,0)",
                 'borderwidth': 2,
                 'bordercolor': "white",
                 }
    ))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    return fig

def create_radar(values, categories):
    fig = go.Figure(data=go.Scatterpolar(
        r=values, theta=categories, fill='toself', line=dict(color='#FF6B6B')
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="gray"),
            angularaxis=dict(gridcolor="gray")
        ),
        showlegend=False, height=400, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}
    )
    return fig

# main page
st.title("Student Career Analytics")

c1, c2 = st.columns(2)
with c1:
    ssc = st.slider("SSC % (10th)", 0, 100, 75)
    hsc = st.slider("HSC % (12th)", 0, 100, 70)
    degree = st.slider("Degree %", 0, 100, 72)
with c2:
    # punya kating itu formatnya 8,00, kita ubah stepnya biar muncul decimal
    cgpa = st.number_input("Current CGPA (0-10)", 0.0, 10.0, 8.0, step=0.01, format="%.2f")
    attendance = st.slider("Attendance %", 0, 100, 85)
    entrance = st.slider("Entrance Score", 0, 100, 70)

tab_skill, tab_bio = st.tabs(["Profil & Keahlian", "Data Tambahan"])
with tab_skill:
    ts1, ts2 = st.columns(2)
    with ts1:
        tech_skill = st.slider('Technical Skill Score', 0, 100, 80)
    with ts2:
        soft_skill = st.slider('Soft Skill Score', 0, 100, 85)

with tab_bio:
    tb1, tb2 = st.columns(2)
    with tb1:
        internship = st.number_input('Internship Count', 0, 15, 1) # pakai plus minus, max dinaikin
        gender = st.radio('Gender', ['Male', 'Female'], horizontal=True)
    with tb2:
        backlogs = st.number_input('Backlogs', 0, 10, 0)
        extra = st.selectbox('Extracurricular Activities', ['Yes', 'No'])

# data mapping
input_df = pd.DataFrame([{
    'ssc_percentage': ssc, 'hsc_percentage': hsc, 'degree_percentage': degree, 
    'cgpa': cgpa, 'entrance_exam_score': entrance,
    'technical_skill_score': tech_skill, 'soft_skill_score': soft_skill,
    'internship_count': internship, 'live_projects': 1,
    'work_experience_months': 0, 'certifications': 1,
    'attendance_percentage': attendance, 'backlogs': backlogs,
    'gender': 1 if gender == 'Male' else 0,
    'extracurricular_activities': 1 if extra == 'Yes' else 0
}])

st.markdown("---")

st.subheader("Live Profiling Insights")

g1, g2 = st.columns(2)
with g1:
    academic_idx = (ssc + hsc + degree + (cgpa*10)) / 4
    st.plotly_chart(create_gauge(academic_idx, "Academic Index", 100, "#55E6C1"), use_container_width=True)
with g2:
    total_competency = tech_skill + soft_skill 
    st.plotly_chart(create_gauge(total_competency, "Total Competency", 200, "#FEA47F"), use_container_width=True)

r1, r2 = st.columns([2, 1])
with r1:
    st.write("**Sebaran Kompetensi**")
    radar_cats = ['SSC', 'HSC', 'Degree', 'Technical', 'Soft Skill']
    radar_vals = [ssc, hsc, degree, tech_skill, soft_skill]
    st.plotly_chart(create_radar(radar_vals, radar_cats), use_container_width=True)
    
    readiness_pts = int(academic_idx / 10) 
    st.info(f"Job Readiness: {readiness_pts} Pts")

with r2:
    st.write("**Hasil Prediksi Model**")
    
    if st.button("Analisa Profil Saya", type="primary", use_container_width=True):
        prediction = clf_model.predict(input_df)
        if prediction[0] == 1:
            st.success("🎯 **STATUS: PLACED**")
            salary = reg_model.predict(input_df)[0]
            st.metric("Estimasi Paket Gaji (LPA)", f"₹ {salary:.2f} LPA")
            st.balloons()
        else:
            st.error("⚠️ **STATUS: NOT PLACED**")
            st.write("Fokus pada peningkatan teknis dan pengurangan backlogs.")