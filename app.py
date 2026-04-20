import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="Career Analytics Dashboard", layout="wide")

@st.cache_resource
def load_models():
    with open('classificationModel.pkl', 'rb') as f:
        clf_model = pickle.load(f) 
    with open('regressionModel.pkl', 'rb') as f:
        reg_model = pickle.load(f)
    return clf_model, reg_model

clf_model, reg_model = load_models()

with st.sidebar:
    st.header("Sistem Navigasi")
    st.info("Prediksi peluang karir berdasarkan metrik performa akademik dan kompetensi teknis mahasiswa.")
    st.markdown("---")
    st.write("**Panduan:**")
    st.caption("Sesuaikan slider pada tab yang tersedia, lalu tekan tombol di bawah untuk melihat proyeksi.")
    st.success("Koneksi Model Aktif")

st.title("Career Path Prediction")
st.write("Analisis komprehensif kesiapan industri mahasiswa.")

tab_data, tab_profil = st.tabs(["Prestasi Akademik", "Kompetensi Profil"])

with tab_data:
    col1, col2 = st.columns(2)
    with col1:
        ssc = st.slider("Nilai SSC (10th) %", 0, 100, 82)
        hsc = st.slider("Nilai HSC (12th) %", 0, 100, 78)
        degree = st.slider("Skor Kelulusan Degree %", 0, 100, 80)
    with col2:
        cgpa = st.number_input("Akumulasi CGPA (Skala 10)", 0.0, 10.0, 3.75, step=0.01, format="%.2f")
        attendance = st.slider("Rasio Kehadiran (%)", 0, 100, 92)
        entrance = st.slider("Skor Tes Masuk", 0, 100, 65)

with tab_profil:
    col3, col4 = st.columns(2)
    with col3:
        tech_skill = st.slider("Skor Kemampuan Teknis", 0, 100, 88)
        soft_skill = st.slider("Skor Kemampuan Interpersonal", 0, 100, 85)
        extra = st.radio("Aktif Ekstrakurikuler", ["Yes", "No"], horizontal=True)
    with col4:
        internship = st.number_input("Pengalaman Magang (Total)", 0, 10, 2)
        backlogs = st.number_input("Jumlah Retake Matkul", 0, 5, 0)
        gender = st.selectbox("Gender", ["Male", "Female"])

st.markdown("---")
st.header("Analisis Profil Real-Time")

m1, m2 = st.columns(2)

with m1:
    academic_idx = (ssc + hsc + degree + (cgpa*10)) / 4
    fig_academic = go.Figure(go.Indicator(
        mode = "number+gauge", value = academic_idx,
        domain = {'x': [0.1, 1], 'y': [0, 1]},
        title = {'text': "Academic Index", 'font': {'size': 18}},
        gauge = {
            'shape': "bullet",
            'axis': {'range': [None, 100]},
            'bar': {'color': "#6c5ce7"},
            'steps': [
                {'range': [0, 50], 'color': "#dfe6e9"},
                {'range': [50, 80], 'color': "#b2bec3"}]
        }
    ))
    fig_academic.update_layout(height=150, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_academic, use_container_width=True)

with m2:
    total_comp = (tech_skill + soft_skill) / 2
    fig_comp = go.Figure(go.Indicator(
        mode = "number+gauge", value = total_comp,
        domain = {'x': [0.1, 1], 'y': [0, 1]},
        title = {'text': "Skill Mastery", 'font': {'size': 18}},
        gauge = {
            'shape': "bullet",
            'axis': {'range': [None, 100]},
            'bar': {'color': "#fd79a8"},
            'steps': [
                {'range': [0, 50], 'color': "#dfe6e9"},
                {'range': [50, 80], 'color': "#b2bec3"}]
        }
    ))
    fig_comp.update_layout(height=150, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")
if st.button("Generate Career Analysis", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        'ssc_percentage': ssc, 'hsc_percentage': hsc, 'degree_percentage': degree, 'cgpa': cgpa, 
        'entrance_exam_score': entrance, 'technical_skill_score': tech_skill, 'soft_skill_score': soft_skill, 
        'internship_count': internship, 'live_projects': 1, 'work_experience_months': 0, 'certifications': 1, 
        'attendance_percentage': attendance, 'backlogs': backlogs, 'gender': 1 if gender == "Male" else 0, 
        'extracurricular_activities': 1 if extra == "Yes" else 0
    }])
    
    res_left, res_right = st.columns([1.2, 1])
    
    with res_left:
        st.subheader("Visual Mapping Kompetensi")
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=[ssc, hsc, degree, tech_skill, soft_skill],
            theta=['SSC', 'HSC', 'Degree', 'Technical', 'Soft Skill'],
            fill='toself', fillcolor='rgba(108, 92, 231, 0.4)', line=dict(color='#6c5ce7')
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=400)
        st.plotly_chart(fig_radar, use_container_width=True)
            
    with res_right:
        st.subheader("Hasil Inferensi Model")
        pred = clf_model.predict(input_df)[0]
        if pred == 1:
            st.success("STATUS: PLACED")
            salary = reg_model.predict(input_df)[0]
            st.metric("Estimasi Paket Gaji (LPA)", f"₹ {salary:.2f}")
            st.balloons()
        else:
            st.error("STATUS: NOT PLACED")
            st.info("Saran: Tingkatkan skor kompetensi teknis dan sertifikasi.")