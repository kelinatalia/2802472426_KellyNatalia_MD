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
    st.header("Informasi Sistem")
    st.info("Sistem berbasis kecerdasan buatan ini dirancang untuk melakukan kalkulasi prediktif terhadap potensi rekruitmen mahasiswa berdasarkan metrik akademik.")
    st.markdown("---")
    st.write("**Instruksi:**")
    st.caption("Input data pada panel yang tersedia. Dashboard akan memperbarui visualisasi secara dinamis.")

st.title("Smart Career Analytics: Prediksi Penempatan & Gaji")
st.write("Evaluasi parameter kompetensi dan performa akademik untuk menentukan indeks kesiapan kerja.")

tab_data, tab_profil = st.tabs(["Indikator Akademik", "Kompetensi & Latar Belakang"])

with tab_data:
    col1, col2 = st.columns(2)
    with col1:
        ssc = st.slider("Pencapaian SSC (10th) %", 0, 100, 84)
        hsc = st.slider("Pencapaian HSC (12th) %", 0, 100, 76)
        degree = st.slider("Persentase Kelulusan Degree", 0, 100, 72)
    with col2:
        cgpa = st.number_input("Skala CGPA Kumulatif (0-10)", 0.0, 10.0, 7.88, step=0.01, format="%.2f")
        attendance = st.slider("Rasio Kehadiran (%)", 0, 100, 93)
        entrance = st.slider("Skor Ujian Masuk", 0, 100, 68)

with tab_profil:
    col3, col4 = st.columns(2)
    with col3:
        tech_skill = st.slider("Skor Kompetensi Teknis", 0, 100, 88)
        soft_skill = st.slider("Skor Interpersonal (Soft Skill)", 0, 100, 67)
        gender = st.radio("Identitas Gender", ["Male", "Female"], horizontal=True)
    with col4:
        internship = st.number_input("Jumlah Pengalaman Magang", 0, 20, 0)
        backlogs = st.number_input("Jumlah Mata Kuliah Mengulang", 0, 10, 1)
        extra = st.radio("Aktivitas Ekstrakurikuler", ["Yes", "No"], horizontal=True)

st.markdown("---")
st.header("Analisis Profil Real-Time")

g1, g2, g3 = st.columns(3)

with g1:
    academic_idx = (ssc + hsc + degree + (cgpa*10)) / 4
    fig_academic = go.Figure(go.Indicator(
        mode = "gauge+number", value = academic_idx,
        title = {'text': "Academic Index", 'font': {'color': '#6c5ce7', 'size': 20}},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#6c5ce7"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#dfe6e9",
            'steps': [
                {'range': [0, 50], 'color': '#f1f2f6'},
                {'range': [50, 80], 'color': '#ced6e0'}]
        }
    ))
    fig_academic.update_layout(height=280, margin=dict(l=25, r=25, t=50, b=25), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_academic, use_container_width=True)

with g2:
    total_comp = (tech_skill + soft_skill) / 2
    fig_comp = go.Figure(go.Indicator(
        mode = "gauge+number", value = total_comp,
        title = {'text': "Competency Score", 'font': {'color': '#e67e22', 'size': 20}},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#e67e22"},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 50], 'color': '#f1f2f6'},
                {'range': [50, 80], 'color': '#ced6e0'}]
        }
    ))
    fig_comp.update_layout(height=280, margin=dict(l=25, r=25, t=50, b=25), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_comp, use_container_width=True)

with g3:
    overall_detail = (academic_idx + total_comp) / 2
    fig_overall = go.Figure(go.Indicator(
        mode = "gauge+number", value = overall_detail,
        title = {'text': "Overall Readiness", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#2ecc71"},
            'shape': "angular"
        }
    ))
    fig_overall.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_overall, use_container_width=True)

st.info(f"Kesiapan Kerja (Readiness Score): {overall_detail:.2f} Points")

st.markdown("---")
if st.button("Jalankan Inferensi Model", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        'ssc_percentage': ssc, 'hsc_percentage': hsc, 'degree_percentage': degree, 'cgpa': cgpa, 
        'entrance_exam_score': entrance, 'technical_skill_score': tech_skill, 'soft_skill_score': soft_skill, 
        'internship_count': internship, 'live_projects': 1, 'work_experience_months': 0, 'certifications': 1, 
        'attendance_percentage': attendance, 'backlogs': backlogs, 'gender': 1 if gender == "Male" else 0, 
        'extracurricular_activities': 1 if extra == "Yes" else 0
    }])
    
    res_col1, res_col2 = st.columns([1.5, 1])
    
    with res_col1:
        pred = clf_model.predict(input_df)[0]
        if pred == 1:
            st.success("Hasil Proyeksi: PLACED (Terklasifikasi Lulus)")
            salary = reg_model.predict(input_df)[0]
            st.write("Estimasi Remunerasi Per Tahun")
            st.title(f"₹ {salary:.2f} LPA")
            st.balloons()
        else:
            st.error("Hasil Proyeksi: NOT PLACED (Peluang Rendah)")
            
    with res_col2:
        st.write("**Matriks Radar Kompetensi**")
        fig_big_radar = go.Figure(data=go.Scatterpolar(
            r=[ssc, hsc, degree, tech_skill, soft_skill],
            theta=['Akademik SSC', 'Akademik HSC', 'Degree %', 'Skor Teknis', 'Skor Soft Skill'],
            fill='toself', fillcolor='rgba(84, 160, 255, 0.5)', line=dict(color='#2e86de')
        ))
        fig_big_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=350)
        st.plotly_chart(fig_big_radar, use_container_width=True)