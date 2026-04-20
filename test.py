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

st.title("Sistem Proyeksi Karir & Estimasi Remunerasi")
st.write("Evaluasi parameter kompetensi dan performa akademik untuk menentukan indeks kesiapan kerja.")

tab_data, tab_profil = st.tabs(["Indikator Akademik", "Kompetensi & Latar Belakang"])

with tab_data:
    col1, col2 = st.columns(2)
    with col1:
        ssc = st.slider("Pencapaian SSC (10th) %", 0, 100, 75)
        hsc = st.slider("Pencapaian HSC (12th) %", 0, 100, 70)
        degree = st.slider("Persentase Kelulusan Degree", 0, 100, 72)
    with col2:
        cgpa = st.number_input("Skala CGPA Kumulatif (0-10)", 0.0, 10.0, 8.0, step=0.01, format="%.2f")
        attendance = st.slider("Rasio Kehadiran (%)", 0, 100, 85)
        entrance = st.slider("Skor Ujian Masuk", 0, 100, 70)

with tab_profil:
    col3, col4 = st.columns(2)
    with col3:
        tech_skill = st.slider("Skor Kompetensi Teknis", 0, 100, 75)
        soft_skill = st.slider("Skor Interpersonal (Soft Skill)", 0, 100, 80)
        gender = st.radio("Identitas Gender", ["Male", "Female"], horizontal=True)
    with col4:
        internship = st.number_input("Jumlah Pengalaman Magang", 0, 20, 1)
        backlogs = st.number_input("Jumlah Mata Kuliah Mengulang", 0, 10, 0)
        extra = st.radio("Aktivitas Ekstrakurikuler", ["Yes", "No"], horizontal=True)

st.markdown("---")
st.header("Analisis Profil Real-Time")

g1, g2, g3 = st.columns(3)

with g1:
    academic_idx = (ssc + hsc + degree + (cgpa*10)) / 4
    fig_academic = go.Figure(go.Indicator(
        mode = "gauge+number", value = academic_idx,
        title = {'text': "Indeks Akademik"},
        gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#00d2d3"}}
    ))
    fig_academic.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_academic, use_container_width=True)

with g2:
    total_comp = tech_skill + soft_skill
    fig_comp = go.Figure(go.Indicator(
        mode = "gauge+number", value = total_comp,
        title = {'text': "Akumulasi Kompetensi"},
        gauge = {'axis': {'range': [None, 200]}, 'bar': {'color': "#ff9f43"}}
    ))
    fig_comp.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_comp, use_container_width=True)

with g3:
    fig_mini_radar = go.Figure(data=go.Scatterpolar(
        r=[ssc, hsc, degree, tech_skill, soft_skill],
        theta=['SSC', 'HSC', 'Degree', 'Tech', 'Soft'],
        fill='toself', line=dict(color='#54a0ff')
    ))
    fig_mini_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=250, margin=dict(l=30, r=30, t=30, b=30))
    st.plotly_chart(fig_mini_radar, use_container_width=True)

st.info(f"Kesiapan Kerja (Readiness Score): {int(academic_idx/10)} Pts")

st.markdown("---")
if st.button("Jalankan Inferensi Model", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        'ssc_percentage': ssc, 'hsc_percentage': hsc, 'degree_percentage': degree, 'cgpa': cgpa, 
        'entrance_exam_score': entrance, 'technical_skill_score': tech_skill, 'soft_skill_score': soft_skill, 
        'internship_count': internship, 'live_projects': 1, 'work_experience_months': 0, 'certifications': 1, 
        'attendance_percentage': attendance, 'backlogs': backlogs, 'gender': 1 if gender == "Male" else 0, 
        'extracurricular_activities': 1 if extra == "Yes" else 0
    }])
    
    res_col1, res_col2 = st.columns([1, 1])
    
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