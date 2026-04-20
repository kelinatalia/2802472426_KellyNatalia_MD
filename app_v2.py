import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Smart Career Analytics (API)", layout="wide")

def create_gauge(value, title, max_val, color):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = value,
        title = {'text': title, 'font': {'size': 20}},
        gauge = {'axis': {'range': [None, max_val]}, 'bar': {'color': color}}
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)')
    return fig

def create_radar(values):
    categories = ['SSC', 'HSC', 'Degree', 'Tech', 'Soft']
    fig = go.Figure(data=go.Scatterpolar(
        r=values, theta=categories, fill='toself', line=dict(color='#4CC9F0')
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=250, margin=dict(l=30, r=30, t=30, b=30))
    return fig

with st.sidebar:
    st.header("Navigasi Sistem")
    
    st.info("""
    **Apa itu Dashboard ini?**

    Alat bantu digital untuk membantu kamu memprediksi peluang diterima kerja berdasarkan data akademik dan keahlian teknis kamu saat ini.
    """)
    
    st.markdown("---")
    
    st.write("**Cara Pakai:**")
    st.caption("1. Geser slider di panel tengah untuk input data.")
    st.caption("2. Lihat perubahan skor kamu secara langsung di bagian grafik.")
    st.caption("3. Klik tombol merah di bawah untuk hasil prediksi final.")
    
    st.markdown("---")
    st.success("Sistem Siap Menganalisa")

st.title("Smart Career Analytics")
st.write("Input data akademik untuk mendapatkan inferensi melalui jalur API.")

tab1, tab2 = st.tabs(["Indikator Akademik", "Kompetensi & Latar Belakang"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        ssc = st.slider("SSC % (10th)", 0, 100, 75)
        hsc = st.slider("HSC % (12th)", 0, 100, 70)
        degree = st.slider("Degree %", 0, 100, 72)
    with col2:
        cgpa = st.number_input("Current CGPA (0-10)", 0.0, 10.0, 8.0, step=0.01, format="%.2f")
        attendance = st.slider("Attendance %", 0, 100, 85)
        entrance = st.slider("Entrance Score", 0, 100, 70)

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        tech_skill = st.slider("Technical Skill Score", 0, 100, 75)
        soft_skill = st.slider("Soft Skill Score", 0, 100, 80)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    with col4:
        internship = st.number_input("Internship Count", 0, 20, 1)
        backlogs = st.number_input("Backlogs", 0, 10, 0)
        extra = st.radio("Extracurricular", ["Yes", "No"], horizontal=True)

user_payload = {
    "ssc_percentage": ssc, "hsc_percentage": hsc, "degree_percentage": degree,
    "cgpa": cgpa, "entrance_exam_score": entrance,
    "technical_skill_score": tech_skill, "soft_skill_score": soft_skill,
    "internship_count": internship, "live_projects": 1, "work_experience_months": 0, "certifications": 1,
    "attendance_percentage": attendance, "backlogs": backlogs,
    "gender": 1 if gender == 'Male' else 0,
    "extracurricular_activities": 1 if extra == 'Yes' else 0
}

st.markdown("---")
st.header("Live Profiling Insights")
g1, g2, g3 = st.columns(3)

with g1:
    academic_idx = (ssc + hsc + degree + (cgpa*10)) / 4
    st.plotly_chart(create_gauge(academic_idx, "Academic Index", 100, "#00d2d3"), use_container_width=True)
with g2:
    st.plotly_chart(create_gauge(tech_skill + soft_skill, "Total Competency", 200, "#ff9f43"), use_container_width=True)
with g3:
    st.plotly_chart(create_radar([ssc, hsc, degree, tech_skill, soft_skill]), use_container_width=True)

st.info(f"Job Readiness: {int(academic_idx/10)} Pts")

st.markdown("---")
if st.button("Predict via API", type="primary", use_container_width=True):
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=user_payload)
        
        if response.status_code == 200:
            result = response.json()
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                if result['placement_status'] == "Placed":
                    st.success("Status Proyeksi: PLACED")
                    st.write("Estimasi Paket Gaji")
                    st.title(f"₹ {result['estimated_salary_lpa']:.2f} LPA")
                    st.balloons()
                else:
                    st.error("Status Proyeksi: NOT PLACED")
                st.caption(f"Confidence Level: {result['confidence']:.2%}")
                
            with res_col2:
                st.write("**Radar Profil Kompetensi**")
                fig_big = go.Figure(data=go.Scatterpolar(
                    r=[ssc, hsc, degree, tech_skill, soft_skill],
                    theta=['SSC', 'HSC', 'Degree', 'Tech', 'Soft'],
                    fill='toself', fillcolor='rgba(0, 210, 211, 0.5)', line=dict(color='#00d2d3')
                ))
                fig_big.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=350)
                st.plotly_chart(fig_big, use_container_width=True)
        else:
            st.error(f"API Error: {response.status_code}")
            
    except Exception as e:
        st.error(f"Koneksi ke Backend Gagal! Pastikan FastAPI menyala. Error: {e}")