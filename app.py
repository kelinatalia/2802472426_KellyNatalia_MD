import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="Career Analytics", layout="wide")

@st.cache_resource
def load_models():
    with open('classificationModel.pkl', 'rb') as f:
        clf_model = pickle.load(f) 
    with open('regressionModel.pkl', 'rb') as f:
        reg_model = pickle.load(f)
    return clf_model, reg_model

clf_model, reg_model = load_models()

with st.sidebar:
    st.title("Settings")
    st.markdown("---")
    st.write("Aplikasi ini menggunakan dual-model sistem untuk memprediksi probabilitas penempatan kerja dan estimasi remunerasi.")
    st.info("Pastikan input data sesuai dengan transkrip akademik terbaru.")

st.title("Career Readiness Dashboard")
st.markdown("---")

col_input, col_viz = st.columns([1, 1.2])

with col_input:
    st.subheader("Personal Metrics")
    t1, t2 = st.tabs(["Akademik", "Skill & Ekstra"])
    
    with t1:
        ssc = st.slider("Nilai SSC (10th) %", 0, 100, 85)
        hsc = st.slider("Nilai HSC (12th) %", 0, 100, 78)
        degree = st.slider("Degree Percentage", 0, 100, 80)
        cgpa = st.number_input("CGPA Actual (0-10)", 0.0, 10.0, 3.8, step=0.01)
        attendance = st.slider("Attendance Rate (%)", 0, 100, 95)
        
    with t2:
        tech_skill = st.slider("Programming Skill", 0, 100, 88)
        soft_skill = st.slider("Communication Score", 0, 100, 85)
        internship = st.number_input("Internship Experience", 0, 5, 2)
        entrance = st.slider("Entrance Test Score", 0, 100, 72)
        gender = st.selectbox("Gender Identification", ["Male", "Female"])
        extra = st.radio("Extracurricular Activities", ["Yes", "No"], horizontal=True)
        backlogs = st.number_input("Total Backlogs", 0, 5, 0)

with col_viz:
    st.subheader("Live Profile Analysis")
    
    # Donut Chart untuk Academic Index (Pengganti Spedometer)
    academic_idx = (ssc + hsc + degree + (cgpa*10)) / 4
    fig_donut = go.Figure(go.Pie(
        values=[academic_idx, 100-academic_idx],
        labels=["Score", "Remaining"],
        hole=.7,
        marker_colors=['#6c5ce7', '#dfe6e9'],
        textinfo='none',
        showlegend=False
    ))
    fig_donut.add_annotation(text=f"{int(academic_idx)}%", font_size=40, showarrow=False, font_color="#6c5ce7")
    fig_donut.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
    
    st.plotly_chart(fig_donut, use_container_width=True)
    
    st.markdown("---")
    
    if st.button("RUN PREDICTION", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            'ssc_percentage': ssc, 'hsc_percentage': hsc, 'degree_percentage': degree, 'cgpa': cgpa, 
            'entrance_exam_score': entrance, 'technical_skill_score': tech_skill, 'soft_skill_score': soft_skill, 
            'internship_count': internship, 'live_projects': 1, 'work_experience_months': 0, 'certifications': 1, 
            'attendance_percentage': attendance, 'backlogs': backlogs, 'gender': 1 if gender == "Male" else 0, 
            'extracurricular_activities': 1 if extra == "Yes" else 0
        }])
        
        # Hasil muncul di bawah donut
        pred = clf_model.predict(input_df)[0]
        
        c1, c2 = st.columns(2)
        with c1:
            # Radar Chart muncul di sini bareng hasil
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=[ssc, hsc, degree, tech_skill, soft_skill],
                theta=['SSC', 'HSC', 'Degree', 'Tech', 'Soft'],
                fill='toself', fillcolor='rgba(108, 92, 231, 0.2)', line=dict(color='#6c5ce7')
            ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 100])), height=250, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with c2:
            st.write("### Result:")
            if pred == 1:
                st.success("STATUS: PLACED")
                salary = reg_model.predict(input_df)[0]
                st.metric("Salary Estimation (LPA)", f"₹ {salary:.2f}")
                st.balloons()
            else:
                st.error("STATUS: NOT PLACED")
                st.caption("Fokus pada peningkatan skor teknis dan sertifikasi.")