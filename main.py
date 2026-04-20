from fastapi import FastAPI
import pandas as pd
import pickle
from pydantic import BaseModel

app = FastAPI(title="Student Placement API")

# load model pipeline
with open('classificationModel.pkl', 'rb') as f:
    clf_model = pickle.load(f)
with open('regressionModel.pkl', 'rb') as f:
    reg_model = pickle.load(f)

# struktur data input
class StudentData(BaseModel):
    ssc_percentage: float = 80.0
    hsc_percentage: float = 80.0
    degree_percentage: float = 80.0
    cgpa: float
    entrance_exam_score: float = 80.0
    technical_skill_score: float
    soft_skill_score: float
    internship_count: int
    live_projects: int = 1
    work_experience_months: int = 0
    certifications: int = 1
    attendance_percentage: float = 90.0
    backlogs: int
    gender: int # 1: Male, 0: Female
    extracurricular_activities: int # 1: Yes, 0: No

@app.get("/")
def home():
    return {"message": "Student Placement API is Running"}

@app.post("/predict")
def predict(data: StudentData):
    # Convert input ke DataFrame 
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])
    
    # Prediksi Klasifikasi
    status = int(clf_model.predict(df)[0])
    prob = float(clf_model.predict_proba(df)[0][1])
    
    # Prediksi Regresi (hanya jika placed)
    salary = 0.0
    if status == 1:
        salary = float(reg_model.predict(df)[0])
        
    return {
        "placement_status": "Placed" if status == 1 else "Not Placed",
        "confidence": prob,
        "estimated_salary_lpa": salary
    }