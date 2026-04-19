import pandas as pd
import numpy as np
import pickle
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, r2_score

# data ingestion
def loadData(fileName):
    # load dataset 
    df = pd.read_csv(fileName)
    
    # drop student_id as it has no predictive value
    if 'student_id' in df.columns:
        df = df.drop(columns=['student_id'])
    
    # convert categorical yes/no to numeric (manual encoding)
    if 'extracurricular_activities' in df.columns:
        df['extracurricular_activities'] = df['extracurricular_activities'].map({'Yes': 1, 'No': 0})
    
    # convert gender to numeric (Male: 1, Female: 0)
    # we do this here so the pipeline only deals with numbers
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
        
    return df

colsToScale = ['ssc_percentage', 'hsc_percentage', 'degree_percentage', 'cgpa', 
               'entrance_exam_score', 'technical_skill_score', 'soft_skill_score', 
               'internship_count', 'live_projects', 'work_experience_months', 
               'certifications', 'attendance_percentage', 'backlogs']

# end-to-end pipeline 
def buildPipeline(modelType='classification'):
    # pisahin mana yang di scale dan mana yang tidak
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), colsToScale)
        ],
        remainder='passthrough' # gender dan extracurricular tetap 0/1 (tidak di-scale)
    )

    if modelType == 'classification':
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    else:
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)

    # ini yang menjamin tidak ada data leakage
    fullPipeline = Pipeline(steps=[
        ('prep', preprocessor),
        ('model', model)
    ])
    
    return fullPipeline

# experiment tracking 
def runExperiment():
    # load data
    df = loadData('B.csv')
    featureCols = colsToScale + ['gender', 'extracurricular_activities']
    X = df[featureCols]
    
    xTrainC, xTestC, yTrainC, yTestC = train_test_split(X, df['placement_status'], test_size=0.2, random_state=42)
    xTrainR, xTestR, yTrainR, yTestR = train_test_split(X, df['salary_package_lpa'], test_size=0.2, random_state=42)

    mlflow.set_experiment("Student_Placement")

    with mlflow.start_run(run_name="Final_Model"):
        # model klasifikasi
        clfPipe = buildPipeline(modelType='classification')
        clfPipe.fit(xTrainC, yTrainC)
        f1 = f1_score(yTestC, clfPipe.predict(xTestC), average='weighted')

        # log param & metric ke MLflow
        mlflow.log_param("clf_model_type", "MLP")
        mlflow.log_metric("clf_f1_score", f1)
        # simpan model sebagai artifact di mlflow
        mlflow.sklearn.log_model(clfPipe, "placement_pipeline_model")

        # model regresi
        regPipe = buildPipeline(modelType='regression')
        regPipe.fit(xTrainR, yTrainR)
        r2 = r2_score(yTestR, regPipe.predict(xTestR))

        # log param & metric ke MLflow
        mlflow.log_param("reg_model_type", "GradientBoosting")
        mlflow.log_metric("reg_r2_score", r2)
        # simpan model sebagai artifact di mlflow
        mlflow.sklearn.log_model(regPipe, "salary_pipeline_model")

        # persistence
        with open('classificationModel.pkl', 'wb') as f:
            pickle.dump(clfPipe, f)

        with open('regressionModel.pkl', 'wb') as f:
            pickle.dump(regPipe, f)

        print(f"F1-Score: {f1:.4f} | R2-Score: {r2:.4f}")


if __name__ == "__main__":
    runExperiment()