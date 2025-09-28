import joblib
import os
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'student_perf_model.joblib')

def load_model(path: str = MODEL_PATH):
    return joblib.load(path)

def predict_from_dict(model, data: dict):
    df = pd.DataFrame([data])
    preds = model.predict(df)
    return float(preds[0])
