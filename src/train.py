import joblib
import os
from src.data_utils import load_data, split_data
from src.model_pipeline import build_pipeline

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

def get_categorical_features(df, numeric_features):
    return [c for c in df.columns if c not in numeric_features + ['G3']]

def train(data_path: str = 'data/student-mat.csv', model_output: str = None):
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df, target='G3')

    numeric_features = ['age', 'absences', 'G1', 'G2']
    categorical_features = get_categorical_features(df, numeric_features)

    pipeline = build_pipeline(categorical_features, numeric_features)

    pipeline.fit(X_train, y_train)

    if model_output is None:
        model_output = os.path.join(MODEL_DIR, 'student_perf_model.joblib')

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, model_output)
    print(f'Model saved to: {model_output}')

    from sklearn.metrics import mean_absolute_error, r2_score
    preds = pipeline.predict(X_test)
    print('MAE:', mean_absolute_error(y_test, preds))
    print('R2 :', r2_score(y_test, preds))

if __name__ == '__main__':
    train()
