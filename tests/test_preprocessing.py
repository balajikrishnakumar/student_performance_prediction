from src.data_utils import load_data, split_data

def test_load_and_split():
    df = load_data('data/student-mat.csv')
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) > 0
    assert len(X_test) > 0
