import streamlit as st
import os, sys

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.append(SRC)

from predict import load_model, predict_from_dict

st.title('Student Performance Predictor')
st.write('Enter student details to predict final grade (G3).')

model = load_model()

with st.form('input_form'):
    age = st.number_input('age', min_value=15, max_value=22, value=17)
    G1 = st.number_input('G1', min_value=0, max_value=20, value=10)
    G2 = st.number_input('G2', min_value=0, max_value=20, value=11)
    absences = st.number_input('absences', min_value=0, max_value=93, value=3)
    sex = st.selectbox('sex', options=['F','M'])
    school = st.selectbox('school', options=['GP','MS'])
    submit = st.form_submit_button('Predict')

if submit:
    data = {
        'age': age,
        'G1': G1,
        'G2': G2,
        'absences': absences,
        'sex': sex,
        'school': school
    }
    pred = predict_from_dict(model, data)
    st.success(f'Predicted final grade (G3): {pred:.2f}')
