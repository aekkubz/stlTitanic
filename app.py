import streamlit as st
from joblib import load


# st.title('Hello world')

loaded = load("titanic_model.joblib")

st.title('Titanic Survival Predictions')
st.sidebar.title('Menu')
menu = ['Home', 'Predictions']
st.sidebar.selectbox('', menu)

age = st.slider('Age', 0, 100, 30)
sibsp = st.slider('Sibsp', 0, 8, 0)
parch = st.slider('Parch', 0, 6, 0)
fare = st.slider('Fare', 0.0, 512.3292, 7.8292)

btn = st.button('Submit')

if btn:
    input_data = [[age, sibsp, parch, fare]]

    prediction = loaded.predict(input_data)

    predictions_proba = loaded.predict_proba(input_data)

    st.subheader('Prediction')
    if prediction[0] == 1:
        st.write('Survived')
    else:
        st.write('Not Survived')

    st.subheader('Prediction Probability')
    st.write(f'Survived: {predictions_proba[0][1]:.2f}')
    st.write(f'Not Survived: {predictions_proba[0][0]:.2f}')
