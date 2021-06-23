from sklearn.datasets import *
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import matplotlib.pyplot as plt

@st.cache
def load_data(df):
    # Сформируем DataFrame
    df_load = pd.DataFrame(data=np.c_[df['data']],
                           columns=df['feature_names'])
    sc = MinMaxScaler()
    df_sc = sc.fit_transform(df.data)
    return df_sc, df.target, df_load.shape[0], df_load

st.header('Выберите датасет')
db = st.radio('',('iris','wine','breast_cancer'))
if db == 'iris':
    df = load_iris()
elif db == 'wine':
    df = load_wine()
elif db == 'breast_cancer':
    df = load_breast_cancer()

st.header('Обучение модели ближайших соседей')

data_load_state = st.text('Загрузка данных...')
data_X, data_Y, data_len, data  = load_data(df)
data_load_state.text('Данные загружены!')

st.write(data.head())

cv_slider = st.slider('Количество фолдов:', min_value=3, max_value=10, value=5, step=1)

#Вычислим количество возможных ближайших соседей
rows_in_one_fold = int(data_len / cv_slider)
allowed_knn = int(rows_in_one_fold * (cv_slider-1))
st.write('Количество строк в наборе данных - {}'.format(data_len))
st.write('Максимальное допустимое количество ближайших соседей с учетом выбранного количества фолдов - {}'.format(allowed_knn))

cv_knn = st.slider('Количество ближайших соседей:', min_value=1, max_value=allowed_knn, value=5, step=1)


scores = cross_val_score(KNeighborsClassifier(n_neighbors=cv_knn),
    data_X, data_Y, scoring='accuracy', cv=cv_slider)


st.subheader('Оценка качества модели')
st.write('Значения accuracy для отдельных фолдов')
st.bar_chart(scores)
st.write('Усредненное значение accuracy по всем фолдам - {}'.format(np.mean(scores)))
