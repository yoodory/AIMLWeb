import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 제목과 설명
st.title("Iris Species Predictor")
st.write("""
This app predicts the Iris flower species based on the input parameters.
You can adjust the parameters using the sliders and see the predicted species.
Additionally, it provides various visualizations of the dataset.
         """)

iris = load_iris()
X = iris.data
y = iris.target   # 0: setosa  1: versicolor  2: virginica

df = pd.DataFrame(X, columns=iris.feature_names) # sepal len, sepal wid, petal len, petal wid
df['species'] = [iris.target_names[i] for i in y]

# 사이드바에서 입력 받기
st.sidebar.header("Input Parameters")
def user_input_features():
    sepal_length = st.sidebar.slider("Sepal length (cm)",
                                    float(df['sepal length (cm)'].min()),
                                    float(df['sepal length (cm)'].max()),
                                    float(df['sepal length (cm)'].mean()), # 최초 슬라이더의 위치
                                    )
    sepal_width = st.sidebar.slider("Sepal width (cm)",
                                    float(df['sepal width (cm)'].min()),
                                    float(df['sepal width (cm)'].max()),
                                    float(df['sepal width (cm)'].mean()), 
                                    )
    petal_length = st.sidebar.slider("Petal length (cm)",
                                    float(df['petal length (cm)'].min()),
                                    float(df['petal length (cm)'].max()),
                                    float(df['petal length (cm)'].mean()), 
                                    )
    petal_width = st.sidebar.slider("Petal width (cm)",
                                    float(df['petal width (cm)'].min()),
                                    float(df['petal width (cm)'].max()),
                                    float(df['petal width (cm)'].mean())
                                    )
    
    data = {'sepal length (cm)':sepal_length,
            'sepal width (cm)':sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)':petal_width,       
            }
    features = pd.DataFrame(data, index=[0])  # 인덱스는 하나
    return features

input_df = user_input_features()

# 사용자 입력 값 표시
st.subheader("User Input Parameters")
st.write(input_df)

# RandomForestClassifier 모델 학습
model = RandomForestClassifier()
model.fit(X, y)  

# 예측
prediction = model.predict(input_df.to_numpy())
prediction_prob = model.predict_proba(input_df.to_numpy())

st.subheader("Prediction")
st.write(iris.target_names[prediction])

st.subheader("Prediction Probability")
st.write(prediction_prob)

# 피쳐 중요도 시각화 
st.subheader('Feature Importance')
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 4))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [iris.feature_names[i] for i in indices])
plt.xlim([-1, X.shape[1]])
st.pyplot(plt.gcf())  # plt.gcf()로 현재 figure 전달

# 히스토그램
st.subheader('Histogram of Features') 
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, ax in enumerate(axes):
    sns.histplot(df[iris.feature_names[i]], kde=True, ax=ax)
    ax.set_title(iris.feature_names[i])
plt.tight_layout()
st.pyplot(fig)

# 상관 행렬
plt.figure(figsize=(10, 8))
st.subheader('Correlation Matrix')
numerical_df = df.drop('species', axis=1)
corr_matrix = numerical_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.tight_layout()
st.pyplot(plt.gcf())

# 페어플롯
st.subheader('Pairplot')
pairplot_fig = sns.pairplot(df, hue="species").fig
st.pyplot(pairplot_fig)
