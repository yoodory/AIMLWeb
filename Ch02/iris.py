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
Additionally, it provides various visualization of the dataset.
         """)

iris = load_iris()
X = iris.data
y = iris.target   # 0: setosa  1: vesicolor  2: virginica
# print(x)
# print(y)

df = pd.DataFrame(X, columns=iris.feature_names) # sepal len, sepal wid, petal len, petal wid, 
df['species'] = [iris.target_names[i] for i in y]
# print(df)

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
    features = pd.DataFrame(data, index = [0])  # 인덱스는 하나
    return features

input_df = user_input_features()
# print(input_df) 

# 사용자 입력 값 표시
st.subheader("User Input Parameters")
st.write(input_df)

#  RandomForestClassifier 모델 학습 / 결정트리를 여러 개 생성 후 -> 다수결로 
model = RandomForestClassifier()
model.fit(X, y)  

# 예측
prediction = model.predict(input_df.to_numpy())
# print(prediction, iris.target_names)
prediction_prob = model.predict_proba(input_df.to_numpy())
# print(type(prediction_prob))
st.subheader("Prediction")
st.write(iris.target_names[prediction])

st.subheader("Prediction Probability")
st.write(prediction_prob)

# 피쳐 중요도 시각화 
st.subheader('Feature Importance')
importances = model.feature_importances_
# print(importances)
indices = np.argsort(importances)[::-1] # 특성의 중요도에 따라 ..내림차순의 인덱스
# print(indices)

plt.figure(figsize = (10, 4))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]),importances[indices], align = "center") 
 # 0~3까지
plt.xticks(range(X.shape[1]), [iris.feature_names[i] for i in indices])
# 0,1,2,3 을 featured의 이름으로
# print(iris.feature_names)
plt.xlim([-1, X.shape[1]]) # -1부터 4까지
st.pyplot(plt)

# 히스토그램
st.subheader('Histogram of Features') 
fig, axes = plt.subplots(2, 2, figsize = (12, 8))  # 2행2열로 분할
axes = axes.flatten()   #  4개가 한 줄로 펼쳐짐
for i, ax in enumerate(axes):   
    sns.histplot(df[iris.feature_names[i]], kde=True, ax=ax)
    ax.set_title(iris.feature_names[i])
plt.tight_layout()
st.pyplot(fig)

# 상관 행렬
plt.figure(figsize = (10, 8))
st.subheader('Correlation Matrix')
numerical_df = df.drop('species', axis = 1) # 축은 열
corr_matrix = numerical_df.corr()  # 상관관계
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5 )
#                        표시 안하면 숫자 표시 안됨
plt.tight_layout()    # 상관관계가 높은 것: 빨강 / 낮은 것: 파랑
st.pyplot(plt)


# 페어플롯
st.subheader('Pairplot')
fig = sns.pairplot(df, hue="species").fig  # hue: 종별로 색상을 다르게
plt.tight_layout()
st.pyplot(fig)