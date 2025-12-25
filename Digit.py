import pandas as p
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import streamlit as st
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Digit Recognition", layout="centered")
st.title("✍️ Handwritten Digit Recognition")
st.write("digit recognition using pixel data")


# load dataset 
@st.cache_data
def load_data():
    return p.read_csv("digit.csv")

Dataset = load_data()

x = Dataset.iloc[:, 1:].values
y = Dataset.iloc[:, 0].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0
)

#train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# accuracy of the model
accuracy = accuracy_score(y_test, y_pred) * 100
st.success(f"Model Accuracy: {accuracy:.2f}%")

st.subheader("🔍 Visualize a Test Digit")

index = st.number_input(
    "Enter an index",
    min_value=0,
    max_value=len(x_test) - 1,
    step=1
)

if st.button("Predict Digit"):
    prediction = model.predict(x_test)[index]
    st.write(f"### Predicted Digit: **{prediction}**")

    fig, ax = plt.subplots()
    ax.imshow(x_test[index].reshape(28, 28), cmap="gray")
    ax.axis("off")
    st.pyplot(fig)