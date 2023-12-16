import streamlit as st
from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt

# adding title and text in the app
st.title("House Price Prediction")

st.write(
    """
        Welcome to the House Price Web App
        """
)

dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ("California_Housing Dataset", "Boston Housing Dataset", "House price prediction Dataset", "Choose File"),
)
st.write(dataset_name)

classifier_name = st.sidebar.selectbox(
    "Select Classifier", ("KNN", "SVM", "Random Forest")
)
st.write(classifier_name)

feature_name = st.sidebar.selectbox(
    "Select Feature", ("Lotsize", "Location", "Neighborhood", "Number of Bedroom")
)
st.write(feature_name)

preprocessing_name = st.sidebar.selectbox(
    "Select Preprocessing", ("Missing values", "Outlier treatment", "Drop")
)
st.write(preprocessing_name)


visualization_name = st.sidebar.selectbox(
    "Select Visualization", ("Bar", "Scatter Plot", "Heat Map", "Graphs")
)
st.write(visualization_name)

report_name = st.sidebar.selectbox(
    "Select Report", ("PDF", "HTML", " Image file")
)
st.write(report_name)


def get_datasets(dataset_name):
    if dataset_name == "California_Housing Dataset":
        data = fetch_california_housing()
    elif dataset_name == "House price prediction Dataset":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y


X, y = get_datasets(dataset_name)

st.write("shape of dataset", X.shape)
st.write("number of classes", len(np.unique(y)))
st.write("Generate  and Print PDF Report")

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        return params
    params = add_parameter_ui(classifier_name)


def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=1234,
        )
    return clf

    clf = get_classifier(classifier_name, params)

    # Classification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf.fix(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.write(f"classifier = {classifier_name}")
    st.write(f"accuracy = {acc}")

    # PLOT
    pca = PCA(2)
    X_projected = pca.fit_transform(X)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig = plt.figure()
    plt.scatter(x1, x2, c=y, alpha=0.8, cmap="boston_house")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()

    # show
    st.pyplot("boston_house")

