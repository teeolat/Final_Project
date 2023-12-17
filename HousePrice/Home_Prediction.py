# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st


# Load and explore the dataset
data = pd.read_csv('data/AmesHousing.csv')

# Explore the dataset
print(data.head())


# Checking for the info
print(data.info())

# Describing the data
print(data.describe())


# Preprocess the data
# Handle missing values and encode categorical variables
data.fillna(0, inplace=True)  # Replace missing values with 0 (you may need a more sophisticated approach)
data = pd.get_dummies(data)

# Split the data into features (X) and target variable (y)
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error: {mse}")


# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
st.write(f"Mean Absolute Error: {mae}")

# Create a Streamlit app
# Streamlit app
st.title("Housing Price Prediction")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Ames Housing Dataset", "California Housing Dataset", "Choose File"))
st.write(dataset_name)

feature_name = st.sidebar.selectbox("Select Feature", ("LotArea", "Location", "Neighborhood", "Number of Bedroom"))
st.write(feature_name)

preprocessing_name = st.sidebar.selectbox("Select Preprocessing", ("Missing values", "Outlier treatment", "Drop"))
st.write(preprocessing_name)

model_name = st.sidebar.selectbox("Select Model", ("KNN", "Decision Trees", "Gradient Boosting Regressor", "Random Forest"))
st.write(model_name)

# Sidebar with Visualization Type
visualization_name = st.sidebar.selectbox("Select Visualization", ("Bar", "Scatter Plot", "Heat Map", "Graphs"))
st.write(visualization_name)

# Sidebar foe Report selection
report_name = st.sidebar.selectbox("Select Report", ("PDF", "HTML", " Price Prediction"))
st.write(report_name)

# Sidebar with input features
st.sidebar.header("Input Features")

# Collect input features from the user
input_features = {}
for feature in X.columns:
    input_features[feature] = st.sidebar.slider(f"Select {feature}", float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))

# Convert input features to DataFrame
input_df = pd.DataFrame([input_features])

# Make predictions using the model
prediction = model.predict(input_df)

# Display the prediction
st.write(f"Predicted Price: ${prediction[0]:,.2f}")
