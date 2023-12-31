# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
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


# Create a Streamlit app
# Streamlit app
st.title("Housing Price Prediction")

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
