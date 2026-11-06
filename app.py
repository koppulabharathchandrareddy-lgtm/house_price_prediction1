
# House Price Prediction using Linear Regression

# Step 1: Import libraries

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# Step 2: Load dataset

data = pd.read_csv("data.csv")

print("Dataset Loaded Successfully")


# Step 3: Select four input columns

X = data[
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot"
    ]
]

# Step 4: Select output column

y = data["price"]


# Step 5: Remove missing values

data_clean = pd.concat([X, y], axis=1)

data_clean = data_clean.dropna()

X = data_clean[
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot"
    ]
]

y = data_clean["price"]


# Step 6: Split training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Step 7: Create Linear Regression model

model = LinearRegression()


# Step 8: Train the model

model.fit(X_train, y_train)

print("Model Training Completed Successfully")


# Step 9: Test the model

y_pred = model.predict(X_test)


# Step 10: Check model performance

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)

print("R2 Score:", r2)


# Step 11: Save model in the same folder

folder_path = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    folder_path,
    "house_price_model.pkl"
)

joblib.dump(model, model_path)

print("Model Saved Successfully")

print("Model Location:", model_path)


# Step 12: Test with new house details

new_house = [[3, 2, 1800, 5000]]

predicted_price = model.predict(new_house)

print("Predicted House Price:", round(predicted_price[0], 2))
