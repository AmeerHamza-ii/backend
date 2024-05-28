import pandas as pd
from matplotlib import pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Assuming df contains the original DataFrame with all the data

# Step 1: Data Preparation
def train_model():
    df = pd.read_csv('./with_petrol_price.csv')

    categories = df['SERVICE_CATG'].unique()
    models = {}
    mse_values = {}
    y_test_values = {}
    y_train_values = {}
    y_pred_test_values = {}
    y_pred_train_values = {}

    for category in categories:
        category_data = df[df['SERVICE_CATG'] == category].copy()

        # Step 2: Feature Engineering
        # Here you can engineer features like lagged values

        # Step 3: Train-Test Split
        X = category_data[['RATE_VALUE', 'VAT_AMOUNT', 'petrol_price']].values
        y = category_data['VAT_AMOUNT'].values  # Target column is VAT_AMOUNT

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 4: Model Selection
        model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can choose any model

        # Step 5: Model Training
        model.fit(X_train, y_train)

        # Step 6: Model Evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"MSE for {category}: {mse}")

        # Step 7: Prediction
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Save the trained model
        models[category] = model
        mse_values[category] = mse
        y_test_values[category] = y_test.tolist()
        y_train_values[category] = y_train.tolist()
        y_pred_test_values[category] = y_pred_test.tolist()
        y_pred_train_values[category] = y_pred_train.tolist()
            
    return models, mse_values, y_train_values, y_test_values, y_pred_train_values, y_pred_test_values   

if __name__ == '__main__':
    train_model()
