# crop_yield_prediction.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate sample data (replace this with your actual dataset)
np.random.seed(42)
num_samples = 100
temperature = np.random.uniform(20, 35, num_samples)
rainfall = np.random.uniform(50, 200, num_samples)
crop_yield = 3 * temperature + 2 * rainfall + np.random.normal(0, 10, num_samples)

# Create a DataFrame
data = pd.DataFrame({'Temperature': temperature, 'Rainfall': rainfall, 'CropYield': crop_yield})

# Split the data into training and testing sets
X = data[['Temperature', 'Rainfall']]
y = data['CropYield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize the predictions
plt.scatter(X_test['Temperature'], y_test, color='black', label='Actual')
plt.scatter(X_test['Temperature'], y_pred, color='blue', label='Predicted')
plt.xlabel('Temperature')
plt.ylabel('Crop Yield')
plt.legend()
plt.show()

