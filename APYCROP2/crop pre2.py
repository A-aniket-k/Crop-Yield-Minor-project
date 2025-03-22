# crop_yield_prediction.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate sample data (replace this with your actual dataset)
np.random.seed(42)
num_samples = 100
temperature = np.random.uniform(20, 35, num_samples)
rainfall = np.random.uniform(50, 200, num_samples)
sunlight = np.random.uniform(5, 10, num_samples)
humidity = np.random.uniform(30, 80, num_samples)

crop_yield = 3 * temperature + 2 * rainfall + 1.5 * sunlight + 0.5 * humidity + np.random.normal(0, 10, num_samples)

# Create a DataFrame
data = pd.DataFrame({
    'Temperature': temperature,
    'Rainfall': rainfall,
    'Sunlight': sunlight,
    'Humidity': humidity,
    'CropYield': crop_yield
})

# Split the data into training and testing sets
X = data[['Temperature', 'Rainfall', 'Sunlight', 'Humidity']]
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

# Visualize the predictions in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_test['Temperature'], X_test['Humidity'], y_test, color='black', label='Actual', marker='o', alpha=0.5)
ax.scatter(X_test['Temperature'], X_test['Humidity'], y_pred, color='blue', label='Predicted', marker='x', alpha=0.5)

ax.set_xlabel('Temperature')
ax.set_ylabel('Humidity')
ax.set_zlabel('Crop Yield')
ax.set_title('Actual vs. Predicted Crop Yield')

plt.legend()
plt.show()

# Visualize the predictions in 3D for Rainfall
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_test['Temperature'], X_test['Rainfall'], y_test, color='black', label='Actual', marker='o', alpha=0.5)
ax.scatter(X_test['Temperature'], X_test['Rainfall'], y_pred, color='blue', label='Predicted', marker='x', alpha=0.5)

ax.set_xlabel('Temperature')
ax.set_ylabel('Rainfall')
ax.set_zlabel('Crop Yield')
ax.set_title('Actual vs. Predicted Crop Yield (Rainfall)')

plt.legend()
plt.show()

# Visualize the predictions in 3D for Sunlight and Rainfall
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_test['Temperature'], X_test['Sunlight'], y_test, color='black', label='Actual', marker='o', alpha=0.5)
ax.scatter(X_test['Temperature'], X_test['Sunlight'], y_pred, color='blue', label='Predicted', marker='x', alpha=0.5)

ax.set_xlabel('Temperature')
ax.set_ylabel('Sunlight')
ax.set_zlabel('Crop Yield')
ax.set_title('Actual vs. Predicted Crop Yield (Sunlight)')

plt.legend()
plt.show()