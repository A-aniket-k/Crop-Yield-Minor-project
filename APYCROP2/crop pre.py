# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt

# Load dataset and strip whitespace from column names
dataset = pd.read_csv('dataS.csv')
dataset.columns = dataset.columns.str.strip()

# Print the cleaned column names to verify
print("Column names:", dataset.columns.tolist())
print("\nFirst few rows:")
print(dataset.head())

# Use only numerical features for model training
X = dataset[['rainfall', 'temperature', 'wind_speed', 'humidity', 'sunlight']]
y = dataset['yield']  # Now this should work with cleaned column names

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mae = metrics.mean_absolute_error(y_test, predictions)
print('Mean Absolute Error:', mae)

# --- Visualizing the predictions with state and crop names ---

# Merge the predictions with the original dataset for better visualization
test_data = X_test.copy()
test_data['Actual Yield'] = y_test.values
test_data['Predicted Yield'] = predictions

# Add state and crop name for easier reference
test_data['state'] = dataset.loc[y_test.index, 'state'].values
test_data['crop_name'] = dataset.loc[y_test.index, 'crop_name'].values

# Plotting by State
plt.figure(figsize=(12, 6))
for state in test_data['state'].unique():
    state_data = test_data[test_data['state'] == state]
    plt.scatter(state_data['rainfall'], state_data['Actual Yield'], label=f'{state} - Actual', alpha=0.7)
    plt.scatter(state_data['rainfall'], state_data['Predicted Yield'], label=f'{state} - Predicted', alpha=0.7)

plt.xlabel('Rainfall')
plt.ylabel('Crop Yield')
plt.title('Actual vs Predicted Crop Yield by State')
plt.legend()
plt.show()

# Plotting by Crop Name
plt.figure(figsize=(12, 6))
for crop in test_data['crop_name'].unique():
    crop_data = test_data[test_data['crop_name'] == crop]
    plt.scatter(crop_data['temperature'], crop_data['Actual Yield'], label=f'{crop} - Actual', alpha=0.7)
    plt.scatter(crop_data['temperature'], crop_data['Predicted Yield'], label=f'{crop} - Predicted', alpha=0.7)

plt.xlabel('Temperature')
plt.ylabel('Crop Yield')
plt.title('Actual vs Predicted Crop Yield by Crop Name')
plt.legend()
plt.show()

# Additional visualizations for wind_speed, humidity, and sunlight
features = ['wind_speed', 'humidity', 'sunlight']

for feature in features:
    plt.figure(figsize=(12, 6))
    plt.scatter(X_test[feature], y_test, color='black', label='Actual')
    plt.scatter(X_test[feature], predictions, color='blue', label='Predicted')
    plt.xlabel(feature)
    plt.ylabel('Crop Yield')
    plt.title(f'Actual vs Predicted Crop Yield - {feature}')
    plt.legend()
    plt.show()
