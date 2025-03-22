# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv('data.csv')

# Remove leading and trailing spaces from column names
dataset.columns = dataset.columns.str.strip()

# Features and target variable
X = dataset[['State', 'District', 'Crop', 'Crop_Year', 'Season', 'Area', 'Production']]
y = dataset['Yield']

# One-hot encode categorical features
X_encoded = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create a Random Forest regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mae = metrics.mean_absolute_error(y_test, predictions)
print('Mean Absolute Error:', mae)

# Combine original features with actual and predicted values
X_test_original = dataset.iloc[y_test.index][['State', 'District', 'Crop', 'Crop_Year', 'Season', 'Area', 'Production']]
results = X_test_original.copy()
results['Actual Yield'] = y_test.values
results['Predicted Yield'] = predictions

# Display the combined results
print("\nActual vs Predicted Yield with All Features:")
print(results)

# Save the results to a CSV file
results.to_csv('actual_vs_predicted_yield.csv', index=False)
print("\nResults saved to 'actual_vs_predicted_yield.csv'")

# Visualize the actual vs predicted yield
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, color='blue', label='Predicted Yield')   # Predicted values in blue
plt.scatter(y_test, y_test, color='red', label='Actual Yield')            # Actual values in red
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Actual vs Predicted Yield')
plt.legend()
plt.grid(True)
plt.show()
