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

# Assume multiple factors as input features, 'Yield' as the target variable
X = dataset[['State', 'District', 'Crop', 'Crop_Year', 'Season', 'Area', 'Production']]
y = dataset['Yield']

# One-hot encode categorical features using pd.get_dummies()
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

# Visualize the predictions
plt.scatter(y_test, predictions, color='blue')
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Actual vs Predicted Yield')
plt.show()
