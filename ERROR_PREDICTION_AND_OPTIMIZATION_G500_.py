import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

start_time = time.time()

# Load the vectorial and simulated data
vectorial_data_path = '/Users/Admin/Desktop/dataset/vectorial sum/C130_Vectorial.csv'
simulated_data_path = '/Users/Admin/Desktop/dataset/vectorial sum/C130_Simulated.csv'

vectorial_data = pd.read_csv(vectorial_data_path)
simulated_data = pd.read_csv(simulated_data_path)

# Correct column names if there are leading/trailing spaces
vectorial_data.columns = vectorial_data.columns.str.strip()
simulated_data.columns = simulated_data.columns.str.strip()

# Merge the data on 'Incident Angle'
merged_data = pd.merge(vectorial_data[['Incident Angle', 'Vectorial Magnitude']],
                       simulated_data[['Incident Angle', 'Simulated Magnitude']],
                       on='Incident Angle', how='inner')

# Plot both the vectorial magnitude and simulated magnitude versus incident angle
plt.plot(merged_data['Incident Angle'], merged_data['Vectorial Magnitude'], label='Vectorial Magnitude')
plt.plot(merged_data['Incident Angle'], merged_data['Simulated Magnitude'], label='Simulated Magnitude')
plt.xlabel('Incident Angle (degrees)')
plt.ylabel('Magnitude')
plt.title('Magnitude vs Incident Angle')
plt.legend()
plt.grid(True)
plt.show()

# Calculate performance metrics
mse = mean_squared_error(merged_data['Simulated Magnitude'], merged_data['Vectorial Magnitude'])
r2 = r2_score(merged_data['Simulated Magnitude'], merged_data['Vectorial Magnitude'])
correlation = merged_data[['Vectorial Magnitude', 'Simulated Magnitude']].corr().iloc[0, 1]

print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Correlation:", correlation)

########### NOW OPTIMZATION OF THE CODE FOR ERROR REDUCTION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the vectorial and simulated data
vectorial_data_path = '/Users/Admin/Desktop/dataset/vectorial sum/C130_Vectorial.csv'
simulated_data_path = '/Users/Admin/Desktop/dataset/vectorial sum/C130_Simulated.csv'

vectorial_data = pd.read_csv(vectorial_data_path)
simulated_data = pd.read_csv(simulated_data_path)

# Correct column names if there are leading/trailing spaces
vectorial_data.columns = vectorial_data.columns.str.strip()
simulated_data.columns = simulated_data.columns.str.strip()

# Merge the data on 'Incident Angle'
merged_data = pd.merge(vectorial_data[['Incident Angle', 'Vectorial Magnitude']],
                       simulated_data[['Incident Angle', 'Simulated Magnitude']],
                       on='Incident Angle', how='inner')

# Extract features and labels
X = merged_data[['Incident Angle', 'Vectorial Magnitude']]
y = merged_data['Simulated Magnitude']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = GradientBoostingRegressor()

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}

# Perform grid search with verbose=0
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)  
grid_search.fit(X_train, y_train)  # Fit the grid search to find the best parameters

# Get the best model
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)


# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("Best Parameters:", grid_search.best_params_)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Sort data for plotting (IMPORTANT)
X_test = X_test.sort_values(by='Incident Angle')  # Ensure correct order

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(X_test['Incident Angle'], y_test, 'bo', label='Actual')
plt.plot(X_test['Incident Angle'], y_pred, 'ro', label='Predicted')
plt.xlabel('Incident Angle (degrees)')
plt.ylabel('Magnitude')
plt.title('Actual vs Predicted Magnitude')
plt.legend()
plt.grid(True)
plt.show()
end_time = time.time()
execution_time = end_time - start_time

print("Execution time:", execution_time, "seconds")

# Create a DataFrame for the plot data
plot_data = pd.DataFrame({
    'Incident Angle': X_test['Incident Angle'],
    'Actual Magnitude': y_test,
    'Predicted Magnitude': y_pred
})

# Save the plot data to a new CSV file
plot_data.to_csv('/Users/Admin/Desktop/dataset/vectorial sum/optimized_plot_data.csv', index=False)

